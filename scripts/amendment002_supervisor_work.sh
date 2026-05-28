#!/usr/bin/env bash
# Amendment-002 supervisor worker: sync, build, diagnose, log — runs every loop tick.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
STATE="$OPT_DIR/amendment002-supervisor-state.json"
WORK_LOG="$OPT_DIR/amendment002-supervisor-work.log"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PSNR_FLOOR="${AMEND002_PSNR_FLOOR:-47.28}"
TT_VS_CPU_FLOOR="${AMEND002_TT_VS_CPU_FLOOR:-45.0}"

echo "[$TS] work_start" | tee -a "$WORK_LOG"

PHASE="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['phase'])")"
echo "[$TS] phase=$PHASE" | tee -a "$WORK_LOG"

# 1. Sync Mac -> bh-30
bash "$REPO_ROOT/scripts/sync_to_bh30.sh" 2>&1 | tee -a "$WORK_LOG"

# 2. Remote build + prep
ssh -o ConnectTimeout=30 "$REMOTE" bash -s <<'REMOTE' 2>&1 | tee -a "$WORK_LOG"
set -euo pipefail
cd /localdev/smarton/gstt2
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache

chmod +x scripts/*.sh 2>/dev/null || true
pkill -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true
bash scripts/fix_tt_metal_cmake_exports.sh

source .venv/bin/activate 2>/dev/null || true
pip install -q torch numpy 2>/dev/null || true

echo "[work] building GSPLAT_WITH_TT..."
GSPLAT_WITH_TT=ON GSPLAT_SIMD_AVX2=ON BUILD_DIR=build-tt bash scripts/build_cpu_cpp.sh
REMOTE

# 3. Diagnose blend quality
DIAG_JSON="$(mktemp)"
if ! bash "$REPO_ROOT/scripts/amendment002_diagnose_blend.sh" >"$DIAG_JSON" 2>>"$WORK_LOG"; then
  echo "[work] diagnose script failed" | tee -a "$WORK_LOG"
  echo '{"has_tt_support": false}' >"$DIAG_JSON"
fi
if ! "$LOCAL_PY" -c "import json; json.load(open('$DIAG_JSON'))" 2>/dev/null; then
  echo "[work] diagnose JSON invalid, skipping state merge" | tee -a "$WORK_LOG"
  rm -f "$DIAG_JSON"
  bash "$REPO_ROOT/scripts/amendment002_supervisor_tick.sh" 2>&1 | tee -a "$WORK_LOG"
  echo "[$TS] work_done (partial)" | tee -a "$WORK_LOG"
  exit 0
fi
echo "[work] diagnose:" | tee -a "$WORK_LOG"
cat "$DIAG_JSON" | tee -a "$WORK_LOG"

# 4. Update state + metal-iters from diagnose
"$LOCAL_PY" - "$STATE" "$DIAG_JSON" "$PSNR_FLOOR" "$TT_VS_CPU_FLOOR" <<'PY' | tee -a "$WORK_LOG"
import json, sys
from datetime import datetime, timezone

state_path, diag_path = sys.argv[1:3]
psnr_floor = float(sys.argv[3])
tt_vs_cpu_floor = float(sys.argv[4])

state = json.load(open(state_path))
diag = json.load(open(diag_path))
gates = state.setdefault("last_gates", {})
gates.update({
    "has_tt_support": diag.get("has_tt_support"),
    "hero_psnr_tt_dB": diag.get("psnr_tt_vs_ref_dB"),
    "psnr_tt_vs_cpu_cpp_mb_dB": diag.get("psnr_tt_vs_cpu_cpp_mb_dB"),
    "psnr_cpu_cpp_mb_vs_ref_dB": diag.get("psnr_cpu_cpp_mb_vs_ref_dB"),
    "device_kernel_ms": diag.get("device_kernel_ms"),
    "max_abs_tt_vs_cpu": diag.get("max_abs_tt_vs_cpu"),
})
state["updated_at"] = datetime.now(timezone.utc).isoformat()

tt_vs_cpu = diag.get("psnr_tt_vs_cpu_cpp_mb_dB")
tt_vs_ref = diag.get("psnr_tt_vs_ref_dB")
layer2 = tt_vs_cpu is not None and tt_vs_cpu >= tt_vs_cpu_floor

milestones = state.setdefault("milestones", {})
if diag.get("has_tt_support"):
    milestones["tt000_ttbackend"] = True
    milestones["tt001a_inprocess_blend"] = layer2

if not diag.get("has_tt_support"):
    state["phase"] = "tt-001a-inprocess-blend"
    state["next_action"] = "BLOCKED: has_tt_support=false after build. Fix TtMetalInTree.cmake / build-tt."
elif not layer2:
    state["phase"] = "tt-001a-inprocess-blend"
    state["next_action"] = (
        f"tt-001a: TT vs cpu_cpp_mb PSNR={tt_vs_cpu:.2f}dB (need >={tt_vs_cpu_floor}). "
        f"TT vs ref={tt_vs_ref:.2f}dB. Align prepare_kernel_inputs payload with cpu_cpp_mb blend; "
        "then tt-001b 4x8 microblock kernel for perf."
    )
else:
    state["phase"] = "tt-001b-microblock"
    state["next_action"] = (
        f"tt-001a PASS: TT vs cpu_cpp_mb PSNR={tt_vs_cpu:.2f}dB. "
        "Port tt-001b 4x8 microblock kernel; drive hero sum_total_ms toward 1ms/view."
    )

json.dump(state, open(state_path, "w"), indent=2)
print("PHASE=" + state["phase"])
print("NEXT_ACTION=" + state["next_action"])
print("TT_VS_CPU_PSNR=" + str(tt_vs_cpu))
print("LAYER2_PASS=" + str(layer2))
PY

rm -f "$DIAG_JSON"

# 5. Re-measure gates (updates render timing etc.)
bash "$REPO_ROOT/scripts/amendment002_supervisor_tick.sh" 2>&1 | tee -a "$WORK_LOG"

echo "[$TS] work_done" | tee -a "$WORK_LOG"
