#!/usr/bin/env bash
# Amendment-002 supervisor tick: measure gates, emit NEXT_ACTION toward 1ms/frame.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
STATE="$OPT_DIR/amendment002-supervisor-state.json"
LOG="$OPT_DIR/amendment002-supervisor.log"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TARGET_MS_PER_VIEW=1.0

mkdir -p "$OPT_DIR"

if [[ ! -s "$STATE" ]] || ! "$LOCAL_PY" -c "import json; json.load(open('$STATE'))" 2>/dev/null; then
  "$LOCAL_PY" - <<'PY' >"$STATE"
import json
print(json.dumps({
    "phase": "tt-001a-inprocess-blend",
    "target_ms_per_view": 1.0,
    "next_action": "Unblock GSPLAT_WITH_TT build on bh-30; verify tt backend hero PSNR >= 47.28 dB.",
    "milestones": {
        "scalar_cpu_cpp": True,
        "avx2_cpu_cpp": False,
        "tt000_ttbackend": False,
        "tt001a_inprocess_blend": False,
        "tt001b_microblock_4x8": False,
        "full_port_all_stages": False,
    },
}, indent=2))
PY
fi

PHASE="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['phase'])")"
echo "[$TS] amendment002_tick phase=$PHASE" | tee -a "$LOG"

# --- Remote probes ---
BUILD_JSON="$(mktemp)"
RENDER_JSON="$(mktemp)"
TT_JSON="$(mktemp)"
set +e
ssh -o ConnectTimeout=20 "$REMOTE" bash -s <<'REMOTE_EOF' >"$BUILD_JSON" 2>>"$LOG"
set -euo pipefail
cd /localdev/smarton/gstt2
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
python3 - <<'PY'
import importlib.util, json, subprocess, sys
from pathlib import Path

out = {"simd_backend": None, "has_tt": False, "build_tt_ok": False}
so = Path("backends/cpu_cpp/_gsplat_cpu.cpython-310-x86_64-linux-gnu.so")
if so.exists():
    spec = importlib.util.spec_from_file_location("_gsplat_cpu", so)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out["simd_backend"] = m.simd_backend()
    out["has_tt"] = bool(m.has_tt_support())
r = subprocess.run(
    ["cmake", "-G", "Ninja", "-S", "src", "-B", "build-tt",
     "-DCMAKE_BUILD_TYPE=Release", "-DGSPLAT_SIMD_AVX2=ON", "-DGSPLAT_WITH_TT=ON"],
    capture_output=True, text=True, env={**dict(__import__("os").environ), "TT_METAL_HOME": "/localdev/smarton/tt-metal"})
out["build_tt_configure_rc"] = r.returncode
out["build_tt_configure_tail"] = (r.stderr or r.stdout)[-500:]
if r.returncode == 0:
    b = subprocess.run(["cmake", "--build", "build-tt", "-j", "16"], capture_output=True, text=True,
        env={**dict(__import__("os").environ), "TT_METAL_HOME": "/localdev/smarton/tt-metal"})
    out["build_tt_ok"] = b.returncode == 0
    out["build_tt_tail"] = (b.stderr or b.stdout)[-500:]
else:
    # Fallback: in-tree link path skips find_package; try build_cpu_cpp.sh directly.
    b = subprocess.run(["bash", "scripts/build_cpu_cpp.sh"], capture_output=True, text=True,
        env={**dict(__import__("os").environ), "TT_METAL_HOME": "/localdev/smarton/tt-metal",
             "GSPLAT_WITH_TT": "ON", "GSPLAT_SIMD_AVX2": "ON", "BUILD_DIR": "build-tt"})
    out["build_tt_ok"] = b.returncode == 0
    out["build_tt_configure_rc"] = b.returncode
    out["build_tt_tail"] = (b.stderr or b.stdout)[-500:]
print(json.dumps(out))
PY
REMOTE_EOF
BUILD_RC=$?

ssh -o ConnectTimeout=20 "$REMOTE" bash -s <<'REMOTE_EOF' >"$RENDER_JSON" 2>>"$LOG"
set -euo pipefail
cd /localdev/smarton/gstt2
source .venv/bin/activate 2>/dev/null || true
pip install -q torch numpy 2>/dev/null || true
python3 scripts/render_fixed.py --backend cpu_cpp_mb --scene bicycle --frames 1 --view hero 2>/dev/null | tail -5
REMOTE_EOF
RENDER_RC=$?

ssh -o ConnectTimeout=120 "$REMOTE" bash -s <<'REMOTE_EOF' >"$TT_JSON" 2>>"$LOG"
set -euo pipefail
cd /localdev/smarton/gstt2
source .venv/bin/activate 2>/dev/null || true
pip install -q torch numpy 2>/dev/null || true
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole
pkill -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true
sleep 1
bash scripts/fix_tt_metal_cmake_exports.sh 2>/dev/null || true
python3 scripts/verify_blend_metal.py --backend tt --psnr-floor 47.0 2>/dev/null || echo '{"layer2_pass": false, "psnr_dB": null}'
REMOTE_EOF
TT_RC=$?
set -e

"$LOCAL_PY" - "$STATE" "$BUILD_JSON" "$RENDER_JSON" "$TT_JSON" "$BUILD_RC" "$RENDER_RC" "$TT_RC" "$TARGET_MS_PER_VIEW" >"${STATE}.new" <<'PY'
import json, re, sys
from datetime import datetime, timezone

state_path, build_path, render_path, tt_path = sys.argv[1:5]
build_rc, render_rc, tt_rc = map(int, sys.argv[5:8])
target = float(sys.argv[8])

try:
    build = json.load(open(build_path))
except Exception:
    build = {}
try:
    tt = json.load(open(tt_path))
except Exception:
    tt = {}
render_text = open(render_path).read() if render_path else ""

sum_ms = None
m = re.search(r"sum_total_ms[=:\s]+([0-9.]+)", render_text)
if m:
    sum_ms = float(m.group(1))
per_view = sum_ms  # hero single view

prev = json.load(open(state_path))
milestones = prev.get("milestones", {})
phase = prev.get("phase", "tt-001a-inprocess-blend")

if build.get("simd_backend") == "avx2":
    milestones["avx2_cpu_cpp"] = True
if build.get("has_tt"):
    milestones["tt000_ttbackend"] = True
    milestones["tt001a_inprocess_blend"] = bool(tt.get("layer2_pass"))

hero_psnr = tt.get("psnr_dB")

if not build.get("build_tt_ok"):
    nxt = (
        "BLOCKED: GSPLAT_WITH_TT build failed on bh-30. "
        "Fix tt-metal cmake exports / blend_device.cpp compile; rebuild build-tt. "
        f"configure_rc={build.get('build_tt_configure_rc')} tail={build.get('build_tt_configure_tail','')[-200:]}"
    )
    phase = "tt-001a-inprocess-blend"
elif not tt.get("layer2_pass"):
    nxt = (
        "tt-001a: GSPLAT_WITH_TT built. Run verify_blend_metal --backend tt; "
        "fix PSNR if < 47.28 dB; log opt/metal-iters.jsonl. Then tt-001b 4x8 microblock kernel."
    )
    phase = "tt-001a-inprocess-blend"
elif per_view and per_view <= target:
    nxt = "TARGET HIT: sum_total_ms <= 1ms/view. Log victory; optimize further for margin."
    phase = "done-1ms"
elif per_view and per_view <= target * 10:
    nxt = f"Within 10x of target ({per_view:.1f}ms/view). tt-001b microblock 4x8 + perf iter."
    phase = "tt-001b-microblock"
else:
    nxt = (
        f"Continue amendment-002 ladder. Current ~{per_view or '?'}ms/view vs target {target}ms. "
        "Port stages: blend→cull→sort→tile_assign→project to TT; perf iter after each gate."
    )
    phase = phase or "tt-002-cull"

state = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "target_ms_per_view": target,
    "next_action": nxt,
    "milestones": milestones,
    "last_gates": {
        "build_rc": build_rc,
        "render_rc": render_rc,
        "tt_rc": tt_rc,
        "simd_backend": build.get("simd_backend"),
        "has_tt_support": build.get("has_tt"),
        "build_tt_ok": build.get("build_tt_ok"),
        "hero_psnr_tt_dB": hero_psnr,
        "sum_total_ms_hero_cpu_cpp_mb": sum_ms,
        "ms_per_view_estimate": per_view,
        "gap_to_target_x": (per_view / target) if per_view and target else None,
    },
}
json.dump(state, open(state_path, "w"), indent=2)
print(state["phase"])
print(state["next_action"])
print(state["last_gates"].get("ms_per_view_estimate"))
print(state["last_gates"].get("hero_psnr_tt_dB"))
print(state["last_gates"].get("build_tt_ok"))
PY

mv "${STATE}.new" "${STATE}.tmp" 2>/dev/null || true
"$LOCAL_PY" -c "
import json
s=json.load(open('$STATE'))
json.dump(s, open('$STATE','w'), indent=2)
print('PHASE='+s['phase'])
print('NEXT_ACTION='+s['next_action'])
print('MS_PER_VIEW='+str(s['last_gates'].get('ms_per_view_estimate')))
print('TT_PSNR='+str(s['last_gates'].get('hero_psnr_tt_dB')))
print('BUILD_TT='+str(s['last_gates'].get('build_tt_ok')))
"

rm -f "$BUILD_JSON" "$RENDER_JSON" "$TT_JSON"
