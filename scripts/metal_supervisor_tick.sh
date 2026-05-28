#!/usr/bin/env bash
# One supervisor cycle: read state, run remote gates, write next_action.
# Invoked by metal_supervisor_loop.sh and by the agent on each wake.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
STATE="$OPT_DIR/metal-supervisor-state.json"
LOG="$OPT_DIR/metal-supervisor.log"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$OPT_DIR"

# Default state if missing or corrupt (e.g. partial write from a crashed tick).
if [[ ! -s "$STATE" ]] || ! "$LOCAL_PY" -c "import json; json.load(open('$STATE'))" 2>/dev/null; then
  "$LOCAL_PY" - <<'PY' >"$STATE"
import json
print(json.dumps({
    "phase": "iter-001-stage2",
    "next_action": "Implement Stage 2 reader DMA for coeff_table/mb_header/mb_stream (legacy compute unchanged).",
    "milestones": {"stage1_host": True, "stage2_reader": False, "stage3_compute": False},
    "last_hero_psnr_dB": None,
    "last_mb_host_psnr_dB": None,
}, indent=2))
PY
fi

PHASE="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['phase'])")"

echo "[$TS] metal_supervisor_tick phase=$PHASE" | tee -a "$LOG"

# --- bh-30 gates (SSH) ---
BLEND_JSON="$(mktemp)"
MB_JSON="$(mktemp)"
set +e
ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$REMOTE" \
  "cd /proj_sw/user_dev/smarton/gsplat_tt_2 && \
   rm -rf /proj_sw/user_dev/smarton/.cache/tt-metal-cache/* 2>/dev/null; \
   bash -lc 'source .venv/bin/activate && \
     python3 scripts/verify_blend_metal.py' " >"$BLEND_JSON" 2>>"$LOG"
BLEND_RC=$?
ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$REMOTE" \
  "cd /proj_sw/user_dev/smarton/gsplat_tt_2 && \
   bash -lc 'source .venv/bin/activate && \
     python3 scripts/verify_microblock_host.py' " >"$MB_JSON" 2>>"$LOG"
MB_RC=$?
set -e

HERO_PSNR="$("$LOCAL_PY" -c "
import json, sys
try:
    d = json.load(open('$BLEND_JSON'))
    print(d.get('psnr_dB', 'null'))
except Exception:
    print('null')
" 2>/dev/null || echo null)"
MB_PSNR="$("$LOCAL_PY" -c "
import json, sys
try:
    d = json.load(open('$MB_JSON'))
    print(d.get('psnr_vs_numpy_ref_dB', 'null'))
except Exception:
    print('null')
" 2>/dev/null || echo null)"

# --- Decide next_action from phase + gates ---
NEXT="$("$LOCAL_PY" - "$PHASE" "$HERO_PSNR" "$MB_PSNR" "$BLEND_RC" "$MB_RC" <<'PY'
import json, sys
phase, hero, mb, blend_rc, mb_rc = sys.argv[1:6]
hero_f = float(hero) if hero not in ("null", "None", "") else None
mb_f = float(mb) if mb not in ("null", "None", "") else None

nxt = None
phase_out = phase

if blend_rc != "0" and hero_f is None:
    nxt = "BLOCKED: verify_blend_metal failed on remote; read opt/metal-supervisor.log, fix daemon/binary/symlink, re-tick."
elif phase == "iter-001-stage1":
    nxt = (
        "Implement Stage 2: extend alpha_blend reader + host DRAM for coeff_table/mb_header/mb_stream; "
        "compute still uses legacy scalars; gate = hero PSNR unchanged (~47.8 dB). "
        "When landed, set milestones.stage2_reader=true in opt/metal-supervisor-state.json."
    )
    phase_out = "iter-001-stage2"
elif phase == "iter-001-stage2":
    nxt = (
        "Implement Stage 2: extend alpha_blend reader + host DRAM for coeff_table/mb_header/mb_stream; "
        "compute still uses legacy scalars; gate = hero PSNR unchanged (~47.8 dB). "
        "When landed, set milestones.stage2_reader=true in opt/metal-supervisor-state.json."
    )
    phase_out = "iter-001-stage2"
elif phase == "iter-001-stage3":
    if hero_f and hero_f >= 80.0:
        nxt = "Run REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 1 microblock-major port; log iter."
        phase_out = "iter-001-done"
    elif hero_f and hero_f >= 65.0:
        nxt = "Push hero PSNR toward 80 dB: tune exp precision / accumulator path."
        phase_out = "iter-001-stage3"
    else:
        nxt = "Stage 3 compute kernel: microblock-major + fp32 Dst state; verify on bh-30."
        phase_out = "iter-001-stage3"
else:
    nxt = "Review opt/metal-iters.jsonl; pick next metal-iter from plan-amendment-001."
    phase_out = phase

print(json.dumps({"phase": phase_out, "next_action": nxt}))
PY
)"

DECIDE_JSON="$(mktemp)"
TICK_OUT="$(mktemp)"
echo "$NEXT" >"$DECIDE_JSON"

"$LOCAL_PY" - "$STATE" "$DECIDE_JSON" "$BLEND_JSON" "$MB_JSON" "$BLEND_RC" "$MB_RC" >"$TICK_OUT" <<'PY'
import json, sys
from datetime import datetime, timezone

state_path, decide_path, blend_path, mb_path, blend_rc, mb_rc = sys.argv[1:7]
decide = json.load(open(decide_path))
try:
    blend = json.load(open(blend_path))
except (json.JSONDecodeError, FileNotFoundError):
    blend = {}
try:
    mb = json.load(open(mb_path))
except (json.JSONDecodeError, FileNotFoundError):
    mb = {}

def fnum(x):
    if x is None or x == "null":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

prev = {}
try:
    prev = json.load(open(state_path))
except (FileNotFoundError, json.JSONDecodeError):
    pass
milestones = prev.get("milestones", {"stage1_host": True, "stage2_reader": False, "stage3_compute": False})

phase = decide["phase"]
next_action = decide["next_action"]
if milestones.get("stage2_reader") and phase == "iter-001-stage2":
    phase = "iter-001-stage3"
    next_action = (
        "Stage 2 reader landed; implement Stage 3 microblock-major compute kernel "
        "(DST-resident fp32 R/G/B/T per microblock). Gate hero PSNR ≥ 65 dB."
    )
if milestones.get("stage3_compute") and phase == "iter-001-stage3":
    hero_f = fnum(blend.get("psnr_dB"))
    if hero_f and hero_f >= 80.0:
        next_action = "Run REMOTE_HOST=bh-30 scripts/run_iter_metal.sh 1 microblock-major port; log iter."
        phase = "iter-001-done"

state = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "phase": phase,
    "next_action": next_action,
    "milestones": milestones,
    "last_gates": {
        "blend_rc": int(blend_rc),
        "mb_host_rc": int(mb_rc),
        "hero_psnr_dB": fnum(blend.get("psnr_dB")),
        "mb_host_psnr_dB": fnum(mb.get("psnr_vs_numpy_ref_dB")),
    },
}
json.dump(state, open(state_path, "w"), indent=2)
print(state["phase"])
print(state["next_action"])
print(state["last_gates"]["hero_psnr_dB"])
print(state["last_gates"]["mb_host_psnr_dB"])
PY

PHASE_NEW="$(sed -n '1p' "$TICK_OUT")"
NEXT_ACTION="$(sed -n '2p' "$TICK_OUT")"
HERO_OUT="$(sed -n '3p' "$TICK_OUT")"
MB_OUT="$(sed -n '4p' "$TICK_OUT")"

echo "[$TS] phase=$PHASE_NEW hero_psnr=$HERO_OUT mb_psnr=$MB_OUT" >>"$LOG"
echo "---"
echo "PHASE=$PHASE_NEW"
echo "HERO_PSNR=$HERO_OUT"
echo "MB_HOST_PSNR=$MB_OUT"
echo "NEXT_ACTION=$NEXT_ACTION"
echo "STATE=$STATE"

rm -f "$DECIDE_JSON" "$TICK_OUT"

rm -f "$BLEND_JSON" "$MB_JSON"
