#!/usr/bin/env bash
# Minimal heartbeat: mechanical sync/build/diagnose on bh-30; emit sentinel for Opus.
# Workers (code edits) are spawned by Opus via Task tool, not here.
set -uo pipefail
source "$(dirname "$0")/_env.sh"

INTERVAL_SEC="${AMEND002_HB_SEC:-180}"
LOG="$OPT_DIR/amendment002-heartbeat.log"
STATE="$OPT_DIR/amendment002-supervisor-state.json"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

cycle() {
  local t0=$(date +%s)
  log "=== cycle start ==="
  if bash "$REPO_ROOT/scripts/amendment002_supervisor_work.sh" >>"$LOG" 2>&1; then
    log "cycle OK"
  else
    log "cycle FAIL (continuing)"
  fi
  if "$LOCAL_PY" "$REPO_ROOT/opt/build_report.py" >>"$LOG" 2>&1; then
    log "report regenerated"
  else
    log "report regen FAILED (continuing)"
  fi
  local phase ms_per_view tt_vs_cpu
  phase="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['phase'])" 2>/dev/null || echo unknown)"
  tt_vs_cpu="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['last_gates'].get('psnr_tt_vs_cpu_cpp_mb_dB'))" 2>/dev/null || echo None)"
  ms_per_view="$("$LOCAL_PY" -c "import json; print(json.load(open('$STATE'))['last_gates'].get('ms_per_view_estimate'))" 2>/dev/null || echo None)"
  local elapsed=$(( $(date +%s) - t0 ))
  log "phase=$phase tt_vs_cpu=$tt_vs_cpu ms_per_view=$ms_per_view elapsed=${elapsed}s"
  # Sentinel for Opus to react.
  echo "AGENT_LOOP_TICK_amendment002 {\"phase\":\"$phase\",\"psnr_tt_vs_cpu\":$tt_vs_cpu,\"ms_per_view\":$ms_per_view}"
}

log "heartbeat start pid=$$ interval=${INTERVAL_SEC}s"
cycle
while true; do
  sleep "$INTERVAL_SEC"
  cycle
done
