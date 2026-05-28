#!/usr/bin/env bash
# Infinite metal-port supervisor loop. Emits AGENT_LOOP_WAKE on each tick so
# Cursor agent continues optimization without user input.
#
# Usage:
#   bash scripts/metal_supervisor_loop.sh          # default 10m interval
#   METAL_LOOP_INTERVAL=5m bash scripts/metal_supervisor_loop.sh
#
# Stop: kill the background shell (or pkill -f metal_supervisor_loop.sh)
set -euo pipefail
source "$(dirname "$0")/_env.sh"

INTERVAL="${METAL_LOOP_INTERVAL:-10m}"
case "$INTERVAL" in
  *s) SEC="${INTERVAL%s}" ;;
  *m) SEC=$(( ${INTERVAL%m} * 60 )) ;;
  *h) SEC=$(( ${INTERVAL%h} * 3600 )) ;;
  *d) SEC=$(( ${INTERVAL%d} * 86400 )) ;;
  *)  SEC=600 ;;
esac

PROMPT='You are the metal port supervisor. Run bash scripts/metal_supervisor_tick.sh, read NEXT_ACTION, execute it fully (code+sync+bh-30 verify), update opt/metal-iters.jsonl on milestones, rebuild REPORT.html. Do not wait for user input. Continue until halt conditions in prompts/metal-supervisor.md.'

echo "[metal_supervisor_loop] interval=${INTERVAL} (${SEC}s) repo=$REPO_ROOT" >&2
echo "[metal_supervisor_loop] running initial tick now" >&2
bash "$REPO_ROOT/scripts/metal_supervisor_tick.sh" || true

while true; do
  sleep "$SEC"
  echo "AGENT_LOOP_WAKE_metal_supervisor {\"prompt\":\"$PROMPT\"}"
done
