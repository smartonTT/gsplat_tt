#!/usr/bin/env bash
# Viewer supervisor (task #357), started by `opt/viewer/viewer.sh start` on the box from
# the deployed tree:  supervise.sh VDIR PORT
# Runs opt/viewer/viewer_clean.py and restarts it when it exits non-zero (the render
# watchdog exits 86 on a hung render; a crash or a failed device open also counts).
# Exit 0 (a clean stop by SIGINT), $VDIR/viewer.stop, exit 75 (port taken by another
# viewer) or death by SIGINT/SIGTERM/SIGKILL (rc 130/143/137: an outside stop, e.g. a bench
# using an older viewer.sh without the stop file) ends the loop. Backoff doubles
# from VIEWER_BACKOFF_S (5) to VIEWER_BACKOFF_MAX_S (300) and resets after a run of
# VIEWER_HEALTHY_S (600); more than VIEWER_MAX_RESTARTS_PER_HOUR (6) restarts in an
# hour gives up and leaves the viewer down. Log: $VDIR/viewer_supervisor.log.
set -u
VDIR=${1:?VDIR}
PORT=${2:?PORT}
BACKOFF0=${VIEWER_BACKOFF_S:-5}
BACKOFF_MAX=${VIEWER_BACKOFF_MAX_S:-300}
HEALTHY=${VIEWER_HEALTHY_S:-600}
MAX_PER_HOUR=${VIEWER_MAX_RESTARTS_PER_HOUR:-6}
PY=${VIEWER_PYTHON:-.venv/bin/python}
SLOG=$VDIR/viewer_supervisor.log
echo $$ > "$VDIR/viewer.sup.pid"
log() { echo "[supervisor $(date -u +%Y-%m-%dT%H:%M:%SZ) pid $$] $*" | tee -a "$SLOG"; }

restarts=()
backoff=$BACKOFF0
while :; do
  if [ -e "$VDIR/viewer.stop" ]; then log "stop file present: not starting"; break; fi
  t0=$(date +%s)
  log "starting viewer_clean.py port $PORT"
  # Foreground child: a background job of a non-interactive shell ignores SIGINT,
  # and viewer.sh stop needs SIGINT for a clean device shutdown.
  bash -c 'echo $$ > "$0/viewer.pid"; exec "$1" opt/viewer/viewer_clean.py scenes/bicycle.ply --port "$2"' \
    "$VDIR" "$PY" "$PORT"
  rc=$?
  ran=$(( $(date +%s) - t0 ))
  log "viewer exited rc=$rc after ${ran}s"
  if [ "$rc" = 0 ]; then log "clean exit: supervisor done"; break; fi
  if [ -e "$VDIR/viewer.stop" ]; then log "stop file present: not restarting"; break; fi
  # Without these two, an outside stop+start made a second viewer that viser moved to the next port.
  if [ "$rc" = 75 ]; then log "port $PORT is taken: another viewer runs; supervisor done"; break; fi
  case $rc in 130|137|143) log "killed by a signal: outside stop; supervisor done"; break ;; esac
  now=$(date +%s)
  kept=()
  for t in "${restarts[@]+"${restarts[@]}"}"; do [ $((now - t)) -lt 3600 ] && kept+=("$t"); done
  restarts=("${kept[@]+"${kept[@]}"}")
  if [ "${#restarts[@]}" -ge "$MAX_PER_HOUR" ]; then
    log "giving up: ${#restarts[@]} restarts in the last hour (cap $MAX_PER_HOUR)"
    rm -f "$VDIR/viewer.sup.pid"
    exit 1
  fi
  [ "$ran" -ge "$HEALTHY" ] && backoff=$BACKOFF0
  log "restart $(( ${#restarts[@]} + 1 ))/$MAX_PER_HOUR this hour in ${backoff}s"
  sleep "$backoff"
  restarts+=("$(date +%s)")
  backoff=$(( backoff * 2 > BACKOFF_MAX ? BACKOFF_MAX : backoff * 2 ))
done
rm -f "$VDIR/viewer.sup.pid"
