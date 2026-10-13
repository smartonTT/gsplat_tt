#!/bin/bash
# t615: page FPS on bh-50 for 1 chip and AFR on 2 chips through a temporary tunnel (local 8095).
# Exit trap: close tunnel, stop viewer, release IRD job 133847, confirm it is gone.
set -eo pipefail
cd "$(dirname "$0")/../.."
. docs/viewer-t615/env.sh
O=tmp/t615/out; mkdir -p "$O"; JOB=133847; LP=8095
S=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
PY=/Users/smarton/dev/gsplat_tt/venv-macos/bin/python
ts() { TZ=America/Los_Angeles date '+%F %T %Z'; }
TUN=
cleanup() {
  rc=$?
  [ -n "$TUN" ] && kill "$TUN" 2>/dev/null && echo "=== $(ts) tunnel pid $TUN closed"
  timeout 120 opt/viewer/viewer.sh stop || echo "=== viewer stop rc=$?"
  if [ "${KEEP_RES:-0}" != 1 ]; then
    sel=$(timeout 120 ssh "${S[@]}" yyz-ird "ird list" 2>/dev/null | awk -v j=$JOB '$2=="smarton" && $10==j {print $1}')
    [ -n "$sel" ] && timeout 300 ssh "${S[@]}" yyz-ird "echo y | ird release $sel" 2>&1 | grep -vi password | tail -3
    if timeout 120 ssh "${S[@]}" yyz-ird "ird list" 2>/dev/null | awk -v j=$JOB '$10==j {f=1} END{exit !f}'; then
      echo "=== RELEASE_FAIL $JOB still listed"; [ $rc = 0 ] && rc=30
    else echo "=== $(ts) RELEASED job $JOB gone from ird list"; fi
  fi
  echo "=== $(ts) end rc=$rc"; echo $rc > "$O/measure.rc"
}
trap cleanup EXIT
# wait for the deploy (detached separately)
DRC=$TTP_RUN_DIR_DEPLOY/t615deploy.rc
for _ in $(seq 180); do [ -f "$DRC" ] && break; sleep 10; done
[ -f "$DRC" ] && [ "$(cat "$DRC")" = 0 ] || { echo "=== deploy failed or not done: $(cat "$DRC" 2>/dev/null)"; exit 3; }
wait_log() {  # wait for READY in the viewer log (selftest runs first)
  for _ in $(seq 120); do
    ssh "${S[@]}" bh-50 "grep -q READY $VIEWER_DIR/viewer.log 2>/dev/null" && return 0
    ssh "${S[@]}" bh-50 "grep -q Traceback $VIEWER_DIR/viewer.log 2>/dev/null" && break
    sleep 10; done
  ssh "${S[@]}" bh-50 "tail -30 $VIEWER_DIR/viewer.log"; return 4
}
arm() {  # arm <label>
  local L=$1
  wait_log
  ssh "${S[@]}" bh-50 "cat $VIEWER_DIR/viewer.log" > "$O/viewer-$L.log"
  grep -E "SELFTEST|HERO|ready|chips" "$O/viewer-$L.log" | tail -8
  for f in hero_viewer.png hero_viewer_chip0.png hero_viewer_chip1.png; do
    ssh "${S[@]}" bh-50 "test -f $VIEWER_DIR/$f && cat $VIEWER_DIR/$f && rm -f $VIEWER_DIR/$f" > "$O/$L-$f" || rm -f "$O/$L-$f"
  done
  [ -n "$TUN" ] || { ssh "${S[@]}" -N -o ExitOnForwardFailure=yes -L $LP:localhost:8080 bh-50 & TUN=$!; sleep 5; echo "=== $(ts) tunnel pid $TUN"; }
  python3 -c "import sys; sys.path.insert(0,'opt/viewer'); import deploy_and_measure as d; ok,m=d.wait_ready($LP,300); print('ready',ok,m); sys.exit(0 if ok else 5)"
  for i in 1 2 3; do
    echo "=== $(ts) sweep $L r$i"
    "$PY" opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:$LP --no-decode --secs 23 --label "$L-r$i" > "$O/sweep-$L-r$i.txt" 2>&1
    grep '^SWEEP ' "$O/sweep-$L-r$i.txt" | tail -1
  done
  ssh "${S[@]}" bh-50 "cat $VIEWER_DIR/viewer.log" > "$O/viewer-$L.log"
}
echo "=== $(ts) arm 1chip (deployed single-chip viewer)"
arm 1chip
echo "=== $(ts) restart AFR 0,1"
VIEWER_AFR_DEVICES=0,1 timeout 300 opt/viewer/viewer.sh restart
arm afr2
echo "=== $(ts) done"
