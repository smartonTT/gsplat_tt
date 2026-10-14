#!/bin/bash
# t661: AFR bit-match check on bh-50 (2-board p150) at the head deployed by deploy.sh.
# Needs JOB (this task's own IRD job on bh-50, resource p150-2board). Exit trap: stop viewer,
# release $JOB, confirm it is gone. Arms (worker dispatch, grid cap 12 as in t615):
#   b0      1 chip, board 0 (13x10 -> 12x10)
#   b1      1 chip, board 1 (harvested, 11x10); also checks viewer stop ends the AFR workers
#   afr     AFR 0,1, default same-grid cap -> both chips 11x10, chip0 == chip1 expected
#   afrdiff AFR 0,1, GSPLAT_VIEWER_AFR_SAME_GRID=0 -> the t615 mismatch (257 px) expected
#   b0g11   1 chip, board 0 capped to 11x10 -> must equal b1 (root cause: core count)
export VIEWER_DISPATCH=worker
set -eo pipefail
cd "$(dirname "$0")/../.."
. docs/viewer-t615/env.sh
O=tmp/t661/out; mkdir -p "$O"; JOB=${JOB:?IRD job id}; LP=8095
S=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
PY=/Users/smarton/dev/gsplat_tt/venv-macos/bin/python
ts() { TZ=America/Los_Angeles date '+%F %T %Z'; }
TUN=
cleanup() {
  rc=$?
  [ -n "$TUN" ] && kill "$TUN" 2>/dev/null
  timeout 180 opt/viewer/viewer.sh stop || echo "=== viewer stop rc=$?"
  ssh "${S[@]}" bh-50 "ps -eo pid,args | grep -c '[m]ultiprocessing' || true" | sed 's/^/=== leftover mp procs: /'
  if [ "${KEEP_RES:-0}" != 1 ]; then
    sel=$(timeout 120 ssh "${S[@]}" yyz-ird "ird list" 2>/dev/null | awk -v j=$JOB '$2=="smarton" && $10==j {print $1}')
    [ -n "$sel" ] && timeout 300 ssh "${S[@]}" yyz-ird "echo y | ird release $sel" 2>&1 | grep -vi password | tail -3
    if timeout 120 ssh "${S[@]}" yyz-ird "ird list" 2>/dev/null | awk -v j=$JOB '$10==j {f=1} END{exit !f}'; then
      echo "=== RELEASE_FAIL $JOB still listed"; [ $rc = 0 ] && rc=30
    else echo "=== $(ts) RELEASED job $JOB"; fi
  fi
  echo "=== $(ts) end rc=$rc"; echo $rc > "$O/verify.rc"
}
trap cleanup EXIT
wait_log() {
  for _ in $(seq 120); do
    ssh "${S[@]}" bh-50 "grep -q READY $VIEWER_DIR/viewer.log 2>/dev/null" && return 0
    ssh "${S[@]}" bh-50 "grep -q Traceback $VIEWER_DIR/viewer.log 2>/dev/null" && break
    sleep 10; done
  ssh "${S[@]}" bh-50 "cat $VIEWER_DIR/viewer.log" > "$O/viewer-fail.log" || true
  tail -20 "$O/viewer-fail.log"; return 4
}
arm() {  # arm <label> [sweep]
  local L=$1
  wait_log
  ssh "${S[@]}" bh-50 "cat $VIEWER_DIR/viewer.log" > "$O/viewer-$L.log"
  grep -E "SELFTEST|HERO|\[afr\]|grid" "$O/viewer-$L.log" | tail -10
  for f in hero_viewer.png hero_viewer_chip0.png hero_viewer_chip1.png; do
    ssh "${S[@]}" bh-50 "test -f $VIEWER_DIR/$f && cat $VIEWER_DIR/$f && rm -f $VIEWER_DIR/$f" > "$O/$L-$f" || rm -f "$O/$L-$f"
  done
  [ "${2:-}" = sweep ] || return 0
  [ -n "$TUN" ] || { ssh "${S[@]}" -N -o ExitOnForwardFailure=yes -L $LP:localhost:8080 bh-50 & TUN=$!; sleep 5; }
  python3 -c "import sys; sys.path.insert(0,'opt/viewer'); import deploy_and_measure as d; ok,m=d.wait_ready($LP,300); print('ready',ok,m); sys.exit(0 if ok else 5)"
  for i in 1 2; do
    "$PY" opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:$LP --subprotocol viser-v1.0.27 --no-decode --secs 23 --label "$L-r$i" > "$O/sweep-$L-r$i.txt" 2>&1
    grep '^SWEEP ' "$O/sweep-$L-r$i.txt" | tail -1
  done
}
run() {  # run <label> <env...> [-- sweep]
  local L=$1; shift; echo "=== $(ts) arm $L: $*"
  env "$@" VIEWER_GRID_X=${GX:-12} timeout 300 opt/viewer/viewer.sh restart
  arm "$L" "${SW:-}"
}
n=$(ssh "${S[@]}" bh-50 "ls /dev/tenstorrent/ | wc -l")
echo "=== cards visible: $n"
[ "$n" -ge 2 ] || { echo "=== need 2 cards (reserve with --num-pcie-chips 2 --model p150)"; exit 6; }
run b0 VIEWER_VISIBLE_DEVICES=0
run b1 VIEWER_VISIBLE_DEVICES=1
SW=sweep run afr VIEWER_AFR_DEVICES=0,1
[ -n "$TUN" ] && kill "$TUN" && TUN=
run afrdiff VIEWER_AFR_DEVICES=0,1 VIEWER_AFR_SAME_GRID=0
GX=11 run b0g11 VIEWER_VISIBLE_DEVICES=0
"$PY" docs/afr-t661/compare.py "$O" | tee "$O/compare.txt"
echo "=== $(ts) done"
