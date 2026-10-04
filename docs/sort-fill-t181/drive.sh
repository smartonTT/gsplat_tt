#!/bin/bash
# t181: sync + build, Tracy captures (bulk fill on = default, GSPLAT_TT_OL_FILL_BULK=0),
# then untraced md5/timing A/B rounds (order swapped).
#   drive.sh [rev] [steps=sync,tracy,time] [rounds="1 2 3"]   (Mac; one ttp lock p100 per device step)
# Stops after the first timing round if the default arm is not md5-identical to md5-r82new.txt.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t181
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/sort-fill-t181/out; mkdir -p $O
STEPS=${2:-sync,tracy,time}
lk() {  # one device step under the p100 lock; a busy lock ends the chain (rc 75)
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo "LOCK_BUSY"; echo CHAIN_DONE; exit 75; fi
  return $rc
}
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t181/run-r*.log" "$H:$T/tmp/t181/md5-r*.txt" $O/ 2>/dev/null; }
if [[ $STEPS == *sync* ]]; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *tracy* ]]; then
  for arm in on off; do
    envs=""; [ $arm = off ] && envs="GSPLAT_TT_OL_FILL_BULK=0"
    lk $DEVRUN --host $H --no-verify --timeout 540 --tag t181-tracy-$arm -- \
      "bash $T/docs/sort-fill-t181/remote_tracy.sh t181-$arm $envs"
    echo "TRACY_${arm}_RC=$?"
    for f in zones.txt gaps.txt fill.txt; do
      scp -q -o BatchMode=yes $H:$T/opt/profiler/t181-$arm/$f $O/tracy-$arm-$f
    done
  done
fi
if [[ $STEPS == *time* ]]; then
  for r in ${3:-1 2 3}; do
    if [ $((r % 2)) -eq 1 ]; then arms="base off:GSPLAT_TT_OL_FILL_BULK=0"; else arms="off:GSPLAT_TT_OL_FILL_BULK=0 base"; fi
    lk $DEVRUN --host $H --no-verify --timeout 560 --tag t181-time$r -- \
      "bash $T/docs/sort-fill-t181/remote_time.sh $r $arms"
    echo "TIME${r}_RC=$?"; fetch
    if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t181/md5-r$r-base.txt" >/dev/null; then
      echo "MD5_GATE_FAIL (r$r base)"; echo CHAIN_DONE; exit 4
    fi
  done
fi
echo CHAIN_DONE
