#!/bin/bash
# t164: sync + build, untraced md5/timing (fold on vs GSPLAT_TT_OL_EMIT_FOLD=0, 2 rounds,
# order swapped), then the emit-part Tracy capture with the fold.
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t164
O=docs/emit-fold-t164/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t164-time1 -- \
    "bash $T/docs/emit-fold-t164/remote_time.sh 1 base off:GSPLAT_TT_OL_EMIT_FOLD=0"
  echo "TIME1_RC=$?"
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t164-time2 -- \
    "bash $T/docs/emit-fold-t164/remote_time.sh 2 off:GSPLAT_TT_OL_EMIT_FOLD=0 base"
  echo "TIME2_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t164/run-r*.log" "$H:$T/tmp/t164/md5-r*.txt" $O/
fi
if [[ $STEPS == *tracy* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag t164-tracy -- \
    "bash $T/docs/emit-fold-t164/remote_tracy.sh t164-ep"
  echo "TRACY_RC=$?"
  for f in zones.txt gaps.txt emit_parts.txt capture.log; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t164-ep/$f $O/$f
  done
fi
echo CHAIN_DONE
