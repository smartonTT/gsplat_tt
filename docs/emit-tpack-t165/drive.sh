#!/bin/bash
# t165: sync + build, smoke (TRISC pack only), untraced md5/timing (default vs
# GSPLAT_TT_OL_EMIT_TPACK=1, 2 rounds, order swapped), then the emit-part Tracy capture with TPACK=1.
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t165
O=docs/emit-tpack-t165/out; mkdir -p $O
STEPS=${2:-sync,smoke,time,tracy}
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *smoke* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 400 --tag t165-smoke -- \
    "bash $T/docs/emit-tpack-t165/remote_time.sh 0 tp:GSPLAT_TT_OL_EMIT_TPACK=1"
  echo "SMOKE_RC=$?"
fi
if [[ $STEPS == *time* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t165-time1 -- \
    "bash $T/docs/emit-tpack-t165/remote_time.sh 1 base tp:GSPLAT_TT_OL_EMIT_TPACK=1"
  echo "TIME1_RC=$?"
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t165-time2 -- \
    "bash $T/docs/emit-tpack-t165/remote_time.sh 2 tp:GSPLAT_TT_OL_EMIT_TPACK=1 base"
  echo "TIME2_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t165/run-r*.log" "$H:$T/tmp/t165/md5-r*.txt" $O/
fi
if [[ $STEPS == *tracy* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag t165-tracy -- \
    "GSPLAT_TT_OL_EMIT_TPACK=${TPACK:-1} bash $T/docs/emit-tpack-t165/remote_tracy.sh t165-ep"
  echo "TRACY_RC=$?"
  for f in zones.txt gaps.txt emit_parts.txt capture.log; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t165-ep/$f $O/$f
  done
fi
echo CHAIN_DONE
