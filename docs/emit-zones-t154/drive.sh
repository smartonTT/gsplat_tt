#!/bin/bash
# t154: sync + build, untraced md5/timing (flag off and on), then the emit-part Tracy capture.
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t154
O=docs/emit-zones-t154/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 420 --tag t154-time -- \
    "bash $T/docs/emit-zones-t154/remote_time.sh 1 base ep:GSPLAT_TT_OL_EMIT_PROF=1 b2:GSPLAT_TT_OL_EMIT_PROF=0"
  echo "TIME_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t154/run-r1-*.log" "$H:$T/tmp/t154/md5-r1-*.txt" $O/
fi
if [[ $STEPS == *tracy* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag t154-tracy -- \
    "bash $T/docs/emit-zones-t154/remote_tracy.sh t154-ep"
  echo "TRACY_RC=$?"
  for f in zones.txt gaps.txt emit_parts.txt capture.log; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t154-ep/$f $O/$f
  done
fi
echo CHAIN_DONE
