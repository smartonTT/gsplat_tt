#!/bin/bash
# t155: sync + build the early one-launch tip on yyzo-bh-07, paired 30-view timing
# (base = early on, off = GSPLAT_TT_SORT_OL_EARLY=0) in two alternating rounds, then Tracy.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t155
O=docs/t155/out; mkdir -p $O
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 750 --tag t155-time-1 -- \
  "bash $T/docs/t155/remote_time.sh 1 base off:GSPLAT_TT_SORT_OL_EARLY=0"
echo "TIME_1_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 750 --tag t155-time-2 -- \
  "bash $T/docs/t155/remote_time.sh 2 off:GSPLAT_TT_SORT_OL_EARLY=0 base"
echo "TIME_2_RC=$?"
scp -q -o BatchMode=yes "$H:$T/tmp/t155/run-r*.log" $O/
if [ "${NO_TRACY:-0}" != 1 ]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t155-tracy -- "bash $T/docs/t155/remote_tracy.sh t155-early"
  echo "TRACY_RC=$?"
  for f in gaps.txt roofline.txt zones.txt zone_occupancy.txt capture.log; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t155-early/$f $O/$f
  done
fi
echo CHAIN_DONE
