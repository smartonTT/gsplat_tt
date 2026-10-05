#!/bin/bash
# t150 round 3: sync the GSPLAT_TT_KCFG_EXTRA_KB fix, a default-env md5 round, then the 30-view Tracy capture.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t150
O=docs/reprofile-t150/out
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 390 --tag t150-time-3 -- \
  "bash $T/docs/reprofile-t150/remote_time.sh 3 base"
echo "TIME_3_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t150-tracy3 -- "bash $T/docs/reprofile-t150/remote_tracy.sh t150-tip"
echo "TRACY_RC=$?"
for f in gaps.txt roofline.txt zones.txt zone_occupancy.txt capture.log; do
  scp -q -o BatchMode=yes $H:$T/opt/profiler/t150-tip/$f $O/$f
done
scp -q -o BatchMode=yes "$H:$T/tmp/t150/run-r3-*.log" $O/
echo CHAIN_DONE
