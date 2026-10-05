#!/bin/bash
# t150 round 2: sync, md5 round for the device-pack arm (PFWC_FUSE=1 PUBOC_PRE=0, review #149),
# then the 30-view Tracy capture again with the full capture log kept (round 1's capture had no device CSV).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t150
O=docs/reprofile-t150/out
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 390 --tag t150-time-2 -- \
  "bash $T/docs/reprofile-t150/remote_time.sh 2 nopre:GSPLAT_TT_PFWC_FUSE=1,GSPLAT_TT_PUBOC_PRE=0"
echo "TIME_2_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t150-tracy2 -- "bash $T/docs/reprofile-t150/remote_tracy.sh t150-tip"
echo "TRACY_RC=$?"
for f in gaps.txt roofline.txt zones.txt zone_occupancy.txt capture.log; do
  scp -q -o BatchMode=yes $H:$T/opt/profiler/t150-tip/$f $O/$f
done
scp -q -o BatchMode=yes "$H:$T/tmp/t150/run-r2-*.log" $O/
echo CHAIN_DONE
