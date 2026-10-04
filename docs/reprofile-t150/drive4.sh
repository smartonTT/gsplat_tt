#!/bin/bash
# t150 round 4: same as round 3 on the iter-180 tip (3372664, #146 blend diet) + the KCFG fix.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t150
O=docs/reprofile-t150/out/iter180; mkdir -p $O
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 390 --tag t150-time-4 -- \
  "bash $T/docs/reprofile-t150/remote_time.sh 4 base"
echo "TIME_4_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t150-tracy4 -- "bash $T/docs/reprofile-t150/remote_tracy.sh t150-i180"
echo "TRACY_RC=$?"
for f in gaps.txt roofline.txt zones.txt zone_occupancy.txt capture.log; do
  scp -q -o BatchMode=yes $H:$T/opt/profiler/t150-i180/$f $O/$f
done
scp -q -o BatchMode=yes "$H:$T/tmp/t150/run-r4-*.log" $O/
echo CHAIN_DONE
