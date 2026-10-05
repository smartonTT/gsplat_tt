#!/bin/bash
# t150 Mac-side driver: sync+build the tip, 1 untraced timing round (base + hp), 1 traced 30-view capture.
# Each device step is its own `ttp lock p100` + devrun job. Run from the worktree root (detached).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t150
if [ "${SKIP_SYNC:-0}" = 1 ]; then
  rc=0; echo "SYNC skipped (remote SHA $(ssh -o BatchMode=yes $H cat $T/SHA))"
else
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
fi
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 390 --tag t150-time-1 -- "bash $T/docs/reprofile-t150/remote_time.sh 1 base hp"
echo "TIME_1_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 390 --tag t150-tracy -- "bash $T/docs/reprofile-t150/remote_tracy.sh t150-tip"
echo "TRACY_RC=$?"
for f in gaps.txt roofline.txt zones.txt zone_occupancy.txt; do
  scp -q -o BatchMode=yes $H:$T/opt/profiler/t150-tip/$f docs/reprofile-t150/out/$f
done
scp -q -o BatchMode=yes "$H:$T/tmp/t150/run-r1-*.log" "$H:$T/tmp/t150/hp-r1.txt" docs/reprofile-t150/out/
echo CHAIN_DONE
