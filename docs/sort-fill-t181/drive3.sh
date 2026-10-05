#!/bin/bash
# t181 after the hang fix (#187's cur_lm -> CB_CUR cherry-picked: the hangs were a
# kernel_main stack overrun, #186): sync + build, 2-view smoke with the bulk fill (stop if it
# hangs), Tracy views 0:10 bulk on / off (fill zones, GB/s), then 3 untraced 30-view md5/timing
# rounds, order swapped (drive.sh time step).
#   drive3.sh   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t181
O=docs/sort-fill-t181/out3; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T HEAD; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
lk $DEVRUN --host $H --no-verify --timeout 300 --tag t181-on3 -- "bash $T/docs/sort-fill-t181/remote_smoke.sh on3"
rc=$?; echo "SMOKE_on3_RC=$rc"
scp -q -o BatchMode=yes "$H:$T/tmp/t181/run-on3.log" $O/ 2>/dev/null
[ $rc -eq 0 ] || { echo "SMOKE_FAIL"; echo CHAIN_DONE; exit 5; }
for arm in on off; do
  envs=""; [ $arm = off ] && envs="GSPLAT_TT_OL_FILL_BULK=0"
  lk $DEVRUN --host $H --no-verify --timeout 540 --tag t181-tracy3-$arm -- \
    "TRACY_VIEWS=0:10 bash $T/docs/sort-fill-t181/remote_tracy.sh t181-$arm-3 $envs"
  echo "TRACY_${arm}_RC=$?"
  for f in zones.txt gaps.txt fill.txt; do scp -q -o BatchMode=yes $H:$T/opt/profiler/t181-$arm-3/$f $O/tracy-$arm-$f; done
done
bash docs/sort-fill-t181/drive.sh HEAD time "1 2 3"; rc=$?
scp -q -o BatchMode=yes "$H:$T/tmp/t181/run-r*.log" "$H:$T/tmp/t181/md5-r*.txt" $O/ 2>/dev/null
echo "DRIVE_TIME_RC=$rc"
echo CHAIN_DONE
