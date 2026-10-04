#!/bin/bash
# t168: sync + build, 3 interleaved untraced md5/timing rounds (shared big-tile sort on vs
# GSPLAT_TT_OL_MAT_SHARED=0), then Tracy captures of both arms (fine mat zones on).
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t168
O=docs/mat-shared-t168/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
OFF=off:GSPLAT_TT_OL_MAT_SHARED=0
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  r=0
  for arms in "base $OFF" "$OFF base" "base $OFF"; do
    r=$((r + 1))
    ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t168-time$r -- \
      "bash $T/docs/mat-shared-t168/remote_time.sh $r $arms"
    echo "TIME${r}_RC=$?"
  done
  scp -q -o BatchMode=yes "$H:$T/tmp/t168/run-r*.log" "$H:$T/tmp/t168/md5-r*.txt" $O/
fi
if [[ $STEPS == *tracy* ]]; then
  for arm in "t168-on" "t168-off GSPLAT_TT_OL_MAT_SHARED=0" "t168-onp GSPLAT_TT_MATCULL_PROF=1" "t168-offp GSPLAT_TT_OL_MAT_SHARED=0 GSPLAT_TT_MATCULL_PROF=1"; do
    tag=${arm%% *}
    ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag $tag -- \
      "bash $T/docs/mat-shared-t168/remote_tracy.sh $arm"
    echo "TRACY_${tag}_RC=$?"
    mkdir -p $O/$tag
    for f in zones.txt gaps.txt capture.log; do
      scp -q -o BatchMode=yes $H:$T/opt/profiler/$tag/$f $O/$tag/$f
    done
  done
fi
echo CHAIN_DONE
