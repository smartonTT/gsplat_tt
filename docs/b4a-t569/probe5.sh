#!/bin/bash
# t570 probe 5 (remote, bh-30 = p150, ETH dispatch, viewer stopped by drive.sh): bisect the ETH fold hang.
# The fold matblend launch entry is identical under ETH (hangs, 120 cores) and worker (passes, 110 cores),
# so try ETH with fewer cores / one command queue. Each arm: fold on + verify, 200 s timeout, reset on hang.
#   A: GRID_X=11 (110 cores, cols 0-10)   B: GRID_Y=9 (108 cores, all 12 cols)   C: MAT_CQ1=0 (full grid, 1 CQ)
#   probe5.sh <out dir> <tree>
set -u
O=$1; T=$2; rc=0
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
rm -rf $O; mkdir -p $O
trap 'echo "=== probe5 end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/probe.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_TT_B4A_FOLD \
  GSPLAT_TT_B4A_VERIFY GSPLAT_TT_B4A_SHADOW GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y GSPLAT_TT_OUT_PINNED TT_METAL_WATCHER \
  GSPLAT_TT_MAT_CQ1
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$V/tt-metal
echo "=== host $(hostname) $(date -u +%FT%TZ) sha=$(cat $T/SHA) load=$(cat /proc/loadavg)" | tee $O/host.txt
cd $T || exit 9; source .venv/bin/activate
source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13
arm() {  # tag env...
  local tag=$1 r; shift
  echo "=== $tag [$*] $(date -u +%T)"
  rm -rf tmp/t570-$tag
  env GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3 "$@" timeout -k 15 200 python3 render/run.py --no-ref --iter-dir t570-$tag \
    > $O/run-$tag.log 2>&1; r=$?
  echo "$tag rc=$r"
  grep -E "^\[DEV\] dispatch|^\[B4A\] (fold view|fold verify|fold:)|TTW_TIMING ms_view=|Traceback|TT_THROW|TT_FATAL" $O/run-$tag.log | cut -c1-200 | head -8
  if [ $r = 124 ] || [ $r = 137 ]; then echo "HANG $tag"; tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$?"
  elif grep -q "^\[B4A\] fold verify view .* OK" $O/run-$tag.log && ! grep -q MISMATCH $O/run-$tag.log; then echo "PASS $tag"
  else echo "FAIL $tag"; fi
  rm -rf tmp/t570-$tag
}
arm A GSPLAT_TT_GRID_X=11
arm B GSPLAT_TT_GRID_Y=9
arm C GSPLAT_TT_MAT_CQ1=0
