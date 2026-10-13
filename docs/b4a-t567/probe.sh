#!/bin/bash
# t567 (remote, bh-30, viewer stopped by drive.sh): reproduce the B4a fold hang under ETH dispatch and
# split its cause. Each case: render/run.py --no-ref, views 0:8, GSPLAT_TT_B4A_FOLD=1, GSPLAT_TT_B4A_VERIFY=3.
#   probe.sh <out dir> <tree>
# Cases: E1 eth 12x10 | E4 worker 11x10 | E3 eth 11x10 (GRID_X=11) | E7 eth 12x9 (GRID_Y=9) |
#        E6 eth 12x10 out pinned off | E2 eth 12x10 watcher (p36 overlay, NOC sanitize + waypoints)
# A case that times out is a hang: tt-smi -r, then the next case. VSTART runs at exit (viewer back).
set -u
O=$1; T=$2; rc=0
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
rm -rf $O; mkdir -p $O
trap 'echo "=== probe end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/probe.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_TT_B4A_FOLD \
  GSPLAT_TT_B4A_VERIFY GSPLAT_TT_B4A_SHADOW GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y GSPLAT_TT_OUT_PINNED TT_METAL_WATCHER
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$V/tt-metal
SYS='/sys/class/tenstorrent/tenstorrent!0'
echo "=== host $(hostname) $(date -u +%FT%TZ) sha=$(cat $T/SHA) card=$(cat "$SYS/tt_card_type") load=$(cat /proc/loadavg)" | tee $O/host.txt
run() {  # tag dispatch(eth|worker|ethw) timeout extra-env...
  local tag=$1 d=$2 to=$3 r; shift 3
  echo "=== $tag [$d $*] $(date -u +%T)"
  ( cd $T || exit 9; source .venv/bin/activate
    case $d in
      eth)  source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13; export GSPLAT_TT_DISPATCH=eth ;;
      ethw) source opt/eth/env.sh $P/ttm-eth12p36 $P/cache567-wp36 || exit 13; export GSPLAT_TT_DISPATCH=eth
            export TT_METAL_LOGS_PATH=$O/wlogs-$tag; mkdir -p $TT_METAL_LOGS_PATH ;;
      worker) export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache567-w/render GSPLAT_TT_DISPATCH=worker ;;
    esac
    rm -rf tmp/t567-$tag
    env GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3 "$@" timeout -k 15 $to python3 render/run.py --no-ref \
      --view-range 0:8 --iter-dir t567-$tag > $O/run-$tag.log 2>&1 ); r=$?
  echo "$tag rc=$r $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^\[eth\]|^\[B4A\]|^SUMMARY|TT_FATAL|TT_THROW|Traceback|watcher|Watcher" $O/run-$tag.log | cut -c1-300 | head -30
  if [ $r = 124 ] || [ $r = 137 ]; then
    echo "HANG $tag"; rc=124
    tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$? $(date -u +%T)"
  fi
  cp $T/tmp/t567-$tag/hero_clean.png $O/hero-$tag.png 2>/dev/null; rm -rf $T/tmp/t567-$tag
}
run E1 eth 180
run E4 worker 180
run E3 eth 180 GSPLAT_TT_GRID_X=11
run E7 eth 180 GSPLAT_TT_GRID_Y=9
run E6 eth 180 GSPLAT_TT_OUT_PINNED=0
run E2 ethw 420 TT_METAL_WATCHER=2 TT_METAL_WATCHER_DISABLE_DISPATCH=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_RING_BUFFER=1 \
  TT_METAL_WATCHER_DISABLE_STACK_USAGE=1
find $O/wlogs-E2 -name 'watcher.log' -exec cp {} $O/watcher-E2.log \; 2>/dev/null
[ -f $O/watcher-E2.log ] && { echo "watcher log lines $(wc -l < $O/watcher-E2.log)"; grep -iE "error|sanitiz|stuck|hang|tripped" $O/watcher-E2.log | head -20; }
