#!/bin/bash
# t567 round 2 (remote, bh-30, viewer stopped by drive.sh): split the ETH fold hang (round 1: every ETH
# grid hangs on the first folded view, worker 11x10 without the overlay passes).
#   probe2.sh <out dir> <tree>
# H1 eth, fold off + B4a shadow (BRISC K2 body in the blend, no pfwc reorder)
# H2 ETH overlay but worker dispatch, fold on (overlay vs dispatch mode)
# H3 eth, fold on, GSPLAT_TT_MAT_CQ1=0 (no CQ1 cross-queue wait)
# H4 eth, fold on, no verify (the plain flow)
# A run still alive at 90 s gets a host gdb backtrace (all threads), then the timeout; a hang gets tt-smi -r.
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
run() {  # tag eth|worker|ethworker timeout extra-env...
  local tag=$1 d=$2 to=$3 r tp pp; shift 3
  echo "=== $tag [$d $*] $(date -u +%T)"
  ( cd $T || exit 9; source .venv/bin/activate
    case $d in
      eth)       source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13; export GSPLAT_TT_DISPATCH=eth ;;
      ethworker) source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13; export GSPLAT_TT_DISPATCH=worker ;;
      worker)    export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache567-w/render GSPLAT_TT_DISPATCH=worker ;;
    esac
    rm -rf tmp/t567-$tag
    env "$@" timeout -k 15 $to python3 render/run.py --no-ref --view-range 0:8 --iter-dir t567-$tag \
      > $O/run-$tag.log 2>&1 &
    tp=$!
    for _ in $(seq 90); do kill -0 $tp 2>/dev/null || break; sleep 1; done
    if kill -0 $tp 2>/dev/null; then
      pp=$(ps --ppid $tp -o pid= | head -1 | tr -d ' ')
      [ -n "$pp" ] && timeout 60 gdb -p $pp -batch -ex "thread apply all bt 30" > $O/gdb-$tag.txt 2>&1
      echo "gdb $tag pid $pp lines $(wc -l < $O/gdb-$tag.txt 2>/dev/null)"
    fi
    wait $tp ); r=$?
  echo "$tag rc=$r $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^\[eth\]|^\[B4A\]|^SUMMARY|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-300 | head -14
  if [ $r = 124 ] || [ $r = 137 ]; then
    echo "HANG $tag"; rc=124
    tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$? $(date -u +%T)"
  fi
  cp $T/tmp/t567-$tag/hero_clean.png $O/hero-$tag.png 2>/dev/null; rm -rf $T/tmp/t567-$tag
}
run H1 eth 150 GSPLAT_TT_B4A_FOLD=0 GSPLAT_TT_B4A_SHADOW=1
run H2 ethworker 150 GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3
run H3 eth 150 GSPLAT_TT_B4A_FOLD=1 GSPLAT_TT_B4A_VERIFY=3 GSPLAT_TT_MAT_CQ1=0
run H4 eth 150 GSPLAT_TT_B4A_FOLD=1
