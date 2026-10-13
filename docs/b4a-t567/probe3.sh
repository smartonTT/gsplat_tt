#!/bin/bash
# t567 round 3 (remote, bh-30, viewer stopped by drive.sh): split the ETH fold hang (round 1: every ETH
# grid hangs on the first folded view, worker 11x10 without the overlay passes).
#   probe3.sh <out dir> <tree>
# R1 eth, fold off + B4a shadow (round 2 H1: BRISC K2 in the blend hangs under ETH without the fold)
# R2 eth, fold on (round 2 H4, the real flow)
# At 75 s a still-running case gets tt-triage (all checks, RISC callstacks, NoC status, fast-dispatch
# state) from p150bench/triage-venv over the inspector RPC, then a host gdb backtrace; a hang gets tt-smi -r.
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
    for _ in $(seq 75); do kill -0 $tp 2>/dev/null || break; sleep 1; done
    if kill -0 $tp 2>/dev/null; then
      pp=$(ps --ppid $tp -o pid= | head -1 | tr -d ' ')
      ( cd $TT_METAL_HOME && timeout 150 $P/triage-venv/bin/python tools/tt-triage.py -v > $O/triage-$tag.txt 2>&1
        echo "triage rc=$?" >> $O/triage-$tag.txt )
      echo "triage $tag lines $(wc -l < $O/triage-$tag.txt)"
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
run R1 eth 330 GSPLAT_TT_B4A_FOLD=0 GSPLAT_TT_B4A_SHADOW=1
run R2 eth 330 GSPLAT_TT_B4A_FOLD=1
