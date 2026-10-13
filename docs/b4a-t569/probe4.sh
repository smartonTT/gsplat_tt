#!/bin/bash
# t569 probe 4 (remote, bh-30, viewer stopped by drive.sh): read the launch-message ring (kernel_config) of every
# worker with kcfg.py (tt-exalens, read-only) at the ETH fold hang, and after passing runs for comparison.
#   probe4.sh <out dir> <tree>
# K1 eth, fold on (hangs): dump at 60 s while hung, then tt-smi -r.
# K2 worker dispatch, fold on (passes): dump after it exits (L1 keeps the last 8 launch messages).
# K3 eth, fold off (passes): dump after it exits (mat+blend without K2: config size without K2).
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
grep -n "KERNEL_CONFIG\|MEM_MAP_END\|LAUNCH" $P/ttm-eth12-t427/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h \
  > $O/memmap.txt 2>&1 || grep -rn "KERNEL_CONFIG" $P/ttm-eth12-t427/tt_metal/hw/inc --include=dev_mem_map.h > $O/memmap.txt 2>&1
dump() {  # tag cache-dir
  local fw; fw=$(ls -t $2/render/*/firmware/brisc/brisc.elf 2>/dev/null | head -1)
  echo "kcfg $1 fw=$fw"
  ( cd $P/ttm-eth12-t427/tools/triage && timeout 120 $P/triage-venv/bin/python $P/kcfg.py "$fw" > $O/kcfg-$1.txt 2>&1
    echo "kcfg rc=$?" >> $O/kcfg-$1.txt )
  echo "kcfg $1 lines $(wc -l < $O/kcfg-$1.txt) errors $(grep -c ERROR $O/kcfg-$1.txt)"
}
run() {  # tag eth|worker timeout extra-env...
  local tag=$1 d=$2 to=$3 r tp c; shift 3
  echo "=== $tag [$d $*] $(date -u +%T)"
  case $d in eth) c=$P/cache567-eth ;; worker) c=$P/cache567-w ;; esac
  ( cd $T || exit 9; source .venv/bin/activate
    case $d in
      eth)    source opt/eth/env.sh $P/ttm-eth12-t427 $P/cache567-eth || exit 13; export GSPLAT_TT_DISPATCH=eth ;;
      worker) export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache567-w/render GSPLAT_TT_DISPATCH=worker ;;
    esac
    rm -rf tmp/t569-$tag
    env "$@" timeout -k 15 $to python3 render/run.py --no-ref --view-range 0:3 --iter-dir t569-$tag \
      > $O/run-$tag.log 2>&1 &
    tp=$!
    for _ in $(seq 60); do kill -0 $tp 2>/dev/null || break; sleep 1; done
    if kill -0 $tp 2>/dev/null; then echo "still running at 60 s: dump while hung"; dump $tag $c; fi
    wait $tp ); r=$?
  echo "$tag rc=$r $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^\[eth\]|^\[B4A\]|^SUMMARY|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-300 | head -14
  if [ $r = 124 ] || [ $r = 137 ]; then
    echo "HANG $tag"; rc=124
    tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$? $(date -u +%T)"
  elif [ ! -e $O/kcfg-$tag.txt ]; then dump $tag $c; fi
  rm -rf $T/tmp/t569-$tag
}
run K1 eth 200 GSPLAT_TT_B4A_FOLD=1
run K2 worker 200 GSPLAT_TT_B4A_FOLD=1
run K3 eth 200 GSPLAT_TT_B4A_FOLD=0
