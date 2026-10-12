#!/bin/bash
# t523 (remote, bh-33 p100a): combined b2b A/B of the stacked small levers at best-iter-223.
#   bench523.sh <out dir> <tree> <rounds>
# One build, arms by env only:
#   off: best-iter-223 defaults (every gate unset)
#   ec:  GSPLAT_TT_MAT_EARLY_CULL=1 alone (#519; the iter-224 candidate landing in #525)
#   st:  full stack = ec + vf48 bundle (PFWC_VIS_FAST=1 OL_RING=4 OL_RING_DEPTH=8, #494)
#        + GAP_CUT=1 (#514) + PFWC_DEAL=range (#517). #511 QF probes are slower: left out.
# One discarded warm-up, then <rounds> rounds rotating the arm order (off ec st / ec st off / st off ec),
# then one --dump-views run per arm (md5 list + device hero). run.py: 1 warm-up + 20 timed passes.
set -u
O=$1; T=$2; ROUNDS=${3:-6}; rc=99
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PFWC_DEAL_W GSPLAT_TT_CHUNK_CULL \
  GSPLAT_TT_OL_RING GSPLAT_TT_OL_RING_DEPTH GSPLAT_TT_CHUNK_SCATTER GSPLAT_TT_PFWC_ACQ_FUSE GSPLAT_TT_PFWC_SKIP_RGB \
  GSPLAT_TT_PFWC_VIS_FAST GSPLAT_TT_GAP_CUT GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_BLEND_QF_PROBE GSPLAT_TT_BLEND_QF_PAD
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME=/localdev/smarton/tt-metal; export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
CACHE=/localdev/smarton/t523-cache
{ echo "=== host $(hostname) $(date -u +%FT%TZ) nproc=$(nproc) sha=$(cat $T/SHA)"
  echo "card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk') load=$(cat /proc/loadavg)"; } | tee $O/host.txt
EC="GSPLAT_TT_MAT_EARLY_CULL=1"
arm_env() { case $1 in
  off) echo "" ;;
  ec) echo "$EC" ;;
  st) echo "$EC GSPLAT_TT_PFWC_VIS_FAST=1 GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=8 GSPLAT_TT_GAP_CUT=1 GSPLAT_TT_PFWC_DEAL=range GSPLAT_TT_PFWC_DEAL_W=$T/docs/pfwc-speed-deal-t517/bicycle-w517.txt" ;;
  *) return 1 ;; esac; }
run() {  # tag arm mode
  local tag=$1 arm=$2 mode=$3 e args r
  e=$(arm_env $arm) || { echo "bad arm $arm"; rc=2; exit 2; }
  args=(--no-ref --iter-dir t523-$tag)
  [ $mode = b2b ] && args+=(--back-to-back)
  [ $mode = dump ] && args+=(--dump-views t523-dump-$tag)
  echo "=== $tag $arm $mode [$e] $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/t523-$tag tmp/t523-dump-$tag
    env $e TT_METAL_CACHE_RENDER=$CACHE/render timeout ${RUN_TO:-600} python3 render/run.py "${args[@]}" > $O/run-$tag.log 2>&1 ); r=$?
  echo "$tag rc=$r"
  grep -E "^B2B scene|^\[DEV\]|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-260
  [ $r = 124 ] && { echo HANG; tt-smi -r 0 > $O/reset-$tag.log 2>&1; rc=124; exit 124; }
  [ $r = 0 ] || { rc=20; return; }
  if [ $mode = dump ]; then
    (cd $T/tmp/t523-dump-$tag && md5sum * | sort -k2) > $O/md5-$tag.txt
    (cd $T && source .venv/bin/activate && python3 opt/md5_golden.py $O/md5-$tag.txt $O/run-$tag.log) | tee -a $O/md5.txt
    cp $T/tmp/t523-$tag/hero_clean.png $O/hero-$arm.png 2>/dev/null || ls $T/tmp/t523-$tag > $O/nohero-$arm.txt
    rm -rf $T/tmp/t523-dump-$tag
  fi
  rm -rf $T/tmp/t523-$tag
}
rc=0
run warm off b2b
for r in $(seq $ROUNDS); do
  case $((r % 3)) in 1) arms="off ec st" ;; 2) arms="ec st off" ;; 0) arms="st off ec" ;; esac
  for a in $arms; do run r$r-$a $a b2b; done
done
for a in off ec st; do run dump-$a $a dump; done
echo "=== summary"; grep -h "^B2B scene" $O/run-r*.log | wc -l
