#!/bin/bash
# t493 (remote, bh-30): chunk cull de-burst b2b A/B, one build (tree493), arms by env only.
#   bench493.sh <out dir> <tree> <rounds>
# Arms: off (iter 223 default) | c48 (CHUNK_CULL=1 OL_RING=4 OL_RING_DEPTH=8) | c416 (ring depth 16)
#   | c416s (c416 + GSPLAT_TT_CHUNK_SCATTER=1, bit-reversed gid order inside each Morton chunk).
# One discarded warm-up, then <rounds> rounds (odd: off c48 c416 c416s; even: reversed), then one
# --dump-views run per arm (md5 list + device hero). run.py defaults: 1 warm-up + 20 timed passes.
# With VSTART set it runs that viewer start script at exit and touches $O/vstarted.
set -u
O=$1; T=$2; ROUNDS=${3:-4}; rc=99
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_CHUNK_CULL GSPLAT_TT_OL_RING \
  GSPLAT_TT_OL_RING_DEPTH GSPLAT_TT_CHUNK_SCATTER GSPLAT_TT_PFWC_ACQ_FUSE GSPLAT_TT_PFWC_SKIP_RGB GSPLAT_TT_PFWC_VIS_FAST
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME=/localdev/smarton/viewer/tt-metal; export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
CACHE=/localdev/smarton/p150bench/cache493
{ echo "=== host $(hostname) $(date -u +%FT%TZ) nproc=$(nproc) sha=$(cat $T/SHA)"
  echo "aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk') load=$(cat /proc/loadavg)"; } | tee $O/host.txt
arm_env() { case $1 in
  off) echo "" ;;
  c48) echo "GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=8" ;;
  c416) echo "GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=16" ;;
  c416s) echo "GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=16 GSPLAT_TT_CHUNK_SCATTER=1" ;;
  *) return 1 ;; esac; }
run() {  # tag arm mode
  local tag=$1 arm=$2 mode=$3 e args r
  e=$(arm_env $arm) || { echo "bad arm $arm"; rc=2; exit 2; }
  args=(--no-ref --iter-dir t493-$tag)
  [ $mode = b2b ] && args+=(--back-to-back)
  [ $mode = dump ] && args+=(--dump-views t493-dump-$tag)
  echo "=== $tag $arm $mode [$e] $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/t493-$tag tmp/t493-dump-$tag
    env $e TT_METAL_CACHE_RENDER=$CACHE/render timeout ${RUN_TO:-300} python3 render/run.py "${args[@]}" > $O/run-$tag.log 2>&1 ); r=$?
  echo "$tag rc=$r"
  grep -E "^B2B scene|^\[DEV\]|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-260
  [ $r = 124 ] && { echo HANG; tt-smi -r > $O/reset-$tag.log 2>&1; rc=124; exit 124; }
  [ $r = 0 ] || { rc=20; return; }
  if [ $mode = dump ]; then
    (cd $T/tmp/t493-dump-$tag && md5sum * | sort -k2) > $O/md5-$tag.txt
    (cd $T && source .venv/bin/activate && python3 opt/md5_golden.py $O/md5-$tag.txt $O/run-$tag.log) | tee -a $O/md5.txt
    cp $T/tmp/t493-$tag/hero_clean.png $O/hero-$arm.png 2>/dev/null || ls $T/tmp/t493-$tag > $O/nohero-$arm.txt
    rm -rf $T/tmp/t493-dump-$tag
  fi
  rm -rf $T/tmp/t493-$tag
}
rc=0
run warm off b2b
for r in $(seq $ROUNDS); do
  arms="off c48 c416 c416s"; [ $((r % 2)) = 0 ] && arms="c416s c416 c48 off"
  for a in $arms; do run r$r-$a $a b2b; done
done
for a in off c416 c416s c48; do run dump-$a $a dump; done
echo "=== summary"; grep -h "^B2B scene" $O/run-r*.log | wc -l
