#!/bin/bash
# t514 (remote, bh-33 p100a): GSPLAT_TT_GAP_CUT b2b A/B, one build, arms by env only.
#   bench514.sh <out dir> <tree> <rounds>
# Arms: off (best-iter-223 default) | gc (GSPLAT_TT_GAP_CUT=1). One discarded warm-up, then <rounds>
# rounds (odd: off gc; even: gc off), then one --dump-views run per arm (md5 list + device hero).
set -u
O=$1; T=$2; ROUNDS=${3:-5}; rc=99
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_GAP_CUT
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME=/localdev/smarton/tt-metal; export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
CACHE=/localdev/smarton/t514-cache
{ echo "=== host $(hostname) $(date -u +%FT%TZ) nproc=$(nproc) sha=$(cat $T/SHA)"
  echo "card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk') load=$(cat /proc/loadavg)"; } | tee $O/host.txt
arm_env() { case $1 in
  off) echo "" ;;
  gc) echo "GSPLAT_TT_GAP_CUT=1" ;;
  *) return 1 ;; esac; }
run() {  # tag arm mode
  local tag=$1 arm=$2 mode=$3 e args r
  e=$(arm_env $arm) || { echo "bad arm $arm"; rc=2; exit 2; }
  args=(--no-ref --iter-dir t514-$tag)
  [ $mode = b2b ] && args+=(--back-to-back)
  [ $mode = dump ] && args+=(--dump-views t514-dump-$tag)
  echo "=== $tag $arm $mode [$e] $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/t514-$tag tmp/t514-dump-$tag
    env $e TT_METAL_CACHE_RENDER=$CACHE/render timeout ${RUN_TO:-600} python3 render/run.py "${args[@]}" > $O/run-$tag.log 2>&1 ); r=$?
  echo "$tag rc=$r"
  grep -E "^B2B scene|^\[DEV\]|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-260
  [ $r = 124 ] && { echo HANG; tt-smi -r 0 > $O/reset-$tag.log 2>&1; rc=124; exit 124; }
  [ $r = 0 ] || { rc=20; return; }
  if [ $mode = dump ]; then
    (cd $T/tmp/t514-dump-$tag && md5sum * | sort -k2) > $O/md5-$tag.txt
    (cd $T && source .venv/bin/activate && python3 opt/md5_golden.py $O/md5-$tag.txt $O/run-$tag.log) | tee -a $O/md5.txt
    cp $T/tmp/t514-$tag/hero_clean.png $O/hero-$arm.png 2>/dev/null || ls $T/tmp/t514-$tag > $O/nohero-$arm.txt
    rm -rf $T/tmp/t514-dump-$tag
  fi
  rm -rf $T/tmp/t514-$tag
}
rc=0
run warm off b2b
for r in $(seq $ROUNDS); do
  arms="off gc"; [ $((r % 2)) = 0 ] && arms="gc off"
  for a in $arms; do run r$r-$a $a b2b; done
done
for a in off gc; do run dump-$a $a dump; done
echo "=== summary"; grep -h "^B2B scene" $O/run-r*.log | wc -l
