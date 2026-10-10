#!/bin/bash
# t470 (remote, bh-30): b2b A/B of the #433 chunk-cull arms on tree470 (device build = opt tip, copied from
# tree465b, 13ce253e). Arms: off (default), ro (GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0: Morton reorder,
# every tile kept), cull (GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_LOG=1). One smoke per arm (kernel compile),
# then ROUNDS rounds with the arm order rotated. Each run: run.py --back-to-back (1 check pass + 3 measured)
# with --dump-views (pass 0 only, outside the measured passes) for the per-run md5 list.
# Env as docs/b2b-gap/probe464.sh run(). Writes $O/*.log, md5-*.txt, hero-<arm>.png, bench.rc; runs
# $VSTART (viewer restart) at exit and touches $O/vstarted.
#   bench470.sh <out dir> <tree> [rounds=3]
set -u
O=$1; T=$2; ROUNDS=${3:-3}; P=/localdev/smarton/p150bench
rc=99
rm -rf $O; mkdir -p $O
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc
  if [ -n "${VSTART:-}" ]; then echo "=== viewer start $(date -u +%FT%TZ)"; bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED \
  GSPLAT_TT_CHUNK_CULL GSPLAT_TT_CHUNK_SKIP GSPLAT_TT_CHUNK_LOG
export TT_METAL_HOME=/localdev/smarton/viewer/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache470/render
echo "=== bench host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type' 2>/dev/null) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "tree=$T sha=$(cut -c1-8 $T/SHA)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW|Traceback'
envof() { case $1 in off) ;; ro) echo GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0 ;; cull) echo GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_LOG=1 ;; esac; }
run() {  # run <tag> <arm> <smoke|b2b>
  local tag=$1 arm=$2 mode=$3 rr args it=t470-$1
  case $mode in
    smoke) args="--view-range 0:1" ;;
    b2b) args="--back-to-back --b2b-passes 3 --dump-views $it-dump" ;;
  esac
  echo "=== $tag start $(date -u +%T) arm=$arm env=[$(envof $arm)] load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/$it tmp/$it-dump
    for kv in $(envof $arm); do export "$kv"; done
    timeout $([ $mode = smoke ] && echo 900 || echo 300) \
      python3 render/run_t470.py --no-ref --iter-dir $it $args > $O/$tag.log 2>&1 ); rr=$?
  echo "$tag rc=$rr $(date -u +%T)"
  grep -E "^B2B |^B2B_STAGES|TTW_TIMING (ms_view|b2b_ms_frame(_median)?)=|$BAD" $O/$tag.log | cut -c1-500 | head -8
  grep -m3 "chunk cull:" $O/$tag.log
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$tag.log 2>&1; echo "reset rc=$?"; fi
  [ $rr = 0 ] && ! grep -qE "$BAD" $O/$tag.log || return 1
  if [ $mode = b2b ]; then
    (cd $T/tmp/$it-dump && md5sum * | sort -k2) > $O/md5-$tag.txt
    echo "LIST_MD5 $tag $(md5sum < $O/md5-$tag.txt | cut -c1-8) views=$(wc -l < $O/md5-$tag.txt)"
    (cd $T && python3 opt/md5_golden.py $O/md5-$tag.txt $O/$tag.log) 2>&1 | tail -2
    [ -f $O/hero-$arm.png ] || cp $T/tmp/$it/hero_clean.png $O/hero-$arm.png
  fi
  rm -rf $T/tmp/$it $T/tmp/$it-dump
}
rc=10
run s0-off off smoke && run s0-ro ro smoke && run s0-cull cull smoke || exit
rc=0
for r in $(seq 1 $ROUNDS); do
  case $((r % 3)) in 1) o="off ro cull" ;; 2) o="ro cull off" ;; 0) o="cull off ro" ;; esac
  for a in $o; do run r$r-$a $a b2b || { rc=20; exit; }; done
done
