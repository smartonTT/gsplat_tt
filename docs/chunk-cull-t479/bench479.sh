#!/bin/bash
# t479 (remote, measurement box): b2b A/B of the chunk-cull arms with the #479 ring depth, as t470's bench470.sh
# (run.py --back-to-back --b2b-passes 3 --dump-views, md5 list per run, one smoke per arm, ROUNDS rotated rounds).
# Arms: off (defaults), cull (GSPLAT_TT_CHUNK_CULL=1), cull48 / off48 (+ GSPLAT_TT_OL_RING=4
# GSPLAT_TT_OL_RING_DEPTH=8: two 4-record runs per tile in flight, same L1). First the town protocol unit test
# and its too-lax-wait mutant (must fail).   bench479.sh <out dir> [rounds=3] [arms] [phase]
# phase (each fits one 540 s devrun): smoke = unit test + one smoke per arm; <r> = rotated round r only; all = both.
set -u
O=$1; ROUNDS=${2:-3}; ARMS=${3:-off cull cull48 off48}; PHASE=${4:-all}
T=${T:-/localdev/smarton/gstt2-t479}
rc=99; case $PHASE in smoke|all) rm -rf $O ;; esac; mkdir -p $O
trap 'echo "=== bench end phase=$PHASE rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench-$PHASE.rc' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED \
  GSPLAT_TT_CHUNK_CULL GSPLAT_TT_CHUNK_SKIP GSPLAT_TT_CHUNK_LOG GSPLAT_TT_OL_RING GSPLAT_TT_OL_RING_DEPTH
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t479-b2b
echo "=== bench host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg) sha=$(cut -c1-8 $T/SHA)"
cd $T || exit 9
D=docs/chunk-cull-t479; mkdir -p tmp
if [ $PHASE = smoke ] || [ $PHASE = all ]; then
c++ -O2 -std=c++17 -Irender/kernels/dataflow tests/unit/test_sort_ol_town.cpp -o tmp/t_town && tmp/t_town | tail -1 | sed 's/^/TOWN_TEST /'
[ "${PIPESTATUS[0]}" = 0 ] || { rc=5; exit; }
sed 's/s.fl\[tt\] + (D - R) < c/s.fl[tt] + D < c/' tests/unit/test_sort_ol_town.cpp > tmp/t_mut.cpp
c++ -O2 -std=c++17 -Irender/kernels/dataflow tmp/t_mut.cpp -o tmp/t_mut && { tmp/t_mut > tmp/t_mut.log 2>&1 && { echo "TOWN_MUTANT passed (bad)"; rc=6; exit; } || echo "TOWN_MUTANT fails as expected: $(grep -c FAIL tmp/t_mut.log) FAIL lines"; }
fi
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW|Traceback'
envof() { case $1 in off) ;; cull) echo GSPLAT_TT_CHUNK_CULL=1 ;; ro) echo GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_SKIP=0 ;;
  cull48) echo GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=8 ;;
  off48) echo GSPLAT_TT_OL_RING=4 GSPLAT_TT_OL_RING_DEPTH=8 ;; esac; }
run() {  # run <tag> <arm> <smoke|b2b>
  local tag=$1 arm=$2 mode=$3 rr args it=t479-$1
  case $mode in smoke) args="--view-range 0:1" ;; b2b) args="--back-to-back --b2b-passes 3 --dump-views $it-dump" ;; esac
  echo "=== $tag start $(date -u +%T) arm=$arm env=[$(envof $arm)]"
  ( source .venv/bin/activate; rm -rf tmp/$it tmp/$it-dump
    for kv in $(envof $arm); do export "$kv"; done
    timeout $([ $mode = smoke ] && echo 900 || echo 300) python3 render/run.py --no-ref --iter-dir $it $args > $O/$tag.log 2>&1 ); rr=$?
  echo "$tag rc=$rr $(date -u +%T)"
  grep -E "^B2B |TTW_TIMING b2b_ms_frame(_median)?=|ONELAUNCH v2|$BAD" $O/$tag.log | cut -c1-300 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$tag.log 2>&1; fi
  [ $rr = 0 ] && ! grep -qE "$BAD" $O/$tag.log || return 1
  if [ $mode = b2b ]; then
    (cd tmp/$it-dump && md5sum * | sort -k2) > $O/md5-$tag.txt
    echo "LIST_MD5 $tag $(md5sum < $O/md5-$tag.txt | cut -c1-8) views=$(wc -l < $O/md5-$tag.txt) hero=$(md5sum < tmp/$it/hero_clean.png | cut -c1-8)"
    [ -f $O/hero-$arm.png ] || cp tmp/$it/hero_clean.png $O/hero-$arm.png
  fi
  rm -rf tmp/$it tmp/$it-dump
}
rc=10
if [ $PHASE = smoke ] || [ $PHASE = all ]; then
  for a in $ARMS; do run s0-$a $a smoke || exit; done
fi
rc=0; [ $PHASE = smoke ] && exit; set -- $ARMS; n=$#
RS=$(seq 1 $ROUNDS); [ $PHASE = all ] || RS=$PHASE
for r in $RS; do
  o=""; for k in $(seq 0 $((n - 1))); do i=$(( (k + r - 1) % n + 1 )); o="$o ${!i}"; done
  for a in $o; do run r$r-$a $a b2b || { rc=20; exit; }; done
done
