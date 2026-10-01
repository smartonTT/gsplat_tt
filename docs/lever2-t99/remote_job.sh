#!/bin/bash
# t99 lever 2 (GSPLAT_TT_SFPU_VIS) device job, run ON yyzo-bh-07 (p100a). Wrap the
# whole sync + run sequence in `ttp lock p100 -- ...` (see README.md).
#   remote_job.sh <round> [steps]
# steps: base legacy check vis vis_nobal vis_w8 vis_w64 vis_notau
#   base      base tree (08f9200), default env
#   legacy    branch tree, GSPLAT_TT_SFPU_VIS=0 (must equal base: md5 and time)
#   check     branch tree, GSPLAT_TT_SFPU_VIS=2 (per-view [VIS-CHECK] lines, slow)
#   vis       branch tree, GSPLAT_TT_SFPU_VIS=1 (the timed candidate)
#   vis_nobal GSPLAT_TT_SFPU_VIS=1 GSPLAT_TT_VIS_BALANCE=0 (balance attribution)
#   vis_w8 / vis_w64  GSPLAT_TT_VIS_TILE_WEIGHT=8 / 64 (cost-model sweep)
#   vis_notau GSPLAT_TT_VIS_EDGE_TAU=0 (insurance off; md5 says if SFPMAD is RNE)
# Each run: 30 bicycle views untraced, md5 vs md5-r82new.txt, stage lines.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
S=/localdev/smarton/t99_scripts; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
B=${T99_BASE:-/localdev/smarton/gstt2-t99b}; F=${T99_TREE:-/localdev/smarton/gstt2-t99}
r=${1:-1}; shift
steps=${*:-base legacy check vis}
run() {  # tag tree [env...]; non-zero if the run failed
  local tag=$1 T=$2; shift 2
  echo "=== $tag $(cut -c1-7 $T/SHA) $*"
  ( cd $T && source .venv/bin/activate && mkdir -p tmp &&
    env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T) timeout 600 \
      python3 render/run.py --no-ref --iter-dir t99-$tag --dump-views t99-dump-$tag \
      > tmp/t99-run-$tag.log 2>&1
    rc=$?; echo "run rc=$rc"
    grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|error:" tmp/t99-run-$tag.log | head -12
    if grep -q VIS-CHECK tmp/t99-run-$tag.log; then
      echo "VIS-CHECK OK=$(grep -c 'VIS-CHECK.* OK$' tmp/t99-run-$tag.log)" \
           "MISMATCH=$(grep -c 'VIS-CHECK.*MISMATCH' tmp/t99-run-$tag.log)"
      grep 'VIS-CHECK.*MISMATCH' tmp/t99-run-$tag.log | head -4
    fi
    d=$(find . -maxdepth 3 -type d -name t99-dump-$tag | head -1)
    [ -n "$d" ] && (cd $d && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "$tag views: $(wc -l < $S/md5-$tag.txt 2>/dev/null || echo 0)"
    diff $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL" ||
      { echo "VIEWS DIFFER"; diff $REF $S/md5-$tag.txt | head -4; }
    exit $rc )
}
for s in $steps; do
  case $s in
    base)      run t99r$r-base $B ;;
    legacy)    run t99r$r-legacy $F GSPLAT_TT_SFPU_VIS=0 ;;
    check)     run t99r$r-check $F GSPLAT_TT_SFPU_VIS=2 ;;
    vis)       run t99r$r-vis $F GSPLAT_TT_SFPU_VIS=1 ;;
    vis_nobal) run t99r$r-vis-nobal $F GSPLAT_TT_SFPU_VIS=1 GSPLAT_TT_VIS_BALANCE=0 ;;
    vis_w8)    run t99r$r-vis-w8 $F GSPLAT_TT_SFPU_VIS=1 GSPLAT_TT_VIS_TILE_WEIGHT=8 ;;
    vis_w64)   run t99r$r-vis-w64 $F GSPLAT_TT_SFPU_VIS=1 GSPLAT_TT_VIS_TILE_WEIGHT=64 ;;
    vis_notau) run t99r$r-vis-notau $F GSPLAT_TT_SFPU_VIS=1 GSPLAT_TT_VIS_EDGE_TAU=0 ;;
  esac || echo "STEP $s FAILED"
done
