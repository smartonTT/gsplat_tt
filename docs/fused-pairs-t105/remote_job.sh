#!/bin/bash
# t105 lever 3 (GSPLAT_TT_FUSED_PAIRS) device job, run ON yyzo-bh-07 (p100a). Wrap the
# whole sync + run sequence in `ttp lock p100 -- ...`.
#   remote_job.sh <round> [steps]
# steps: on (default env, lever 3 on), off (GSPLAT_TT_FUSED_PAIRS=0, kill switch = base path)
# Each run: 30 bicycle views untraced, md5 vs md5-r82new.txt, stage lines.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
S=/localdev/smarton/t105_scripts; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
F=${T105_TREE:-/localdev/smarton/gstt2-t105}
r=${1:-1}; shift
steps=${*:-on off}
run() {  # tag [env...]; non-zero if the run failed
  local tag=$1; shift
  echo "=== $tag $(cut -c1-7 $F/SHA) $*"
  ( cd $F && source .venv/bin/activate && mkdir -p tmp &&
    find . -maxdepth 3 -type d -name t105-dump-$tag -exec rm -rf {} + ; rm -f $S/md5-$tag.txt;
    env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $F) timeout 600 \
      python3 render/run.py --no-ref --iter-dir t105-$tag --dump-views t105-dump-$tag \
      > tmp/t105-run-$tag.log 2>&1
    rc=$?; echo "run rc=$rc"
    grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|error:|hard fail" tmp/t105-run-$tag.log | head -12
    d=$(find . -maxdepth 3 -type d -name t105-dump-$tag | head -1)
    [ -n "$d" ] && (cd $d && md5sum * | sort -k2) > $S/md5-$tag.txt
    echo "$tag views: $(wc -l < $S/md5-$tag.txt 2>/dev/null || echo 0)"
    diff $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL" ||
      { echo "VIEWS DIFFER"; diff $REF $S/md5-$tag.txt | head -4; }
    exit $rc )
}
for s in $steps; do
  case $s in
    on)  run t105r$r-on ;;
    off) run t105r$r-off GSPLAT_TT_FUSED_PAIRS=0 ;;
  esac || echo "STEP $s FAILED"
done
echo "=== done round $r"
