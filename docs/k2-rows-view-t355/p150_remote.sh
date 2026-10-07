#!/bin/bash
# t355: remote half of the p150 A/B on bh-30 (viewer stopped). Untraced bicycle 30-view sweep,
# alternating arms base (K2 rows 4 KB view on) / off (GSPLAT_TT_K2_ROWS_VIEW=0), N rounds;
# md5 of each arm's 30 dumped views vs the 906e0435 golden list. Pattern: docs/p150-bench-t346.
#   p150_remote.sh <rounds>   -> $P/out/{run-rN-ARM.log,md5-rN-ARM.txt,hero-r1-base.png,bench.rc}
set -u
P=/localdev/smarton/p150bench-t355; V=/localdev/smarton/viewer
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal   # read-only use of the viewer's build
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1
cd $P/tree || exit 1; source .venv/bin/activate
O=$P/out; mkdir -p $O
REF=$P/tree/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) $(date -u +%FT%TZ)"
echo "--- device users before run:"; fuser -v /dev/tenstorrent/* 2>&1 | tail -n +2 || true
tt-smi -ls 2>/dev/null | grep -E "Blackhole" | head -2
rc=0
for r in $(seq 1 ${1:-3}); do
  arms="base off"; [ $((r % 2)) = 0 ] && arms="off base"
  for a in $arms; do
    tag=t355-r$r-$a; rm -rf tmp/$tag tmp/$tag-dump
    env=(); [ $a = off ] && env=(GSPLAT_TT_K2_ROWS_VIEW=0)
    echo "=== r$r-$a start $(date -u +%T) ${env[*]}"
    env "${env[@]}" TT_METAL_CACHE_RENDER=$P/cache timeout ${RUN_TIMEOUT:-330} \
      python3 render/run.py --no-ref --iter-dir $tag --dump-views $tag-dump > $O/run-r$r-$a.log 2>&1
    rr=$?; echo "run r$r-$a rc=$rr $(date -u +%T)"
    grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/run-r$r-$a.log | head -6
    if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG r$r-$a: tt-smi -r"; tt-smi -r > $O/reset-r$r.log 2>&1; echo "reset rc=$?"; rc=124; break 2; fi
    [ $rr = 0 ] || { rc=$rr; break 2; }
    (cd tmp/$tag-dump && md5sum * | sort -k2) > $O/md5-r$r-$a.txt; rm -rf tmp/$tag-dump
    [ $r = 1 ] && [ $a = base ] && cp tmp/$tag/hero_clean.png $O/hero-r1-base.png
    echo "VIEWS=$(wc -l < $O/md5-r$r-$a.txt) SWEEP_MD5=$(md5sum < $O/md5-r$r-$a.txt | cut -c1-8)"
    diff -q $REF $O/md5-r$r-$a.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$r-$a" || echo "VIEWS_DIFFER r$r-$a ($(diff $REF $O/md5-r$r-$a.txt | grep -c '^>'))"
  done
done
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc
