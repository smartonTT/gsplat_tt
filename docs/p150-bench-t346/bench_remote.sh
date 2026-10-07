#!/bin/bash
# t346: remote half of the p150 bench on bh-30 (viewer stopped). Untraced bicycle 30-view
# sweep at best-iter-207 defaults, same command as iter-207 (docs/iter207-t315/remote_time.sh,
# base arm), N rounds; per round md5 of the 30 dumped views vs the 906e0435 golden list.
#   bench_remote.sh <rounds>   -> $P/out/{run-rN.log,md5-rN.txt,hero-rN.png,bench.rc}
set -u
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal   # read-only use of the viewer's build
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1
cd $P/tree || exit 1; source .venv/bin/activate
O=$P/out; mkdir -p $O
REF=$P/tree/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) $(date -u +%FT%TZ)"
echo "--- device users before run:"; fuser -v /dev/tenstorrent/* 2>&1 | tail -n +2 || true
tt-smi -ls 2>/dev/null | grep -E "Blackhole" | head -2
rc=0
for r in $(seq 1 ${1:-1}); do
  tag=t346-r$r; rm -rf tmp/$tag tmp/$tag-dump
  echo "=== r$r start $(date -u +%T)"
  TT_METAL_CACHE_RENDER=$P/cache timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $tag --dump-views $tag-dump > $O/run-r$r.log 2>&1
  rr=$?; echo "run r$r rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL" $O/run-r$r.log | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG r$r: tt-smi -r"; tt-smi -r > $O/reset-r$r.log 2>&1; echo "reset rc=$?"; rc=124; break; fi
  [ $rr = 0 ] || { rc=$rr; break; }
  (cd tmp/$tag-dump && md5sum * | sort -k2) > $O/md5-r$r.txt; rm -rf tmp/$tag-dump
  cp tmp/$tag/hero_clean.png $O/hero-r$r.png
  echo "VIEWS=$(wc -l < $O/md5-r$r.txt) SWEEP_MD5=$(md5sum < $O/md5-r$r.txt | cut -c1-8)"
  diff -q $REF $O/md5-r$r.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$r" || echo "VIEWS_DIFFER r$r ($(diff $REF $O/md5-r$r.txt | grep -c '^>'))"
done
grep -m2 -E "firmware bundle version|KMD version" $O/run-r1.log
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc
