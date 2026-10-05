#!/bin/bash
# t187: untraced bicycle 30-view run of one tree, md5 vs md5-r82new.txt.
#   remote_run.sh <tag> <tree> [ENV=V ...]   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
# Trees: /localdev/smarton/gstt2-t187a (f4d91df, stack cur_lm), -t187b (e8754ac, curp),
#        -t187f (t177 cd707df + the fix, fold kernel).
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
tag=$1; T=$2; shift 2
cd "$T" || exit 1; source .venv/bin/activate
S=/localdev/smarton/t187-out; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
echo "=== $tag $(cut -c1-7 SHA) $T $* $(date +%T)"
rm -rf tmp/t187-dump-$tag
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T) timeout ${RUN_TIMEOUT:-330} \
  python3 render/run.py --no-ref --iter-dir t187-$tag --dump-views t187-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL" $S/run-$tag.log | head -12
d=$(find . -maxdepth 3 -type d -name t187-dump-$tag | head -1)
if [ -n "$d" ]; then
  (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
  diff -q $REF $S/md5-$tag.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL ($(wc -l < $S/md5-$tag.txt) views)" \
    || echo "VIEWS DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"
  rm -rf "$d"
fi
exit $rc
