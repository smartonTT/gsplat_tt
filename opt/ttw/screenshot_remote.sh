#!/bin/bash
# Remote half of opt/ttw/screenshot.sh (runs on the device host through devrun.sh,
# under ttp lock p100). Renders the bicycle 30-view sweep on the device, keeps the
# hero view and the per-view md5 list.
#   screenshot_remote.sh <tree> <tag> [ENV=V ...]
# Writes <tree>/tmp/shot/{hero-<tag>.png, md5-<tag>.txt, run-<tag>.log}.
set -u
export TT_METAL_HOME=${TT_METAL_HOME:-/localdev/smarton/tt-metal}
export TT_METAL_RUNTIME_ROOT=${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${1:?tree}; tag=${2:?tag}; shift 2
cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/shot; mkdir -p $S
REF=${SHOT_MD5_REF:-/localdev/smarton/t82_scripts/md5-r82new.txt}
echo "=== shot $tag sha=$(cut -c1-7 SHA) env: $* $(date +%T)"
rm -rf tmp/shot-$tag tmp/shot-dump-$tag
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename "$T") timeout ${RUN_TO:-240} \
  python3 render/run.py --no-ref --iter-dir shot-$tag --dump-views shot-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^([A-Z_]*STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL|hard fail" $S/run-$tag.log | sort | uniq -c | sort -rn | head -8
[ $rc -eq 0 ] || exit $rc
cp tmp/shot-$tag/hero_clean.png $S/hero-$tag.png || exit 5
(cd tmp/shot-dump-$tag && md5sum * | sort -k2) > $S/md5-$tag.txt
rm -rf tmp/shot-dump-$tag
echo "VIEWS=$(wc -l < $S/md5-$tag.txt) SWEEP_MD5=$(md5sum < $S/md5-$tag.txt | cut -c1-8)"
if diff -q $REF $S/md5-$tag.txt > /dev/null; then echo "ALL_VIEWS_IDENTICAL"
else echo "VIEWS_DIFFER ($(diff $REF $S/md5-$tag.txt | grep -c '^>') of $(wc -l < $S/md5-$tag.txt))"; fi
