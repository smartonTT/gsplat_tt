#!/bin/bash
# t218 (copy of t202 remote_smoke.sh): 2-view bicycle smoke (hang check) with an env, md5 vs md5-r82new.txt; exit 5 on a miss.
#   remote_smoke.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t218; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t218; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
tag=$1; shift
echo "=== smoke $tag $(cut -c1-7 SHA) $* $(date +%T)"
rm -rf tmp/t218-dump-$tag
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t218 timeout 300 \
  python3 render/run.py --no-ref --view-range 0:2 --iter-dir t218-$tag --dump-views t218-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^SUMMARY|\[SORT\] ONELAUNCH|Traceback|TT_THROW|TT_FATAL|hard fail" $S/run-$tag.log | head -8
d=$(find . -maxdepth 3 -type d -name t218-dump-$tag | head -1)
n=0; ok=0
if [ -n "$d" ]; then
  for f in "$d"/*; do [ -f "$f" ] || continue; n=$((n+1)); grep -q "$(md5sum < "$f" | cut -c1-32)" $REF && ok=$((ok+1)); done
  rm -rf "$d"
fi
echo "MD5 $ok of $n views in md5-r82new.txt"
echo "=== smoke done $tag rc=$rc $(date +%T)"
[ $rc -eq 0 ] || exit $rc
[ $n -gt 0 ] && [ $ok -eq $n ] || exit 5
