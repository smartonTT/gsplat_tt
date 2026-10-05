#!/bin/bash
# t181: 2-view bicycle smoke (hang check) with an env, md5 vs md5-r82new.txt.
#   remote_smoke.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=${T154_TREE:-/localdev/smarton/gstt2-t181}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t181; mkdir -p $S
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
tag=$1; shift
echo "=== smoke $tag $(cut -c1-7 SHA) $* $(date +%T)"
rm -rf tmp/t181-dump-$tag $S/dprint-$tag.txt
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t181 timeout 200 \
  python3 render/run.py --no-ref --view-range 0:2 --iter-dir t181-$tag --dump-views t181-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^SUMMARY|Traceback|TT_THROW|TT_FATAL" $S/run-$tag.log | head -5
d=$(find . -maxdepth 3 -type d -name t181-dump-$tag | head -1)
if [ -n "$d" ]; then
  n=0; ok=0
  for f in "$d"/*; do [ -f "$f" ] || continue; n=$((n+1)); grep -q "$(md5sum < "$f" | cut -c1-32)" $REF && ok=$((ok+1)); done
  echo "MD5 $ok of $n views in md5-r82new.txt"; rm -rf "$d"
fi
[ -f $S/dprint-$tag.txt ] && { echo "dprint lines: $(wc -l < $S/dprint-$tag.txt)"; grep -c "bad 0" $S/dprint-$tag.txt; grep -v "bad 0" $S/dprint-$tag.txt | head -20; grep "bad 0" $S/dprint-$tag.txt | head -3; }
echo "=== smoke done $tag rc=$rc $(date +%T)"
exit $rc
