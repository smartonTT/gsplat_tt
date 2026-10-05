#!/bin/bash
# t217: 4-view bicycle at 960x960 (tiles_x=30, not a power of two) with an env; prints per-view md5 and SUMMARY.
#   remote_run.sh <tag> [ENV=V ...]   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t217; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t217; mkdir -p $S
tag=$1; shift
echo "=== run $tag $(cut -c1-7 SHA) $* $(date +%T)"
rm -rf tmp/t217-dump-$tag
env "$@" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t217 timeout 300 \
  python3 render/run.py --no-ref --cameras docs/emit-trisc-own-t200/review-t217/cameras_960.json \
  --view-range 0:4 --iter-dir t217-$tag --dump-views t217-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^SUMMARY|avg_frame|\[SORT\] ONELAUNCH|Traceback|TT_THROW|TT_FATAL|hard fail" $S/run-$tag.log | head -8
d=$(find . -maxdepth 3 -type d -name t217-dump-$tag | head -1)
[ -n "$d" ] && (cd "$d" && md5sum * | sed "s/^/MD5 $tag /")
echo "=== done $tag rc=$rc $(date +%T)"
exit $rc
