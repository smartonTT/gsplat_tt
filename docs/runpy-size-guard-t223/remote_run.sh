#!/bin/bash
# t223: run.py with/without --no-ref at 960 px (4 views) or 1024 px default (2 views); prints rc, SUMMARY, md5s.
#   remote_run.sh <tag> <args...>   (on yyzo-bh-07 via devrun.sh, under ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
T=/localdev/smarton/gstt2-t223; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t223; mkdir -p $S
tag=$1; shift
echo "=== run $tag $(cut -c1-7 SHA) $* $(date +%T)"
rm -rf tmp/t223-dump-$tag
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t223 timeout 600 \
  python3 render/run.py "$@" --iter-dir t223-$tag --dump-views t223-dump-$tag > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc"
grep -E "^SUMMARY|WARNING|Traceback|Segmentation|TT_THROW|TT_FATAL" $S/run-$tag.log | head -8
d=$(find . -maxdepth 3 -type d -name t223-dump-$tag | head -1)
[ -n "$d" ] && (cd "$d" && md5sum * | sed "s/^/MD5 $tag /")
echo "=== done $tag rc=$rc $(date +%T)"
exit $rc
