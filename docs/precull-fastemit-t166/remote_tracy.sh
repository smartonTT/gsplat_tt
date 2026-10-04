#!/bin/bash
# t166: 30-view Tracy device capture with the emit part counters on (env from the caller,
# e.g. GSPLAT_TT_PRECULL=1 / GSPLAT_TT_OL_EMIT_FAST=0), then zone + emit-part tables.
#   remote_tracy.sh <tag>   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=${T166_TREE:-/localdev/smarton/gstt2-t166}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t166-prof
export GSPLAT_TT_OL_EMIT_PROF=${GSPLAT_TT_OL_EMIT_PROF:-1}
tag=${1:?tag}
echo "=== tracy $tag $(cut -c1-7 SHA) EMIT_PROF=$GSPLAT_TT_OL_EMIT_PROF PRECULL=${GSPLAT_TT_PRECULL:-} FAST=${GSPLAT_TT_OL_EMIT_FAST:-} $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES|^SORT_STAGES|Traceback' | head -20
grep -B2 -A25 Traceback opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/emit_parts.py $D/dev30.csv 30 > $D/emit_parts.txt 2>&1; echo "emit_parts rc=$?"
cat $D/emit_parts.txt; grep -E "sort_ol|zone " $D/zones.txt
echo "=== tracy done $(date +%T)"
