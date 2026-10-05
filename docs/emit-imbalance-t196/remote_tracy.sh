#!/bin/bash
# t196: Tracy device capture of views ${TRACY_VIEWS:-0:10} with the emit part counters on
# (GSPLAT_TT_OL_EMIT_PROF=1), then zones, program gaps, emit parts and per-mover tables.
#   remote_tracy.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=/localdev/smarton/gstt2-t196; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t196-prof
export GSPLAT_TT_OL_EMIT_PROF=${EMIT_PROF:-1}
tag=${1:-t196-on}; shift
for kv in "$@"; do export "$kv"; done
V=${TRACY_VIEWS:-0:10}; NV=$(( ${V#*:} - ${V%:*} ))
echo "=== tracy $tag $(cut -c1-7 SHA) views $V env: EMIT_PROF=$GSPLAT_TT_OL_EMIT_PROF $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag $V 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback' | head -20
grep -B2 -A25 Traceback opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/chunks/${V/:/-}/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv $NV > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/emit_parts.py $D/dev30.csv $NV > $D/emit_parts.txt 2>&1; echo "emit_parts rc=$?"
python3 docs/emit-imbalance-t196/movers.py $D/dev30.csv $D/capture.log > $D/movers.txt 2>&1; echo "movers rc=$?"
cat $D/gaps.txt; grep -E "sort_ol|zone " $D/zones.txt; cat $D/emit_parts.txt; head -40 $D/movers.txt
echo "=== tracy done $(date +%T)"
