#!/bin/bash
# t194: 30-view Tracy device capture (emit part counters off), then zone + program gap tables.
#   remote_tracy.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=${T154_TREE:-/localdev/smarton/gstt2-t194}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t194-prof
export GSPLAT_TT_OL_EMIT_PROF=0
tag=${1:-t194-on}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) env: $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag ${TRACY_VIEWS:-} 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|k2_fold|Traceback' | head -20
grep -B2 -A25 Traceback opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
C=$D/profile_log_device.csv; [ -n "${TRACY_VIEWS:-}" ] && C=$D/chunks/${TRACY_VIEWS/:/-}/profile_log_device.csv
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $C 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
cat $D/gaps.txt; grep -E "k2_|sort_ol|zone " $D/zones.txt; python3 $T/docs/tip-t194/fill.py $D/dev30.csv $D/capture.log | tee $D/fill.txt
echo "=== tracy done $(date +%T)"
