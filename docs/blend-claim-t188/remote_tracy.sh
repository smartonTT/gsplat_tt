#!/bin/bash
# t188: 30-view Tracy device capture (emit part counters off), then zone + program gap tables.
#   remote_tracy.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=${T154_TREE:-/localdev/smarton/gstt2-t188}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t188-prof
export GSPLAT_TT_OL_EMIT_PROF=0
tag=${1:-t188-on}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) env: $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|k2_fold|Traceback' | head -20
grep -B2 -A25 Traceback opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
cat $D/gaps.txt; grep -E "blend|zone " $D/zones.txt | head -40
python3 docs/blend-tail-t183/parse.py $D/dev30.csv $D/blend.pkl && python3 docs/blend-tail-t183/step1.py $D/blend.pkl > $D/step1.txt; tail -8 $D/step1.txt
echo "=== tracy done $(date +%T)"
