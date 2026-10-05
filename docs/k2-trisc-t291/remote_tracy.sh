#!/bin/bash
# t291 (copy of t274 remote_tracy.sh): 30-view Tracy capture with the K2 part counters on
# (GSPLAT_TT_K2_PROF=1), then zone, gap and k2_parts tables.  remote_tracy.sh <tag> [ENV=V ...]
# Run on yyzo-bh-07 through devrun.sh, under ttp lock p100.
set -u
T=${T291_TREE:-/localdev/smarton/gstt2-t291}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t291-prof
export GSPLAT_TT_OL_EMIT_PROF=0 GSPLAT_TT_K2_PROF=1
tag=${1:-t291-k2p}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) K2_PROF=$GSPLAT_TT_K2_PROF env: $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback' | head -20
grep -B2 -A25 -E "Traceback|TT_FATAL|error:" opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/k2_parts.py $D/dev30.csv 30 > $D/k2_parts.txt 2>&1; echo "k2_parts rc=$?"
grep -c ',k2p_' $D/dev30.csv | sed 's/^/k2p rows: /'
cat $D/k2_parts.txt; grep -E "k2_|sort_ol|zone " $D/zones.txt
echo "=== tracy done $(date +%T)"
