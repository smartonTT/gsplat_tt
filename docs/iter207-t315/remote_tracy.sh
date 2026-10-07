#!/bin/bash
# t315: 30-view Tracy capture of iter 207 (defaults, MATCULL_TRISC_FILL on) in the t315 tree.
#   remote_tracy.sh <tag> [ENV=V ...]   run on yyzo-bh-04 through devrun.sh, under ttp lock p100.
set -u
T=${T:-/localdev/smarton/gstt2-t315}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t315-prof
tag=${1:?tag}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-8 SHA 2>/dev/null) env: $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback|TT_FATAL' | head -20
D=opt/profiler/$tag
[ -s $D/render.tracy ] || { echo "=== tracy FAIL $tag"; exit 2; }
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
grep -E "tile_blend|zone " $D/zones.txt | head -30
echo "=== tracy done $(date +%T)"
