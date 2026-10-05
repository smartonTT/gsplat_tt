#!/bin/bash
# t86: 10-view Tracy chunk (views 0:10) with the fine mat/cull zones
# (GSPLAT_TT_MATCULL_PROF=1) + zone table + program gaps + zone windows.
set -u
source /localdev/smarton/t86_scripts/remote_env.sh
tag=${1:-t86-prof}
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-$(basename $T)-prof GSPLAT_TT_MATCULL_PROF=1 GSPLAT_TT_MAT_STATS=1
timeout 400 bash opt/profiler/capture_tracy.sh $tag 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES|MAT|mat' | head -20
csv=opt/profiler/$tag/chunks/0-10/profile_log_device.csv
python3 opt/profiler/analyze_zones.py $csv 10 > opt/profiler/$tag/zones.txt 2>&1
python3 opt/profiler/program_gaps.py $csv --skip-first --min-gap-us 1 > opt/profiler/$tag/gaps.txt 2>&1
python3 opt/profiler/zone_windows.py $csv --skip-first > opt/profiler/$tag/windows.txt 2>&1
ls -la $csv; cat opt/profiler/$tag/zones.txt opt/profiler/$tag/gaps.txt opt/profiler/$tag/windows.txt
