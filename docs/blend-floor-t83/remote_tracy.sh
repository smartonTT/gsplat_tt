#!/bin/bash
# 10-view Tracy chunk (views 0:10) for ablation <a> with the fine blend zones
# (GSPLAT_TT_BLEND_PROF=1) + zone table + per-program windows.
set -u
source /localdev/smarton/t83_scripts/remote_env.sh
a=$1; tag=t83-a$a${2:-}
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=$(cache $a -prof) GSPLAT_TT_BLEND_ABL=$a GSPLAT_TT_BLEND_PROF=1
timeout 330 bash opt/profiler/capture_tracy.sh $tag 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES' | head
csv=opt/profiler/$tag/chunks/0-10/profile_log_device.csv
python3 opt/profiler/analyze_zones.py $csv 10 > opt/profiler/$tag/zones.txt 2>&1
python3 opt/profiler/program_gaps.py $csv --skip-first --min-gap-us 1 > opt/profiler/$tag/gaps.txt 2>&1
ls -la $csv; tail -5 opt/profiler/$tag/gaps.txt
