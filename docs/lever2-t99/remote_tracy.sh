#!/bin/bash
# t102: 10-view Tracy chunk (views 0:10) of the lever-2 tree, zone table + program gaps.
#   remote_tracy.sh <tag> <GSPLAT_TT_SFPU_VIS value>   (run ON yyzo-bh-07, inside ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T99_TREE:-/localdev/smarton/gstt2-t99}; cd $T || exit 1; source .venv/bin/activate
tag=${1:-t102-vis}; export GSPLAT_TT_SFPU_VIS=${2:-1}
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t99-prof
timeout 400 bash opt/profiler/capture_tracy.sh $tag 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES' | head
csv=opt/profiler/$tag/chunks/0-10/profile_log_device.csv
python3 opt/profiler/analyze_zones.py $csv 10 > opt/profiler/$tag/zones.txt 2>&1
python3 opt/profiler/program_gaps.py $csv --skip-first --min-gap-us 1 > opt/profiler/$tag/gaps.txt 2>&1
ls -la $csv; cat opt/profiler/$tag/zones.txt; cat opt/profiler/$tag/gaps.txt
