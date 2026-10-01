#!/bin/bash
# t114: 10-view Tracy chunk (views 0:10) with GSPLAT_TT_SORT_ONELAUNCH=1, program gaps
# and per-zone breakdown (sort_ol_*, mat_ol_* zones). From docs/gaps-t85/remote_tracy.sh.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T114_TREE:-/localdev/smarton/gstt2-t114}; cd $T || exit 1; source .venv/bin/activate
tag=${1:-t114-ol}
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-t114-prof GSPLAT_TT_SORT_ONELAUNCH=${OL:-1}
timeout 400 bash opt/profiler/capture_tracy.sh $tag 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES' | head
csv=opt/profiler/$tag/chunks/0-10/profile_log_device.csv
python3 opt/profiler/program_gaps.py $csv --skip-first --min-gap-us 1 > opt/profiler/$tag/gaps.txt 2>&1
python3 opt/profiler/analyze_zones.py $csv 10 > opt/profiler/$tag/zones.txt 2>&1
python3 opt/profiler/deep_zones.py $csv > opt/profiler/$tag/deep.txt 2>&1
ls -la $csv opt/profiler/$tag/chunks/0-10/*.tracy
echo "--- gaps"; head -60 opt/profiler/$tag/gaps.txt
echo "--- zones"; head -80 opt/profiler/$tag/zones.txt
echo "--- deep (sort_ol / mat_ol)"; grep -iE 'sort_ol|mat_ol|onelaunch|materialize' opt/profiler/$tag/deep.txt | head -60
