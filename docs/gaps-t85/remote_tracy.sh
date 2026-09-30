#!/bin/bash
# t85: 10-view Tracy chunk (views 0:10) of the fix tree + zone table + program gaps.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T85_TREE:-/localdev/smarton/gstt2-t85}; cd $T || exit 1; source .venv/bin/activate
tag=${1:-t85-fix}
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-t85-prof
timeout 330 bash opt/profiler/capture_tracy.sh $tag 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES' | head
csv=opt/profiler/$tag/chunks/0-10/profile_log_device.csv
python3 opt/profiler/program_gaps.py $csv --skip-first --min-gap-us 1 > opt/profiler/$tag/gaps.txt 2>&1
ls -la $csv opt/profiler/$tag/chunks/0-10/*.tracy; cat opt/profiler/$tag/gaps.txt
