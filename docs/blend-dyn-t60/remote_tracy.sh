#!/bin/bash
# One 10-view Tracy chunk (views 0:10) for one tree + zone table.
# Usage: remote_tracy.sh <variant>   (tree = /localdev/smarton/gstt2-<variant>)
set -u
V=$1
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=/localdev/smarton/gstt2-$V
cd $T || exit 1
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-$V-prof
timeout 330 bash opt/profiler/capture_tracy.sh t60-$V 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY' | head
source .venv/bin/activate
python3 opt/profiler/analyze_zones.py opt/profiler/t60-$V/chunks/0-10/profile_log_device.csv 10 > opt/profiler/t60-$V/zones.txt 2>&1
grep -E "^zone|tile_mb_mask|tile_l1_cull_rd|tile_blend|rd_l1_bulk|sort_subchunk_mat|TRISC-KERNEL|NCRISC-KERNEL|BRISC-KERNEL" opt/profiler/t60-$V/zones.txt
