#!/bin/bash
# 10-view Tracy chunk (views 0:10) for ablation <a> + zone table.
set -u
source /localdev/smarton/t78_scripts/remote_env.sh
a=$1
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=$(cache $a -prof) GSPLAT_TT_BLEND_ABL=$a
timeout 330 bash opt/profiler/capture_tracy.sh t78-a$a 0:10 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY' | head
python3 opt/profiler/analyze_zones.py opt/profiler/t78-a$a/chunks/0-10/profile_log_device.csv 10 > opt/profiler/t78-a$a/zones.txt 2>&1
grep -E "^zone|tile_mb_mask|tile_blend|rd_l1_bulk|TRISC-KERNEL|NCRISC-KERNEL|BRISC-KERNEL" opt/profiler/t78-a$a/zones.txt
