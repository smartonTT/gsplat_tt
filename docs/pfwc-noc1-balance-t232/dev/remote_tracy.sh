#!/bin/bash
# t232 (copy of t231 remote_tracy.sh): 30-view Tracy capture at default env (P2 + RD_REST=0x0F), then zone,
# program-gap and per-RISC tables.  remote_tracy.sh <tag> [ENV=V ...]
# Run on yyzo-bh-07 through devrun.sh, under ttp lock p100.
set -u
T=/localdev/smarton/gstt2-t232; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t232-tr
export GSPLAT_TT_OL_EMIT_PROF=0
tag=${1:-t232-def}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) env: $* $(date +%T)"
mkdir -p opt/profiler/$tag; timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee opt/profiler/$tag/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback' | head -20
grep -B2 -A25 Traceback opt/profiler/$tag/capture.log | head -60
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/risc_roofline.py $D/dev30.csv --skip-first > $D/roofline.txt 2>&1; echo "roofline rc=$?"
cat $D/gaps.txt; grep -E "tile_blend|zone " $D/zones.txt
echo "=== tracy done $(date +%T)"
