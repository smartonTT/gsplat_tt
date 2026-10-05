#!/bin/bash
# t289: 30-view Tracy device capture + per-core mat/blend timeline (model: docs/tip-t199/remote_tracy.sh).
#   remote_tracy.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=/localdev/smarton/gstt2-t289; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t289-prof
export GSPLAT_TT_OL_EMIT_PROF=0
tag=${1:-t289-fuse}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) env: $* $(date +%T)"
D=opt/profiler/$tag; mkdir -p $D
timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee $D/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback' | head -20
grep -B2 -A25 Traceback $D/capture.log | head -60
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/matblend_cores.py $D/dev30.csv --skip-first > $D/cores.txt 2>&1; echo "cores rc=$?"
cat $D/gaps.txt $D/cores.txt; grep -E "zone " $D/zones.txt | head
mkdir -p tmp/t289; cp $D/capture.log tmp/t289/tracy-$tag-capture.log; cp $D/gaps.txt tmp/t289/tracy-$tag-gaps.txt
cp $D/zones.txt tmp/t289/tracy-$tag-zones.txt; cp $D/cores.txt tmp/t289/tracy-$tag-cores.txt
echo "=== tracy done $(date +%T)"
