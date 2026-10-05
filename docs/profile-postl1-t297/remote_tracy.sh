#!/bin/bash
# t297: 30-view Tracy device capture + per-core post-L1 timeline (copy of t289 remote_tracy.sh + postl1_cores.py).
#   remote_tracy.sh <tag> [ENV=V ...]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=/localdev/smarton/gstt2-t297; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t297-prof
export GSPLAT_TT_OL_EMIT_PROF=0
tag=${1:-t297-ns}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-7 SHA) env: $* $(date +%T)"
D=opt/profiler/$tag; mkdir -p $D
timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee $D/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback' | head -20
grep -B2 -A25 Traceback $D/capture.log | head -60
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/matblend_cores.py $D/dev30.csv --skip-first > $D/cores.txt 2>&1; echo "cores rc=$?"
python3 opt/profiler/postl1_cores.py $D/dev30.csv --percore $D/percore.csv > $D/postl1.txt 2>&1; echo "postl1 rc=$?"
cat $D/gaps.txt $D/cores.txt $D/postl1.txt; grep -E "zone " $D/zones.txt | head
mkdir -p tmp/t297; cp $D/capture.log tmp/t297/tracy-$tag-capture.log; cp $D/gaps.txt tmp/t297/tracy-$tag-gaps.txt
cp $D/zones.txt tmp/t297/tracy-$tag-zones.txt; cp $D/postl1.txt tmp/t297/tracy-$tag-postl1.txt; cp $D/percore.csv tmp/t297/tracy-$tag-percore.csv; cp $D/cores.txt tmp/t297/tracy-$tag-cores.txt
echo "=== tracy done $(date +%T)"
