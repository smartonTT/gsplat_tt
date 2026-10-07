#!/bin/bash
# t319: 30-view Tracy capture at defaults (MATCULL_TRISC_FILL on) with the fz_* counters, then
# the stitched CSV, analyze_zones and the fill_zones per-core report.
#   remote_tracy.sh <tag> [ENV=V ...]   run on yyzo-bh-04 through devrun.sh, under ttp lock p100.
set -u
T=${T:-/localdev/smarton/gstt2-t319}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t319-prof
tag=${1:?tag}; shift
for kv in "$@"; do export "$kv"; done
echo "=== tracy $tag $(cut -c1-8 SHA 2>/dev/null) env: $* $(date +%T)"
D=opt/profiler/$tag; rm -rf $D; mkdir -p $D
timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | tee $D/capture.log | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|Traceback|TT_FATAL' | head -20
rc=${PIPESTATUS[0]}
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in tracy: tt-smi -r"; tt-smi -r > $D/reset.log 2>&1; echo "reset rc=$?"; fi
[ -s $D/profile_log_device.csv ] || { echo "=== tracy FAIL $tag rc=$rc"; exit 2; }
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/fill_zones.py $D/dev30.csv --percore $D/percore.csv > $D/fill.txt 2>&1; echo "fill rc=$?"
cat $D/fill.txt
gzip -kf $D/dev30.csv
echo "=== tracy done $(date +%T)"
