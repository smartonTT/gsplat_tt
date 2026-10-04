#!/bin/bash
# t150: 30-view Tracy device capture of the tip, then zone / gap / per-RISC tables.
#   remote_tracy.sh [tag]   (run on yyzo-bh-07 through devrun.sh, under ttp lock p100)
set -u
T=${T150_TREE:-/localdev/smarton/gstt2-t150}; cd "$T" || exit 1; source .venv/bin/activate
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t150-prof
tag=${1:-t150-tip}
echo "=== tracy $tag $(cut -c1-7 SHA) $(date +%T)"
timeout 450 bash opt/profiler/capture_tracy.sh $tag 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|^SUMMARY|^STAGES|^SORT_STAGES|Traceback' | head -20
D=opt/profiler/$tag
python3 opt/profiler/stitch_device_csv.py -o $D/dev30.csv $D/profile_log_device.csv 2>&1 | tail -3
python3 opt/profiler/zone_occupancy.py $D/dev30.csv > $D/zone_occupancy.txt 2>&1; echo "zone_occ rc=$?"
python3 opt/profiler/program_gaps.py $D/dev30.csv --min-gap-us 1 > $D/gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/analyze_zones.py $D/dev30.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
python3 opt/profiler/risc_roofline.py $D/dev30.csv > $D/roofline.txt 2>&1; echo "roofline rc=$?"
cat $D/gaps.txt; cat $D/roofline.txt | head -120
echo "=== tracy done $(date +%T)"
