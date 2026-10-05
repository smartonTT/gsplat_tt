#!/bin/bash
# t197 (remote, under ttp lock p100): pfwc per-step cycle split from a Tracy device capture.
#   remote_prof.sh <stepcyc=1|2> <steprisc=9|0..4> [views=0:4]
#   stepcyc 1 = per-step marks, 2 = also per-op split inside cov_cam; steprisc 9 = all 5 RISCs.
# DPRINT does not fit the kernel config buffer; the counters are profiler timestamped data.
set -u
T=/localdev/smarton/gstt2-t197; cd "$T" || exit 1; source .venv/bin/activate
L=${1:-1}; SR=${2:-9}; V=${3:-0:4}; tag=t197-pc$L-r$SR
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t197-prof
export GSPLAT_TT_PFWC_STEPCYC=$L GSPLAT_TT_PFWC_STEPRISC=$SR GSPLAT_TT_KCFG_EXTRA_KB=${KX:-10}
echo "=== $tag views $V $(cut -c1-7 SHA) $(date +%T)"
D=opt/profiler/$tag; mkdir -p $D
timeout 450 bash opt/profiler/capture_tracy.sh $tag $V > $D/capture.log 2>&1; echo "rc=$?"
grep -E 'capture_tracy\] (OK|FAIL|DONE)|TT_FATAL|too large|Traceback' $D/capture.log | head -8
C=$D/chunks/${V/:/-}/profile_log_device.csv
python3 docs/pfwc-breakdown-t197/pc_split.py $C | tee $D/pc_split.txt
gzip -c $C > $D/dev.csv.gz
echo "=== done $(date +%T)"
