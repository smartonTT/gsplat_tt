#!/bin/bash
# t232 (copy of t228 remote_prof.sh): pfwc per-step cycle split from a Tracy device capture.
#   remote_prof.sh <tag> <stepcyc> [ENV=V ...]   views ${V:-0:4}, all 5 RISCs (STEPRISC=9)
set -u
T=/localdev/smarton/gstt2-t232; cd "$T" || exit 1; source .venv/bin/activate
tag=$1; L=$2; shift 2; V=${V:-0:4}
export TTW_DEVRUN=1 GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t232-prof
export GSPLAT_TT_PFWC_STEPCYC=$L GSPLAT_TT_PFWC_STEPRISC=9 GSPLAT_TT_KCFG_EXTRA_KB=${KX:-10}
for kv in "$@"; do export "$kv"; done
echo "=== $tag views $V $(cut -c1-7 SHA) $* $(date +%T)"
D=opt/profiler/$tag; mkdir -p $D
timeout 450 bash opt/profiler/capture_tracy.sh $tag $V > $D/capture.log 2>&1; echo "rc=$?"
grep -E 'capture_tracy\] (OK|FAIL|DONE)|TT_FATAL|too large|Traceback|error' $D/capture.log | head -12
C=$D/chunks/${V/:/-}/profile_log_device.csv
python3 docs/pfwc-writer-split-t207/dev-t221/pc_split.py $C | tee $D/pc_split.txt
gzip -c $C > $D/dev.csv.gz
echo "=== done $(date +%T)"
