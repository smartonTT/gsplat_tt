#!/bin/bash
# t290: land GSPLAT_TT_PFWC_RECIP_NEWTON default-on. Mac side, one ttp lock p100 per device step.
#   drive.sh [rev]: sync + build, 3 rotated rounds def (NEWTON default 1) vs off (=0),
#   then the device screenshot (opt/ttw/screenshot.sh, NO_SYNC) at defaults.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t290
P=docs/land-t290
O=$P/out; mkdir -p $O
OFF=GSPLAT_TT_PFWC_RECIP_NEWTON=0
rev=${1:-HEAD}
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T "$rev"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
arm() { lk $DEVRUN --host $H --no-verify --timeout 540 --tag t290-r$1 -- "bash $T/$P/remote_time.sh $1 ${*:2}"; echo "TIME_r$1_RC=$?"; }
arm 1 def:255 off:255:$OFF
arm 2 off:255:$OFF def:255
arm 3 def:255 off:255:$OFF
scp -q -o BatchMode=yes "$H:$T/tmp/t290/run-*.log" "$H:$T/tmp/t290/md5-*.txt" "$H:$T/tmp/t290/hero-*.png" $O/ 2>/dev/null
NO_SYNC=1 T=$T opt/ttw/screenshot.sh ${ITER:?ITER} "$rev"; echo "SHOT_RC=$?"
echo CHAIN_DONE
