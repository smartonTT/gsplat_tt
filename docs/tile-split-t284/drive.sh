#!/bin/bash
# t284 Mac-side driver: one ttp lock p100 per device step.
#   drive.sh [rev]: sync base (6a7dae5) + new, build; 2 ABBA rounds of the 30-view sweep
#   (md5 + ms/view); far pose (dolly -2) at 1/255 and 1/16384, a pulled-back-4 pose, and
#   the 1/255 far pose under GSPLAT_TT_TEST_TILE_CAP=20000 (floor fallback still works);
#   then the device hero screenshot (opt/ttw/screenshot.sh, NO_SYNC, new tree).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t284; TB=/localdev/smarton/gstt2-t284b
P=docs/tile-split-t284
O=$P/out; mkdir -p $O
rev=${1:-HEAD}; base=${BASE_REV:-6a7dae50}
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $TB "$base" > $O/sync-base.log 2>&1; rc=$?; echo "SYNC_BASE_RC=$rc"; tail -3 $O/sync-base.log
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
lk opt/sync_remote.sh $H $T "$rev" > $O/sync-new.log 2>&1; rc=$?; echo "SYNC_NEW_RC=$rc"; tail -3 $O/sync-new.log
[ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
arm() { lk $DEVRUN --host $H --no-verify --timeout 900 --tag t284-r$1 -- "bash $T/$P/remote_t284.sh $1 ${*:2}"; echo "ROUND_r$1_RC=$?"; }
arm 1 base:base new:new new2:new base2:base
arm 2 new:new base:base base2:base new2:new
arm 3 far:far2_255:-2:255 far:far2_16k:-2:16384 far:far4_255:-4:255 far:far2_255cap:-2:255:GSPLAT_TT_TEST_TILE_CAP=20000
scp -q -o BatchMode=yes "$H:$T/tmp/t284/run-*.log" "$H:$T/tmp/t284/md5-*.txt" "$H:$T/tmp/t284/hero-*.png" "$H:$T/tmp/t284/view-*" $O/ 2>/dev/null
NO_SYNC=1 T=$T NAME=t284 opt/ttw/screenshot.sh t284 "$rev"; echo "SHOT_RC=$?"
echo CHAIN_DONE
