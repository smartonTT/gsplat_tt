#!/bin/bash
# t114 driver (Mac side), run inside `ttp lock p100 -- `. Steps 1-5 of the A/B plan.
#   drive.sh <rev> [phase]   phase: all (default) | chk | ab | tracy
set -o pipefail
REV=${1:?rev}; PH=${2:-all}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t114
J="T114_TREE=$D bash $D/docs/sort-onelaunch-t114/remote_job.sh"
CHK=chk:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_SORT_ONELAUNCH_CHECK=1,TT_METAL_WATCHER=2
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
if [ $PH = all ] || [ $PH = chk ]; then
  out=$(ssh -o BatchMode=yes $H "VIEWS=0:5 TMO=600 $J c $CHK"); rc=$?; echo "$out"; echo "chk rc=$rc"
  [ $rc = 0 ] || { echo CHK_FAIL; exit 2; }
  echo "$out" | grep -q 'bad_tiles=[1-9]' && { echo CHK_BAD_TILES; exit 3; }
  echo "$out" | grep -q 'VIEWS DIFFER' && { echo CHK_VIEWS_DIFFER; exit 4; }
  echo "$out" | grep -q 'ALL_VIEWS_IDENTICAL' || { echo CHK_NO_DUMP; exit 5; }
fi
if [ $PH = all ] || [ $PH = ab ]; then
  ssh -o BatchMode=yes $H "$J h on on on"; echo "hang rc=$?"
  ssh -o BatchMode=yes $H "$J 1 base on"; echo "r1 rc=$?"
  ssh -o BatchMode=yes $H "$J 2 on base"; echo "r2 rc=$?"
  ssh -o BatchMode=yes $H "$J 3 base on"; echo "r3 rc=$?"
  ssh -o BatchMode=yes $H "$J m ms:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_MAT_STATS=1"; echo "ms rc=$?"
fi
if [ $PH = all ] || [ $PH = tracy ]; then
  ssh -o BatchMode=yes $H "T114_TREE=$D bash $D/docs/sort-onelaunch-t114/remote_tracy.sh"; echo "tracy rc=$?"
fi
echo ALLDONE
