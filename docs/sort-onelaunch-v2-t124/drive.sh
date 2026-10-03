#!/bin/bash
# t124 driver (Mac side), run inside `ttp lock p100 -- `. Steps 1-5 of the A/B plan.
#   drive.sh <rev> [phase]   phase: all (default) | chk | ab | tracy
set -o pipefail
REV=${1:?rev}; PH=${2:-all}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t124
J="T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_job.sh"
# The watcher build of PFWC_VIS (~66 KB) overflows the 70.6 KB kernel config buffer
# (76128 B), so the checked run has no watcher; a second watcher run uses SFPU_VIS=0.
CHK=chk:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_SORT_ONELAUNCH_CHECK=1
CHKW=chkw:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_SORT_ONELAUNCH_CHECK=1,GSPLAT_TT_SFPU_VIS=0,TT_METAL_WATCHER=2
# v2 arms. "on" = GSPLAT_TT_SORT_ONELAUNCH=1 with the v2 defaults (OL_PB=8 OL_RING=8
# OL_MAT_SELECT=1). KILL = v1 one-launch (#114). NOSEL = v2 emit only. V1SEL = select only.
KILL=kill:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_PB=1,GSPLAT_TT_OL_RING=0,GSPLAT_TT_OL_MAT_SELECT=0
NOSEL=nosel:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_MAT_SELECT=0
V1SEL=v1sel:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_OL_PB=1,GSPLAT_TT_OL_RING=0
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
if [ $PH = all ] || [ $PH = chk ]; then
  out=$(ssh -o BatchMode=yes $H "VIEWS=0:5 TMO=600 $J c $CHK"); rc=$?; echo "$out"; echo "chk rc=$rc"
  [ $rc = 0 ] || { echo CHK_FAIL; exit 2; }
  echo "$out" | grep -q 'bad_tiles=[1-9]' && { echo CHK_BAD_TILES; exit 3; }
  echo "$out" | grep -q 'VIEWS DIFFER' && { echo CHK_VIEWS_DIFFER; exit 4; }
  echo "$out" | grep -q 'ALL_VIEWS_IDENTICAL' || { echo CHK_NO_DUMP; exit 5; }
  out=$(ssh -o BatchMode=yes $H "VIEWS=0:3 TMO=600 $J w $CHKW"); rc=$?; echo "$out"; echo "chkw rc=$rc"
  echo "$out" | grep -q 'bad_tiles=[1-9]' && { echo CHKW_BAD_TILES; exit 6; }
fi
if [ $PH = all ] || [ $PH = ab ]; then
  ssh -o BatchMode=yes $H "$J h on on on"; echo "hang rc=$?"
  ssh -o BatchMode=yes $H "$J 1 base on"; echo "r1 rc=$?"
  ssh -o BatchMode=yes $H "$J 2 on base"; echo "r2 rc=$?"
  ssh -o BatchMode=yes $H "$J 3 base on"; echo "r3 rc=$?"
  # partial arms: v1 emit with select (kill switch minus select), v2 emit without select
  ssh -o BatchMode=yes $H "$J 4 $KILL $NOSEL"; echo "r4 rc=$?"
  ssh -o BatchMode=yes $H "$J 5 $V1SEL"; echo "r5 rc=$?"
  ssh -o BatchMode=yes $H "$J m ms:GSPLAT_TT_SORT_ONELAUNCH=1,GSPLAT_TT_MAT_STATS=1"; echo "ms rc=$?"
fi
if [ $PH = all ] || [ $PH = tracy ]; then
  ssh -o BatchMode=yes $H "T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh"; echo "tracy rc=$?"
  # fine materialize zones (mat_ol_keys/sort/gather/wr) to calibrate the worklist cost model
  ssh -o BatchMode=yes $H "MPROF=1 T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh t124-mprof"; echo "tracy-mprof rc=$?"
fi
echo ALLDONE
