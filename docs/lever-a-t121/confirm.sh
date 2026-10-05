#!/bin/bash
# t121 default-on confirm (Mac side), one `ttp lock p100 -- ` hold: sync <rev>
# (GSPLAT_TT_SORT_ONELAUNCH default on) and run 3 rotated rounds of
#   dflt   (no env: one-launch v2, OL_MAT_SELECT=1)
#   nosel  (GSPLAT_TT_OL_MAT_SELECT=0: v2 emit, v1 big-tile items)
#   legacy (GSPLAT_TT_SORT_ONELAUNCH=0: the kill switch, multi-launch sort)
# Every arm must be md5-identical to md5-r82new.txt over 30 views.
#   confirm.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t121
J="T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_job.sh"
NS=nosel:GSPLAT_TT_OL_MAT_SELECT=0
LG=legacy:GSPLAT_TT_SORT_ONELAUNCH=0
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
for rr in "d1 base $NS $LG" "d2 $NS $LG base" "d3 $LG base $NS"; do
  echo "##### $rr $(date +%H:%M:%S)"
  ssh -o BatchMode=yes $H "$J $rr"; rc=$?; echo "##### rc=$rc"
  [ $rc = 0 ] || { echo STOP; exit 10; }
done
echo ALLDONE
