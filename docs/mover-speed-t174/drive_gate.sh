#!/bin/bash
# t174 gate driver (Mac side; copy of t166 drive_gate.sh). ONE `ttp lock p100` hold:
# sync + build, 3 interleaved rounds x 30 views of
#   tip  = GSPLAT_TT_PRECULL=1 (the current smarton/tt-project-opt default; md5 must
#          equal md5-r82new.txt),
#   base = the new default (PRECULL=2 + GSPLAT_TT_OL_MOVER_SPEED=1),
#   rows = PRECULL=2 + t166 row split (GSPLAT_TT_OL_MOVER_SPEED=0),
# then per-view PSNR of base vs tip and md5 of base across rounds and vs rows.
#   ttp lock p100 -- bash docs/mover-speed-t174/drive_gate.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t166
J="T162_TREE=$D bash $D/docs/precull-default-t162/remote_job.sh"
TIP=tip:GSPLAT_TT_PRECULL=1
ROWS=rows:GSPLAT_TT_PRECULL=2,GSPLAT_TT_OL_MOVER_SPEED=0
r() { ssh -o BatchMode=yes $H "$@"; }
step() {
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
r "cat $D/SHA" | grep -q "$(git rev-parse $REV)" || { echo "SHA mismatch"; exit 1; }
step smoke "VIEWS=0:2 TMO=600 $J 0 base"
step r1 "$J 1 $TIP base $ROWS"
step r2 "$J 2 $ROWS base $TIP"
step r3 "$J 3 base $TIP $ROWS"
for k in 1 2 3; do
  r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t162-dump-t162r$k-tip tmp/t162-dump-t162r$k-base" \
    | awk -v k=$k '{print "psnr r" k " " $0}'
done
# base must be deterministic and equal the t166 PRECULL=2 image (46a725ab...) and the rows arm
r "cd /localdev/smarton/t162_scripts && md5sum md5-t162r1-base.txt md5-t162r2-base.txt md5-t162r3-base.txt md5-t162r1-rows.txt"
echo ALLDONE
