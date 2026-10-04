#!/bin/bash
# t162 driver (Mac side): review of the pixel-centre pre-cull default flip
# (GSPLAT_TT_PRECULL=2 default; 1 = lever C integer rect, 0 = off). ONE
# `ttp lock p100` hold: sync + build, 3 rounds x 30 views of old (PRECULL=1, the
# pre-#162 default; md5 must equal the current golden md5-r82new.txt) vs base
# (the new default), arm order alternated, then per-view PSNR of base vs old.
#   ttp lock p100 -- bash docs/precull-default-t162/drive.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t162
J="T162_TREE=$D bash $D/docs/precull-default-t162/remote_job.sh"
OLD=old:GSPLAT_TT_PRECULL=1
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
step r1 "$J 1 $OLD base"
step r2 "$J 2 base $OLD"
step r3 "$J 3 $OLD base"
for k in 1 2 3; do
  r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t162-dump-t162r$k-old tmp/t162-dump-t162r$k-base" \
    | awk -v k=$k '{print "psnr r" k " " $0}'
done
# determinism of the new default across rounds (the md5 golden refresh source)
r "cd /localdev/smarton/t162_scripts && md5sum md5-t162r1-base.txt md5-t162r2-base.txt md5-t162r3-base.txt"
echo ALLDONE
