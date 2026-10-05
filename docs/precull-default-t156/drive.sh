#!/bin/bash
# t156 driver (Mac side): review of lever C's default flip (GSPLAT_TT_PRECULL=1
# default, 0 = kill switch). ONE `ttp lock p100` hold: sync + build, 3 rounds x
# 30 views of off (the pre-#156 default; md5 must equal the old golden) vs base
# (the new default), arm order alternated, then per-view PSNR of base vs off.
#   ttp lock p100 -- bash docs/precull-default-t156/drive.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t156
J="T156_TREE=$D bash $D/docs/precull-default-t156/remote_job.sh"
OFF=off:GSPLAT_TT_PRECULL=0
r() { ssh -o BatchMode=yes $H "$@"; }
step() {
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
step smoke "VIEWS=0:2 TMO=600 $J 0 base"
step r1 "$J 1 $OFF base"
step r2 "$J 2 base $OFF"
step r3 "$J 3 $OFF base"
for k in 1 2 3; do
  r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t156-dump-t156r$k-off tmp/t156-dump-t156r$k-base" \
    | awk -v k=$k '{print "psnr r" k " " $0}'
done
# determinism of the new default across rounds (the md5 golden refresh source)
r "cd /localdev/smarton/t156_scripts && md5sum md5-t156r1-base.txt md5-t156r2-base.txt md5-t156r3-base.txt"
echo ALLDONE
