#!/bin/bash
# t169 driver (Mac side): chunk frustum cull before pfwc (GSPLAT_TT_CHUNK_CULL).
# ONE `ttp lock p100` hold: sync + build, smoke, 3 rounds x 30 views of off
# (CHUNK_CULL=0, md5 must equal md5-r82new.txt) vs on (Morton reorder + skip),
# arm order alternated, a reorder-only arm (CHUNK_SKIP=0), per-view PSNR on vs off,
# then per-view stage timers of off and on.
#   ttp lock p100 -- bash docs/chunk-cull-t169/drive.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t169
J="T169_TREE=$D bash $D/docs/chunk-cull-t169/remote_job.sh"
OFF=off:GSPLAT_TT_CHUNK_CULL=0
ON=on:GSPLAT_TT_CHUNK_CULL=1
r() { ssh -o BatchMode=yes $H "$@"; }
step() {
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
step smoke "VIEWS=0:2 TMO=600 $J 0 on:GSPLAT_TT_CHUNK_CULL=1,GSPLAT_TT_CHUNK_LOG=1"
r "grep -m3 'chunk cull' $D/tmp/t169-run-t169r0-on.log"
step r1 "$J 1 $OFF $ON"
step r2 "$J 2 $ON $OFF"
step r3 "$J 3 $OFF $ON"
step r4 "$J 4 noskip:GSPLAT_TT_CHUNK_CULL=1,GSPLAT_TT_CHUNK_SKIP=0"
for k in 1 2 3; do
  r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t169-dump-t169r$k-off tmp/t169-dump-t169r$k-on" \
    | awk -v k=$k '{print "psnr r" k " " $0}'
done
r "cd /localdev/smarton/t169_scripts && md5sum md5-t169r1-on.txt md5-t169r2-on.txt md5-t169r3-on.txt"
step st "$J 5 stoff:GSPLAT_TT_CHUNK_CULL=0,GSPLAT_PER_VIEW_STAGES=1 ston:GSPLAT_TT_CHUNK_CULL=1,GSPLAT_PER_VIEW_STAGES=1"
echo ALLDONE
