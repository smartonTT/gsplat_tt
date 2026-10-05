#!/bin/bash
# t147 driver (Mac side): tip stage timings (30 views) + per-tile blend MATH
# cycles (GSPLAT_TT_MB_TILECYC=1, DPRINT on TRISC1, views 0:2) for the fused
# materialize+blend model. One `ttp lock p100` hold:
#   ttp lock p100 -- bash docs/fuse-matblend-t147/drive_tc.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t147
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
echo "##### job $(date +%H:%M:%S)"
ssh -o BatchMode=yes $H "bash $D/docs/fuse-matblend-t147/remote_tc.sh"
rc=$?
echo "##### job rc=$rc $(date +%H:%M:%S)"
scp -q -o BatchMode=yes $H:$D/tmp/t147-tc.dprint docs/fuse-matblend-t147/out/t147-tc.dprint
exit $rc
