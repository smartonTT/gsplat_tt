#!/bin/bash
# t148 driver (Mac side): tip 30-view timing, then the GSPLAT_TT_MB_STATS=1 waste
# counters over the same 30 bicycle views. One `ttp lock p100` hold:
#   ttp lock p100 -- bash docs/blend-waste-t148/drive.sh <rev>
set -o pipefail
REV=${1:?rev}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t148
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
echo "##### job $(date +%H:%M:%S)"
ssh -o BatchMode=yes $H "bash $D/docs/blend-waste-t148/remote.sh"
rc=$?
echo "##### job rc=$rc $(date +%H:%M:%S)"
scp -q -o BatchMode=yes "$H:$D/tmp/t148-*.log" "$H:$D/tmp/t148-stats.dprint" docs/blend-waste-t148/out/
exit $rc
