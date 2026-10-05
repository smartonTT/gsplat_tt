#!/bin/bash
# t214 review of t202: sync the reviewed tree to an own remote dir, then two 2-view
# bicycle md5 smokes with defaults on (OL_EMIT_TOWN=1): default OL_PB=8, and
# GSPLAT_TT_OL_PB=1 (16-pair batches: ~8x more batches, so the 4-slot READY/DONE
# reuse, fl waits and queue service run far more often). Both must match md5 46a725ab.
#   drive.sh [rev]   (Mac; one ttp lock p100 for the whole sync+run sequence)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t214
P=docs/emit-trisc-own-t200/review-t214
O=$P/out; mkdir -p $O
ttp lock p100 -- bash -c "
  opt/sync_remote.sh $H $T ${1:-HEAD} || exit 1
  $DEVRUN --host $H --no-verify --timeout 400 --tag t214-smoke-d -- 'bash $T/$P/remote_smoke.sh dflt' || exit 5
  $DEVRUN --host $H --no-verify --timeout 400 --tag t214-smoke-pb1 -- 'bash $T/$P/remote_smoke.sh pb1 GSPLAT_TT_OL_PB=1' || exit 6
"
rc=$?
for a in dflt pb1; do scp -q -o BatchMode=yes "$H:$T/tmp/t214/run-$a.log" $O/smoke-$a.log 2>/dev/null; done
echo "CHAIN_RC=$rc"
exit $rc
