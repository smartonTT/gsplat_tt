#!/bin/bash
# t181 hang triage (after run 484 hung in sort_ol): sync + build, then one devrun per step,
# safest first: 2-view smoke with the kill switch, Tracy (kill switch, views 0:10, fill zones),
# 2-view smoke with the bulk fill, 2-view smoke with GSPLAT_TT_OL_FILL_BULK=2 (verify + DPRINT).
#   drive2.sh   (Mac; one ttp lock p100 per device step; every step runs even if one hangs)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t181
O=docs/sort-fill-t181/out2; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T HEAD; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
smoke() {  # tag [ENV=V ...]
  local tag=$1; shift
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t181-$tag -- "bash $T/docs/sort-fill-t181/remote_smoke.sh $tag $*"
  echo "SMOKE_${tag}_RC=$?"
}
smoke off GSPLAT_TT_OL_FILL_BULK=0
lk $DEVRUN --host $H --no-verify --timeout 540 --tag t181-tracy-off2 -- \
  "TRACY_VIEWS=0:10 bash $T/docs/sort-fill-t181/remote_tracy.sh t181-off2 GSPLAT_TT_OL_FILL_BULK=0"
echo "TRACY_off2_RC=$?"
for f in zones.txt gaps.txt fill.txt; do scp -q -o BatchMode=yes $H:$T/opt/profiler/t181-off2/$f $O/tracy-off2-$f; done
smoke on
smoke chk GSPLAT_TT_OL_FILL_BULK=2 TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_RISCVS=BR,NC \
  TT_METAL_DPRINT_FILE=$T/tmp/t181/dprint-chk.txt
scp -q -o BatchMode=yes "$H:$T/tmp/t181/run-*.log" "$H:$T/tmp/t181/dprint-*.txt" $O/ 2>/dev/null
echo CHAIN_DONE
