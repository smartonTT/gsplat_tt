#!/bin/bash
# t196: sync + build, 2-view smoke (bulk blendrec on; stop on a hang or md5 miss), 3 untraced
# 30-view A/B rounds (base = default = bulk on, off = GSPLAT_TT_OL_BREC_BULK=0; order rotated;
# md5 gate on both arms), then Tracy views 0:10 with GSPLAT_TT_OL_EMIT_PROF=1 for both arms.
#   drive.sh [rev] [steps=sync,smoke,time,tracy] [rounds="1 2 3"]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t196
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/emit-imbalance-t196/out; mkdir -p $O
STEPS=${2:-sync,smoke,time,tracy}
lk() {  # one device step under the p100 lock; a busy lock ends the chain (rc 75)
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo "LOCK_BUSY"; echo CHAIN_DONE; exit 75; fi
  return $rc
}
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t196/run-*.log" "$H:$T/tmp/t196/md5-r*.txt" $O/ 2>/dev/null; }
if [[ $STEPS == *sync* ]]; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *smoke* ]]; then
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t196-smoke -- "bash $T/docs/emit-imbalance-t196/remote_smoke.sh on"
  rc=$?; echo "SMOKE_RC=$rc"; fetch
  [ $rc -eq 0 ] || { echo "SMOKE_FAIL"; echo CHAIN_DONE; exit 4; }
fi
if [[ $STEPS == *time* ]]; then
  for r in ${3:-1 2 3}; do
    if [ $((r % 2)) -eq 1 ]; then arms="base off:GSPLAT_TT_OL_BREC_BULK=0"; else arms="off:GSPLAT_TT_OL_BREC_BULK=0 base"; fi
    lk $DEVRUN --host $H --no-verify --timeout 560 --tag t196-time$r -- \
      "bash $T/docs/emit-imbalance-t196/remote_time.sh $r $arms"
    echo "TIME${r}_RC=$?"; fetch
    for a in base off; do
      if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t196/md5-r$r-$a.txt" >/dev/null; then
        echo "MD5_GATE_FAIL (r$r $a)"; echo CHAIN_DONE; exit 4
      fi
    done
  done
fi
if [[ $STEPS == *tracy* ]]; then
  for arm in on off; do
    envs=""; [ $arm = off ] && envs="GSPLAT_TT_OL_BREC_BULK=0"
    lk $DEVRUN --host $H --no-verify --timeout 540 --tag t196-tracy-$arm -- \
      "bash $T/docs/emit-imbalance-t196/remote_tracy.sh t196-$arm $envs"
    echo "TRACY_${arm}_RC=$?"
    for f in zones.txt gaps.txt emit_parts.txt movers.txt capture.log; do
      scp -q -o BatchMode=yes $H:$T/opt/profiler/t196-$arm/$f $O/tracy-$arm-$f
    done
  done
fi
echo CHAIN_DONE
