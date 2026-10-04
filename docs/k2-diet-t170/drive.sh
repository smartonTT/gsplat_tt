#!/bin/bash
# t170: sync + build, untraced md5/timing (default = diet K2 + fold vs GSPLAT_TT_K2_DIET=0,
# 2 rounds, order swapped; diet without fold GSPLAT_TT_K2_FOLD=0 once, paired with the
# default), then Tracy captures of the default and of GSPLAT_TT_K2_DIET=0.
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
# Stops after round 1 if the default arm is not md5-identical to md5-r82new.txt.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t170
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/k2-diet-t170/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
lk() {  # one device step under the p100 lock; a busy lock ends the chain (rc 75)
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo "LOCK_BUSY"; echo CHAIN_DONE; exit 75; fi
  return $rc
}
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t170/run-r*.log" "$H:$T/tmp/t170/md5-r*.txt" $O/ 2>/dev/null; }
if [[ $STEPS == *sync* ]]; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t170-time1 -- \
    "bash $T/docs/k2-diet-t170/remote_time.sh 1 base off:GSPLAT_TT_K2_DIET=0"
  echo "TIME1_RC=$?"; fetch
  if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t170/md5-r1-base.txt" >/dev/null; then
    echo "MD5_GATE_FAIL (r1 base)"; echo CHAIN_DONE; exit 4
  fi
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t170-time2 -- \
    "bash $T/docs/k2-diet-t170/remote_time.sh 2 off:GSPLAT_TT_K2_DIET=0 base"
  echo "TIME2_RC=$?"
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t170-time3 -- \
    "bash $T/docs/k2-diet-t170/remote_time.sh 3 nf:GSPLAT_TT_K2_FOLD=0 base"
  echo "TIME3_RC=$?"; fetch
fi
if [[ $STEPS == *tracy* ]]; then
  for arm in on off; do
    envs=""; [ $arm = off ] && envs="GSPLAT_TT_K2_DIET=0"
    lk $DEVRUN --host $H --no-verify --timeout 540 --tag t170-tracy-$arm -- \
      "bash $T/docs/k2-diet-t170/remote_tracy.sh t170-$arm $envs"
    echo "TRACY_${arm}_RC=$?"
    for f in zones.txt gaps.txt capture.log; do
      scp -q -o BatchMode=yes $H:$T/opt/profiler/t170-$arm/$f $O/tracy-$arm-$f
    done
  done
fi
echo CHAIN_DONE
