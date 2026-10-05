#!/bin/bash
# t188: sync + build, untraced md5/timing of three arms (off = LPT + early claim,
# late = LPT + late claim, base = default late claim + record-count-descending order),
# 3 rounds with the arm order rotated, then one Tracy capture of the default.
#   drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
# Stops after round 1 if any arm is not md5-identical to md5-r82new.txt.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t188
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/blend-claim-t188/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
OFF=off:GSPLAT_TT_BLEND_LATE_CLAIM=0,GSPLAT_TT_BLEND_CLAIM_DESC=0
LATE=late:GSPLAT_TT_BLEND_CLAIM_DESC=0
lk() {  # one device step under the p100 lock; a busy lock ends the chain (rc 75)
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo "LOCK_BUSY"; echo CHAIN_DONE; exit 75; fi
  return $rc
}
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t188/run-r*.log" "$H:$T/tmp/t188/md5-r*.txt" $O/ 2>/dev/null; }
if [[ $STEPS == *sync* ]]; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
# One devrun per arm (each run.py is capped at 330 s remotely), so each reservation
# stays under the 600 s devrun ceiling.
arm() {  # round armspec
  lk $DEVRUN --host $H --no-verify --timeout 420 --tag t188-r$1-${2%%:*} -- \
    "bash $T/docs/blend-claim-t188/remote_time.sh $1 $2"
  echo "TIME_r$1_${2%%:*}_RC=$?"
}
if [[ $STEPS == *time* ]]; then
  for a in $OFF $LATE base; do arm 1 $a; done; fetch
  for a in off late base; do
    if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t188/md5-r1-$a.txt" >/dev/null; then
      echo "MD5_GATE_FAIL (r1 $a)"; echo CHAIN_DONE; exit 4
    fi
  done
  echo "MD5_R1_OK"
  for a in base $OFF $LATE; do arm 2 $a; done
  for a in $LATE base $OFF; do arm 3 $a; done; fetch
fi
if [[ $STEPS == *tracy* ]]; then
  lk $DEVRUN --host $H --no-verify --timeout 540 --tag t188-tracy-on -- \
    "bash $T/docs/blend-claim-t188/remote_tracy.sh t188-on"
  echo "TRACY_on_RC=$?"
  for f in zones.txt gaps.txt capture.log step1.txt; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t188-on/$f $O/tracy-on-$f
  done
fi
echo CHAIN_DONE
