#!/bin/bash
# t191: sync + build the t190 chain walk, then untraced md5/timing of three arms
# (cw0 = GSPLAT_TT_BLEND_CHAIN_WALK=0 (default), cw1 = 1, cw2 = 2) on the same build,
# 3 rounds with the arm order rotated. Stops after round 1 if any arm is not
# md5-identical to md5-r82new.txt.
#   drive.sh [rev] [steps=sync,time]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t191
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/blend-chain-t191/out; mkdir -p $O
STEPS=${2:-sync,time}
C0=cw0:GSPLAT_TT_BLEND_CHAIN_WALK=0
C1=cw1:GSPLAT_TT_BLEND_CHAIN_WALK=1
C2=cw2:GSPLAT_TT_BLEND_CHAIN_WALK=2
lk() {  # one device step under the p100 lock; a busy lock ends the chain (rc 75)
  ttp lock p100 -- "$@"; local rc=$?
  if [ $rc -eq 75 ]; then echo "LOCK_BUSY"; echo CHAIN_DONE; exit 75; fi
  return $rc
}
fetch() { scp -q -o BatchMode=yes "$H:$T/tmp/t191/run-r*.log" "$H:$T/tmp/t191/md5-r*.txt" $O/ 2>/dev/null; }
if [[ $STEPS == *sync* ]]; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
# One devrun per arm (each run.py is capped at 330 s remotely), under the 600 s devrun ceiling.
arm() {  # round armspec
  lk $DEVRUN --host $H --no-verify --timeout 420 --tag t191-r$1-${2%%:*} -- \
    "bash $T/docs/blend-chain-t191/remote_time.sh $1 $2"
  echo "TIME_r$1_${2%%:*}_RC=$?"
}
if [[ $STEPS == *time* ]]; then
  for a in $C0 $C1 $C2; do arm 1 $a; done; fetch
  for a in cw0 cw1 cw2; do
    if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t191/md5-r1-$a.txt" >/dev/null; then
      echo "MD5_GATE_FAIL (r1 $a)"; echo CHAIN_DONE; exit 4
    fi
  done
  echo "MD5_R1_OK"
  for a in $C1 $C2 $C0; do arm 2 $a; done
  for a in $C2 $C0 $C1; do arm 3 $a; done; fetch
fi
echo CHAIN_DONE
