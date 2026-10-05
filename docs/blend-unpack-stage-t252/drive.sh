#!/bin/bash
# t252: device gate of U2 unpacker-staged blend coefficients (GSPLAT_TT_BLEND_UNPACK_STAGE=1, on top of
# DECODE_AHEAD=2). Mac side, one ttp lock p100 per step (copy of t231 drive.sh).
#   drive.sh [rev] [steps]   steps: any of sync smoke 1 2 3 tracy
#   smoke: Stage 0 probe arm P (UNPACK_STAGE=1 + U2_PROBE=1, DPRINT TR1 "U2P n= bad= raw_bad=") and U1, md5-gated.
#   1-3:   rotated untraced 30-view rounds of def / U1, md5-gated.  tracy: def and U1 captures.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t252
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/blend-unpack-stage-t252
O=$P/out; mkdir -p $O tmp/t252
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() {
  scp -q -o BatchMode=yes "$H:$T/tmp/t252/run-r$1-$2.log" "$H:$T/tmp/t252/md5-r$1-$2.txt" $O/ 2>/dev/null
  scp -q -o BatchMode=yes "$H:$T/tmp/t252/dprint-r$1-$2.txt" $O/ 2>/dev/null
  scp -q -o BatchMode=yes "$H:$T/tmp/t252-r$1-$2/hero_clean.png" tmp/t252/hero-r$1-$2.png 2>/dev/null
}
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t252/md5-r$1-$2.txt" >/dev/null; }
gate() {
  local r=$1; shift; local bad=0
  for a in "$@"; do
    fetch $r $a
    if md5ok $r $a; then echo "MD5_OK r$r-$a"; else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
tm() {
  local r=$1 to=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t252-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
U1="U1:GSPLAT_TT_BLEND_UNPACK_STAGE=1"
DEF="def:GSPLAT_TT_DUMMY=0"
STEPS=" ${2:-sync smoke 1 2 3 tracy} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  RT=300 tm s 560 P $U1
  gate s P U1
fi
for r in 1 2 3; do
  has $r || continue
  if [ $((r % 2)) -eq 1 ]; then tm $r 360 $DEF $U1; else tm $r 360 $U1 $DEF; fi
  gate $r def U1
done
if has tracy; then
  for a in def U1; do
    e=GSPLAT_TT_DUMMY=0; [ $a = U1 ] && e=GSPLAT_TT_BLEND_UNPACK_STAGE=1
    lk $DEVRUN --host $H --no-verify --timeout 560 --tag t252-tracy-$a -- "bash $T/$P/remote_tracy.sh t252-$a $e"
    echo "TRACY_${a}_RC=$?"
  done
fi
echo CHAIN_DONE
