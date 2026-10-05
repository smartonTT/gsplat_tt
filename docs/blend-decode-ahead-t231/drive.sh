#!/bin/bash
# t231: device gate of TRISC0 decode-ahead staging (GSPLAT_TT_BLEND_DECODE_AHEAD 0 / 1 = S2a / 2 = S2)
# plus the zero-build calibration arm GSPLAT_TT_BLEND_RAW_STAGE=1 (t230 models +0.27 ms/view).
# Mac side, one ttp lock p100 per step, one sync + build for every arm (the knobs are JIT defines).
#   drive.sh [rev] [steps]   steps: any of sync smoke 1 2 3
#   smoke: 30 views at DA 1 and 2, md5-gated. A ring deadlock shows up as a run timeout without dumps.
#   1-3:   rotated untraced 30-view rounds of def / DA1 / DA2 / RAW, md5-gated (two devrun calls of
#          two arms each: devrun refuses a timeout over 600 s).
# Every run also saves its device hero render (run.py tmp/<iter-dir>/hero_clean.png); the chain
# fetches them to tmp/t231/hero-r<round>-<arm>.png (untracked) for the iteration's screenshot.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t231
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/blend-decode-ahead-t231
O=$P/out; mkdir -p $O tmp/t231
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
fetch() {
  scp -q -o BatchMode=yes "$H:$T/tmp/t231/run-r$1-$2.log" "$H:$T/tmp/t231/md5-r$1-$2.txt" $O/ 2>/dev/null
  scp -q -o BatchMode=yes "$H:$T/tmp/t231-r$1-$2/hero_clean.png" tmp/t231/hero-r$1-$2.png 2>/dev/null
}
md5ok() { ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t231/md5-r$1-$2.txt" >/dev/null; }
gate() {  # round arm...: fetch logs, fail the chain on any md5 mismatch
  local r=$1; shift; local bad=0
  for a in "$@"; do
    fetch $r $a
    if md5ok $r $a; then echo "MD5_OK r$r-$a"; else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
tm() {  # round timeout arms...  (devrun refuses timeouts over 600 s: at most 2 arms per call)
  local r=$1 to=$2; shift 2
  lk $DEVRUN --host $H --no-verify --timeout $to --tag t231-time$r -- "RUN_TO=${RT:-150} bash $T/$P/remote_time.sh $r $*"
  echo "TIME${r}_RC=$?"
}
arm() {  # name -> remote_time arm spec
  case $1 in
    def) echo "def:GSPLAT_TT_DUMMY=0" ;;
    DA1) echo "DA1:GSPLAT_TT_BLEND_DECODE_AHEAD=1" ;;
    DA2) echo "DA2:GSPLAT_TT_BLEND_DECODE_AHEAD=2" ;;
    RAW) echo "RAW:GSPLAT_TT_BLEND_RAW_STAGE=1" ;;
  esac
}
STEPS=" ${2:-sync smoke 1 2 3} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  RT=180 tm s 400 $(arm DA1) $(arm DA2)
  gate s DA1 DA2
fi
ord=("def DA1 DA2 RAW" "DA2 RAW def DA1" "RAW def DA1 DA2")
for r in 1 2 3; do
  has $r || continue
  set -- ${ord[$((r-1))]}
  tm $r 360 $(arm $1) $(arm $2); tm $r 360 $(arm $3) $(arm $4); gate $r def DA1 DA2 RAW
done
echo CHAIN_DONE
