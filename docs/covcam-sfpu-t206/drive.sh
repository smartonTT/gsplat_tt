#!/bin/bash
# t218: device check of t206 (GSPLAT_TT_PFWC_COVCAM_SFPU). Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps]   steps: any of sync smoke pc0 pc1 1 2 3 dflt (default: sync smoke)
# smoke: 2-view md5 with the knob on. pcN: STEPCYC=2 split (t197 flow) with the knob at N.
# 1 2 3: swapped untraced 30-view rounds base/sfpu, every run md5-gated. dflt: default vs =0.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t218
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/covcam-sfpu-t206
O=$P/out; mkdir -p $O
KN=GSPLAT_TT_PFWC_COVCAM_SFPU
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
gate() {  # round arm...: fetch logs, fail the chain on any md5 mismatch
  local r=$1; shift; local bad=0
  for a in "$@"; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t218/run-r$r-$a.log" "$H:$T/tmp/t218/md5-r$r-$a.txt" $O/ 2>/dev/null
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t218/md5-r$r-$a.txt" >/dev/null; then echo "MD5_OK r$r-$a"
    else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
STEPS=" ${2:-sync smoke} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t218-smoke -- "bash $T/$P/remote_smoke.sh sfpu $KN=1"
  rc=$?; echo "SMOKE_RC=$rc"
  scp -q -o BatchMode=yes "$H:$T/tmp/t218/run-sfpu.log" $O/smoke-sfpu.log 2>/dev/null
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit 5; }
fi
for k in 0 1; do
  has pc$k || continue
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag t218-pc$k -- "bash $T/$P/remote_prof.sh t218-pc2-k$k 2 $KN=$k"
  echo "PC_K${k}_RC=$?"
  D=$T/opt/profiler/t218-pc2-k$k
  for x in capture.log pc_split.txt dev.csv.gz; do scp -q -o BatchMode=yes "$H:$D/$x" $O/pc2-k$k-$x 2>/dev/null; done
done
ord=("base sfpu:$KN=1" "sfpu:$KN=1 base" "base sfpu:$KN=1")
for r in 1 2 3; do
  has $r || continue
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t218-time$r -- \
    "RUN_TO=130 bash $T/$P/remote_time.sh $r ${ord[$((r-1))]}"
  echo "TIME${r}_RC=$?"; gate $r base sfpu
done
if has dflt; then
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t218-dflt -- \
    "RUN_TO=130 bash $T/$P/remote_time.sh d base off:$KN=0"
  echo "DFLT_RC=$?"; gate d base off
fi
echo CHAIN_DONE
