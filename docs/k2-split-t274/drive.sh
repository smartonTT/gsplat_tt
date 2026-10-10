#!/bin/bash
# t274: sync + K2 part-counter Tracy capture. Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps=sync prof]
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t274
P=docs/k2-split-t274
O=$P/out; mkdir -p $O tmp/t274
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync prof} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has prof; then
  n=t274-k2p
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n"
  echo "PROF_RC=$?"
  for f in zones.txt gaps.txt k2_parts.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/$n-$f 2>/dev/null; done
  scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/dev30.csv tmp/t274/$n-dev30.csv 2>/dev/null; echo "CSV_RC=$?"
fi
if has ab; then
  B=B:GSPLAT_TT_K2_TRISC=1
  for r in 1 2 3; do
    case $r in 2) arms="A $B";; *) arms="$B A";; esac
    lk $DEVRUN --host $H --no-verify --timeout 560 --tag t274-ab$r -- "bash $T/$P/remote_ab.sh $r $arms" > $O/ab-r$r.log 2>&1
    echo "AB${r}_RC=$?"; grep -E "^===|SUMMARY|SWEEP_MD5|rc=" $O/ab-r$r.log
    [ "$(grep -c "SWEEP_MD5=46a725ab" $O/ab-r$r.log)" = 2 ] || { echo "not both md5-clean in r$r, stop"; break; }
  done
fi
echo CHAIN_DONE
