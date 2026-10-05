#!/bin/bash
# t230: re-baseline Tracy on the tip (BLEND_SCHED=2 default) at GSPLAT_TT_KCFG_EXTRA_KB=32.
# Mac side, each device step under ttp lock p100.   drive.sh [rev] [steps: sync prof]
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t230
P=docs/blend-trisc1-t230
O=$P/out; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync prof} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has prof; then
  n=t230-tip
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n GSPLAT_TT_KCFG_EXTRA_KB=32"
  echo "PROF_RC=$?"
  for f in zones.txt gaps.txt roofline.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/tracy-$f 2>/dev/null; done
  scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/dev30.csv "${TTP_RUN_DIR:-/tmp}/t230-dev30.csv" 2>/dev/null
  cat $O/tracy-gaps.txt 2>/dev/null
fi
echo CHAIN_DONE
