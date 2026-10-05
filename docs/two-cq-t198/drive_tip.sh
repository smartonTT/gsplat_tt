#!/bin/bash
# t198 confirm on the rebased project tip with both switches on by default: sync + build;
# smoke (defaults + GSPLAT_TT_SORT_ONELAUNCH_CHECK=1, md5-gated); 3 paired untraced 30-view
# rounds "base" (GSPLAT_TT_SORT_OL_EARLY=0, i.e. both off) vs "both" (defaults), order
# swapped each round, md5-gated vs md5-r82new.txt; then a 30-view Tracy capture of the defaults.
#   drive_tip.sh [rev] [steps]   (Mac; one ttp lock p100 per device step)
#   steps: any of sync smoke 1 2 3 tracy (default: all, in that order)
# Every devrun step stays under its 600 s ceiling (smoke 400 s, rounds 300 s, Tracy 540 s).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t198
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/two-cq-t198/out-tip; mkdir -p $O
OFF=GSPLAT_TT_SORT_OL_EARLY=0
DEF=GSPLAT_TT_T198_DEFAULTS=1   # no-op variable: the "both" arm runs the defaults
CK=GSPLAT_TT_SORT_ONELAUNCH_CHECK=1
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
gate() {  # round arm...: fetch logs, fail the chain on any md5 mismatch
  local r=$1; shift; local bad=0
  for a in "$@"; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t198/run-r$r-$a.log" "$H:$T/tmp/t198/md5-r$r-$a.txt" $O/ 2>/dev/null
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t198/md5-r$r-$a.txt" >/dev/null; then echo "MD5_OK r$r-$a"
    else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
STEPS=" ${2:-sync smoke 1 2 3 tracy} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t198-tsmoke -- \
    "RUN_TO=240 bash $T/docs/two-cq-t198/remote_time.sh ts defchk:$DEF,$CK"
  echo "SMOKE_RC=$?"; gate ts defchk
fi
ord=("base:$OFF both:$DEF" "both:$DEF base:$OFF" "base:$OFF both:$DEF")
for r in 1 2 3; do
  has $r || continue
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t198-ttime$r -- \
    "RUN_TO=150 bash $T/docs/two-cq-t198/remote_time.sh t$r ${ord[$((r-1))]}"
  echo "TIME${r}_RC=$?"; gate t$r base both
done
has tracy || { echo CHAIN_DONE; exit 0; }
lk $DEVRUN --host $H --no-verify --timeout 540 --tag t198-ttracy -- \
  "bash $T/docs/two-cq-t198/remote_tracy.sh t198-tip"
echo "TRACY_RC=$?"
for f in zones.txt gaps.txt fill.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/t198-tip/$f $O/tracy-$f; done
echo CHAIN_DONE
