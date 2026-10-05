#!/bin/bash
# t198: 2-CQ bridge hiding. Sync + build; a smoke round (early and both with
# GSPLAT_TT_SORT_ONELAUNCH_CHECK=1, md5-gated); 4 paired untraced 30-view rounds of
# off / early / both (arm order rotated), every run md5-gated vs md5-r82new.txt;
# then one 30-view Tracy capture of "both" (program gaps, zones).
#   drive.sh [rev] [steps]   (Mac; one ttp lock p100 per device step)
#   steps: any of sync smoke 1 2 3 4 tracy (default: all, in that order)
# devrun refuses --timeout over 600 s (reservation ceiling); a warm 30-view run is ~15-25 s,
# so each step stays at <=400 s (the first run in a fresh kernel cache also JIT-compiles).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t198
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/two-cq-t198/out; mkdir -p $O
E=GSPLAT_TT_SORT_OL_EARLY=1
B=GSPLAT_TT_SORT_OL_EARLY=1,GSPLAT_TT_MAT_CQ1=1
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
STEPS=" ${2:-sync smoke 1 2 3 4 tracy} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t198-smoke -- \
    "RUN_TO=240 bash $T/docs/two-cq-t198/remote_time.sh s earlychk:$E,$CK bothchk:$B,$CK"
  echo "SMOKE_RC=$?"; gate s earlychk bothchk
fi
ord=("base early:$E both:$B" "early:$E both:$B base" "both:$B base early:$E" "base early:$E both:$B")
for r in 1 2 3 4; do
  has $r || continue
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t198-time$r -- \
    "RUN_TO=150 bash $T/docs/two-cq-t198/remote_time.sh $r ${ord[$((r-1))]}"
  echo "TIME${r}_RC=$?"; gate $r base early both
done
has tracy || { echo CHAIN_DONE; exit 0; }
lk $DEVRUN --host $H --no-verify --timeout 540 --tag t198-tracy -- \
  "bash $T/docs/two-cq-t198/remote_tracy.sh t198-both GSPLAT_TT_SORT_OL_EARLY=1 GSPLAT_TT_MAT_CQ1=1"
echo "TRACY_RC=$?"
for f in zones.txt gaps.txt fill.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/t198-both/$f $O/tracy-$f; done
echo CHAIN_DONE
