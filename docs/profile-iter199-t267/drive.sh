#!/bin/bash
# t267: profile the iter-199 tip. Mac side, one ttp lock p100 per step (copy of t258 drive.sh).
#   drive.sh [rev] [steps]   steps: any of sync v1 v2 repro prof
#   v1, v2: untraced 30-view rounds: default, then default + host stage profile; md5-gated vs md5-r82new.txt.
#   repro:  Tracy at GSPLAT_TT_KCFG_EXTRA_KB=4 (the old capture_tracy.sh default): expected TT_FATAL.
#   prof:   Tracy of the default with GSPLAT_TT_KCFG_EXTRA_KB unset (host auto-sizes: +24 +8 profiler).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t267
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/profile-iter199-t267
O=$P/out; mkdir -p $O tmp/t267
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
STEPS=" ${2:-sync v1 prof} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
bad=0
for r in v1 v2; do
  has $r || continue
  lk $DEVRUN --host $H --no-verify --timeout 360 --tag t267-$r -- "RUN_TO=150 bash $T/$P/remote_time.sh $r def:GSPLAT_TT_DUMMY=0 hp:GSPLAT_TT_HOST_PROFILE=1,GSPLAT_PER_VIEW_STAGES=1"
  echo "TIME_${r}_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t267/run-r$r-*.log" "$H:$T/tmp/t267/md5-r$r-*.txt" $O/
  for a in def hp; do
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t267/md5-r$r-$a.txt" >/dev/null; then echo "MD5_OK r$r-$a"; else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
done
[ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
tr() {  # name [ENV=V]
  local n=$1; shift
  lk $DEVRUN --host $H --no-verify --timeout 560 --tag $n -- "bash $T/$P/remote_tracy.sh $n $*"
  echo "PROF_${n}_RC=$?"
  for f in zones.txt gaps.txt roofline.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/$f $O/$n-$f 2>/dev/null; done
  mkdir -p tmp/t267; scp -q -o BatchMode=yes $H:$T/opt/profiler/$n/dev30.csv tmp/t267/$n-dev30.csv 2>/dev/null; echo "CSV_RC=$?"
}
has repro && tr t267-repro GSPLAT_TT_KCFG_EXTRA_KB=4
has prof && tr t267-def  # GSPLAT_TT_KCFG_EXTRA_KB unset: verifies t258's auto-size (d5508ae)
echo CHAIN_DONE
