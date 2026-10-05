#!/bin/bash
# t202: tile-owned TRISC emit pack (GSPLAT_TT_OL_EMIT_TOWN=1) gate. Sync + build; a 2-view
# smoke of TOWN (md5 vs md5-r82new.txt, stops on a hang or miss); Tracy views 0:10 with
# GSPLAT_TT_OL_EMIT_PROF=1 for town and base (TRISC counters town_pc, mover ep_*, emit zone);
# then, if the gate passes, 3 swapped untraced 30-view rounds base/town, every run md5-gated.
#   drive.sh [rev] [steps]   (Mac; one ttp lock p100 per device step)
#   steps: any of sync smoke tracy 1 2 3 dflt (default: sync smoke tracy)
# devrun refuses --timeout over 600 s; a warm 30-view run is ~15-25 s (first run JIT-compiles).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t202
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
P=docs/emit-trisc-own-t200
O=$P/out; mkdir -p $O
TW=GSPLAT_TT_OL_EMIT_TOWN=1
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
gate() {  # round arm...: fetch logs, fail the chain on any md5 mismatch
  local r=$1; shift; local bad=0
  for a in "$@"; do
    scp -q -o BatchMode=yes "$H:$T/tmp/t202/run-r$r-$a.log" "$H:$T/tmp/t202/md5-r$r-$a.txt" $O/ 2>/dev/null
    if ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t202/md5-r$r-$a.txt" >/dev/null; then echo "MD5_OK r$r-$a"
    else echo "MD5_GATE_FAIL r$r-$a"; bad=1; fi
  done
  [ $bad -eq 0 ] || { echo CHAIN_DONE; exit 4; }
}
STEPS=" ${2:-sync smoke tracy} "
has() { case "$STEPS" in *" $1 "*) return 0;; esac; return 1; }
if has sync; then
  lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if has smoke; then
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t202-smoke -- "bash $T/$P/remote_smoke.sh town $TW"
  rc=$?; echo "SMOKE_RC=$rc"
  scp -q -o BatchMode=yes "$H:$T/tmp/t202/run-town.log" $O/smoke-town.log 2>/dev/null
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit 5; }
fi
if has tracy; then
  for a in town base; do
    e=""; [ $a = town ] && e=$TW
    lk $DEVRUN --host $H --no-verify --timeout 540 --tag t202-tracy-$a -- \
      "KX=${KX:-} bash $T/$P/remote_tracy.sh t202-$a $e"
    echo "TRACY_${a}_RC=$?"
    for f in zones.txt gaps.txt emit_parts.txt town_parts.txt capture.log; do
      scp -q -o BatchMode=yes $H:$T/opt/profiler/t202-$a/$f $O/tracy-$a-$f 2>/dev/null
    done
  done
fi
ord=("base town:$TW" "town:$TW base" "base town:$TW")
for r in 1 2 3; do
  has $r || continue
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t202-time$r -- \
    "RUN_TO=130 bash $T/$P/remote_time.sh $r ${ord[$((r-1))]}"
  echo "TIME${r}_RC=$?"; gate $r base town
done
# dflt: after the default flip, default env (TOWN on) vs the kill switch (=0), both md5-gated.
if has dflt; then
  lk $DEVRUN --host $H --no-verify --timeout 300 --tag t202-dflt -- \
    "RUN_TO=130 bash $T/$P/remote_time.sh d base off:GSPLAT_TT_OL_EMIT_TOWN=0"
  echo "DFLT_RC=$?"; gate d base off
fi
echo CHAIN_DONE
