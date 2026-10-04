#!/bin/bash
# t164: sync + build, untraced md5/timing (fold on vs GSPLAT_TT_OL_EMIT_FOLD=0, ROUNDS rounds,
# order swapped), then the emit-part Tracy capture with the fold.
#   [ROUNDS="1 2"] drive.sh [rev] [steps=sync,time,tracy]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t164
O=docs/emit-fold-t164/out; mkdir -p $O
STEPS=${2:-sync,time,tracy}
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  # ROUNDS (default "1 2"): odd rounds run the fold first, even rounds FOLD=0 first.
  for r in ${ROUNDS:-1 2}; do
    if (( r % 2 )); then arms="base off:GSPLAT_TT_OL_EMIT_FOLD=0"; else arms="off:GSPLAT_TT_OL_EMIT_FOLD=0 base"; fi
    ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t164-time$r -- \
      "bash $T/docs/emit-fold-t164/remote_time.sh $r $arms"
    echo "TIME${r}_RC=$?"
  done
  scp -q -o BatchMode=yes "$H:$T/tmp/t164/run-r*.log" "$H:$T/tmp/t164/md5-r*.txt" $O/
fi
if [[ $STEPS == *tracy* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 540 --tag t164-tracy -- \
    "bash $T/docs/emit-fold-t164/remote_tracy.sh t164-ep"
  echo "TRACY_RC=$?"
  for f in zones.txt gaps.txt emit_parts.txt capture.log; do
    scp -q -o BatchMode=yes $H:$T/opt/profiler/t164-ep/$f $O/$f
  done
fi
echo CHAIN_DONE
