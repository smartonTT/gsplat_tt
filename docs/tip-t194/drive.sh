#!/bin/bash
# t194: measure the combined tip (#181 bulk fill + #188 blend late claim): sync + build,
# 3 untraced 30-view rounds (default env, md5 vs md5-r82new.txt), then one 30-view
# default-chain Tracy capture (zones, program gaps, sort_ol fill/prefix/barrier/emit).
#   drive.sh [rev]   (Mac; one ttp lock p100 per device step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t194
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/tip-t194/out; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
for r in 1 2 3; do
  lk $DEVRUN --host $H --no-verify --timeout 400 --tag t194-time$r -- "bash $T/docs/tip-t194/remote_time.sh $r base"
  echo "TIME${r}_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t194/run-r$r-base.log" "$H:$T/tmp/t194/md5-r$r-base.txt" $O/ 2>/dev/null
  if ! ssh -o BatchMode=yes $H "diff -q $REF $T/tmp/t194/md5-r$r-base.txt" >/dev/null; then
    echo "MD5_GATE_FAIL (r$r)"; echo CHAIN_DONE; exit 4
  fi
done
lk $DEVRUN --host $H --no-verify --timeout 540 --tag t194-tracy -- "bash $T/docs/tip-t194/remote_tracy.sh t194-tip"
echo "TRACY_RC=$?"
for f in zones.txt gaps.txt fill.txt capture.log; do scp -q -o BatchMode=yes $H:$T/opt/profiler/t194-tip/$f $O/tracy-$f; done
echo CHAIN_DONE
