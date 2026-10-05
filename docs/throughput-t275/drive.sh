#!/bin/bash
# t275: latency vs back-to-back throughput at the tip. Mac side, one ttp lock p100 per step.
#   drive.sh [rev] [steps]   steps: sync r1 r2 r3 (each round: lat, b2b, drop arms)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t275
P=docs/throughput-t275
O=$P/out; mkdir -p $O
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; exit 75; }; return $rc; }
for s in ${2:-sync r1 r2 r3}; do
  case $s in
    sync) lk opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || exit $rc ;;
    r*) lk $DEVRUN --host $H --no-verify --timeout 590 --tag t275-$s -- "bash $T/$P/remote_time.sh ${s#r}"
        echo "ROUND_${s}_RC=$?"
        scp -q -o BatchMode=yes "$H:$T/tmp/t275/run-$s-*.log" "$H:$T/tmp/t275/md5-$s-*.txt" $O/ ;;
  esac
done
