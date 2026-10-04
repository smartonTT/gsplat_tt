#!/bin/bash
# t171 gate: sync + build the tip, then untraced host-profile runs (GSPLAT_TT_HOST_PROFILE=1 +
# GSPLAT_PER_VIEW_STAGES=1) and a plain base run, to size the removable host bridge before
# materialize and the d2h tail.   drive.sh [rev] [steps=sync,time]  (Mac; one ttp lock p100 per step)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t171
O=docs/t171/out; mkdir -p $O
STEPS=${2:-sync,time}
if [[ $STEPS == *sync* ]]; then
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
  [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
fi
if [[ $STEPS == *time* ]]; then
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 560 --tag t171-time1 -- \
    "T154_TREE=$T bash $T/docs/t171/remote_time.sh 1 hp base"
  echo "TIME1_RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t171/run-r*.log" "$H:$T/tmp/t171/md5-r*.txt" "$H:$T/tmp/t171/hp-r*.txt" $O/
fi
echo CHAIN_DONE
