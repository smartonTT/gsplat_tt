#!/bin/bash
# t298: one arm-run (sync+build, then remote_time.sh) under one ttp lock p100, <5 min.
#   run.sh <round> <arm> [ENV=V ...]   (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t298; O=docs/k2-folded-t298/out
r=$1; a=$2; shift 2
timeout 290 ttp lock p100 -- bash -c "opt/sync_remote.sh $H $T HEAD > /dev/null && $DEVRUN --host $H --no-verify --timeout 250 --tag t298-r$r-$a -- 'bash $T/docs/k2-folded-t298/remote_time.sh $r $a $*'"
rc=$?; echo "RUN_RC=$rc"
mkdir -p $O
scp -q -o BatchMode=yes "$H:$T/tmp/t298/run-r$r-$a.log" "$H:$T/tmp/t298/md5-r$r-$a.txt" $O/ 2>/dev/null
exit $rc
