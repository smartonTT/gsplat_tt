#!/bin/bash
# t289: one device step (sync+build, then remote_time.sh rounds) under one ttp lock p100.
#   run.sh <timeout_s> <round> <arm> ...   (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t289
to=$1; r=$2; shift 2
ttp lock p100 -- bash -c "opt/sync_remote.sh $H $T HEAD && $DEVRUN --host $H --no-verify --timeout $to --tag t289-r$r -- 'bash $T/docs/matblend-ready-t273/t289/remote_time.sh $r $*'"
rc=$?; echo "RUN_RC=$rc"
mkdir -p docs/matblend-ready-t273/t289/out
scp -q -o BatchMode=yes "$H:$T/tmp/t289/run-r$r-*.log" "$H:$T/tmp/t289/md5-r$r-*.txt" docs/matblend-ready-t273/t289/out/ 2>/dev/null
exit $rc
