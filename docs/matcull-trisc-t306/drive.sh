#!/bin/bash
# t306: sync+build, then A/B rounds, all under one ttp lock p100 (Mac, repo root).
#   drive.sh <round:arm,arm> ...   e.g. drive.sh 1:on,base 2:base,on
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t306; O=docs/matcull-trisc-t306/out
cmd="opt/sync_remote.sh $H $T HEAD"
for ra in "$@"; do r=${ra%%:*}; arms=${ra#*:}
  cmd="$cmd && $DEVRUN --host $H --no-verify --timeout 1500 --tag t306-r$r -- 'bash $T/docs/matcull-trisc-t306/remote_time.sh $r ${arms//,/ }'"
done
ttp lock p100 -- bash -c "$cmd"; rc=$?
echo "DRIVE_RC=$rc"
mkdir -p $O
scp -q -o BatchMode=yes "$H:$T/tmp/t306/run-*.log" "$H:$T/tmp/t306/md5-*.txt" $O/ 2>/dev/null
exit $rc
