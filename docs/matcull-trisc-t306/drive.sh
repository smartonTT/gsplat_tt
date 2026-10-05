#!/bin/bash
# t306: sync+build, then A/B rounds, all under one ttp lock p100 (Mac, repo root).
#   drive.sh <round:arm,arm> ...   e.g. drive.sh 1:on,base 2:base,on
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t306; O=docs/matcull-trisc-t306/out
cmd="opt/sync_remote.sh $H $T HEAD"
for ra in "$@"; do r=${ra%%:*}; arms=${ra#*:}
  for a in ${arms//,/ }; do  # one devrun per arm: each stays under the 600 s reservation ceiling
    cmd="$cmd && $DEVRUN --host $H --no-verify --timeout 400 --tag t306-r$r-$a -- 'bash $T/docs/matcull-trisc-t306/remote_time.sh $r $a'"
  done
done
ttp lock p100 -- bash -c "$cmd"; rc=$?
echo "DRIVE_RC=$rc"
mkdir -p $O
scp -q -o BatchMode=yes "$H:$T/tmp/t306/run-*.log" "$H:$T/tmp/t306/md5-*.txt" $O/ 2>/dev/null
exit $rc
