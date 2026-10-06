#!/bin/bash
# t315: sync+build, then A/B rounds, all under one ttp lock p100 (Mac, repo root).
#   drive.sh <round:arm,arm> ...   e.g. drive.sh 1:base,off 2:off,base 3:base,off
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t315; O=docs/iter207-t315/out
cmd="opt/sync_remote.sh $H $T HEAD"
for ra in "$@"; do r=${ra%%:*}; arms=${ra#*:}
  for a in ${arms//,/ }; do  # one devrun per arm: each stays under the 600 s reservation ceiling
    cmd="$cmd && $DEVRUN --host $H --no-verify --timeout 400 --tag t315-r$r-$a -- 'bash $T/docs/iter207-t315/remote_time.sh $r $a'"
  done
done
ttp lock p100 -- bash -c "$cmd"; rc=$?
echo "DRIVE_RC=$rc"
mkdir -p $O
scp -q -o BatchMode=yes "$H:$T/tmp/t315/run-*.log" "$H:$T/tmp/t315/md5-*.txt" $O/ 2>/dev/null
exit $rc
