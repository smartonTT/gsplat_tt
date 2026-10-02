#!/bin/bash
# t115 Mac-side driver: sync+build the tip, 2 untraced timing rounds, 1 traced 30-view capture.
# Each device step is its own `ttp lock p100` + devrun job. Run from the worktree root (detached).
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t115
if [ "${SKIP_SYNC:-0}" = 1 ]; then  # remote tree already at the wanted SHA and built
  rc=0; echo "SYNC skipped (remote SHA $(ssh -o BatchMode=yes $H cat $T/SHA))"
else
  ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?; echo "SYNC_RC=$rc"
fi
[ $rc -eq 0 ] || exit $rc
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 590 --tag t115-time-1 -- "bash $T/docs/reprofile-t115/remote_time.sh 1 base hp dc base"
echo "TIME_1_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t115-tracy -- "bash $T/docs/reprofile-t115/remote_tracy.sh t115-tip"
echo "TRACY_RC=$?"
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 500 --tag t115-time-2 -- "bash $T/docs/reprofile-t115/remote_time.sh 2 base base"
echo "TIME_2_RC=$?"
echo CHAIN_DONE
