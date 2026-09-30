#!/bin/bash
# Task #61 device driver, run detached (ttp detach) from the worktree root:
#  1. sync base (opt tip) + candidate trees, build, 30-view dump, md5 compare
#  2. if identical: 3 interleaved A/B rounds (base, cand) in one devrun session
# Each device session holds `ttp lock p100` for one devrun call (< 600 s).
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/img-pack-t61
BASE=${BASE:-27348a1}
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t61_scripts
scp -q $D/remote_build_dump.sh $D/remote_verify_pair.sh $D/remote_ab.sh yyzo-bh-07:$S/ || exit 9
ttp lock p100 -- bash -c "$D/sync_tree.sh t61base $BASE && $D/sync_tree.sh t61 HEAD && $DEVRUN --no-verify --timeout 420 --tag t61-verify -- 'bash $S/remote_verify_pair.sh t61base t61'"
RC=$?; echo "VERIFY_RC=$RC"
[ $RC = 0 ] || exit $RC
ttp lock p100 -- $DEVRUN --no-verify --timeout 580 --tag t61-ab -- "bash $S/remote_ab.sh t61base t61 3"
echo "AB_RC=$?"
