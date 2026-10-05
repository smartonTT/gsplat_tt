#!/bin/bash
# Follow-up device work for task #59, run detached (ttp detach) from the repo root:
#  1. rebased verify: base 7b9358d vs candidate 4a7df4e, build + 30-view dump + md5
#  2. one 10-view Tracy chunk each for the fdef315 base / candidate trees
# Each device session holds `ttp lock p100` for one devrun call (< 400 s).
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/cull-band-t59
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t59_scripts
scp -q $D/remote_build_dump.sh $D/remote_verify_pair.sh $D/remote_tracy.sh $D/remote_ab.sh yyzo-bh-07:$S/
ttp lock p100 -- bash -c "$D/sync_tree.sh t59b2 7b9358d && $D/sync_tree.sh t59c2 4a7df4e && $DEVRUN --no-verify --timeout 390 --tag t59-verify2 -- 'bash $S/remote_verify_pair.sh t59b2 t59c2'"
echo "VERIFY_RC=$?"
for v in t59base t59; do
  ttp lock p100 -- $DEVRUN --no-verify --timeout 380 --tag t59-tracy-$v -- "bash $S/remote_tracy.sh $v"
  echo "TRACY_${v}_RC=$?"
done
