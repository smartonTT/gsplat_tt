#!/bin/bash
# Task #60 device chain (run detached from repo root): verify md5 -> 3-round A/B -> Tracy x2.
# Usage: docs/blend-dyn-t60/run_all.sh <base_rev> <cand_rev>
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/blend-dyn-t60
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t60_scripts
ssh -o BatchMode=yes yyzo-bh-07 "mkdir -p $S"
scp -q $D/remote_*.sh yyzo-bh-07:$S/
ttp lock p100 -- bash -c "$D/sync_tree.sh t60base $1 && $D/sync_tree.sh t60 $2 && $DEVRUN --no-verify --timeout 590 --tag t60-verify -- 'bash $S/remote_verify_pair.sh t60base t60'" 2>&1 | tee /tmp/t60-verify.$$
echo "VERIFY_RC=${PIPESTATUS[0]}"
grep -q ALL_30_VIEWS_IDENTICAL /tmp/t60-verify.$$ || { echo "STOP: views not identical"; exit 1; }
ttp lock p100 -- $DEVRUN --no-verify --timeout 1000 --tag t60-ab -- "bash $S/remote_ab.sh"
echo "AB_RC=$?"
for v in t60base t60; do
  ttp lock p100 -- $DEVRUN --no-verify --timeout 380 --tag t60-tracy-$v -- "bash $S/remote_tracy.sh $v"
  echo "TRACY_${v}_RC=$?"
done
