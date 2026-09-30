#!/bin/bash
# Task #60 verify: sync base + candidate, build both, one 30-view dump each, md5 compare.
# Usage (repo root): docs/blend-dyn-t60/run_verify.sh <base_rev> <cand_rev>
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/blend-dyn-t60
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t60_scripts
ssh -o BatchMode=yes yyzo-bh-07 "mkdir -p $S"
scp -q $D/remote_*.sh yyzo-bh-07:$S/
ttp lock p100 -- bash -c "$D/sync_tree.sh t60base $1 && $D/sync_tree.sh t60 $2 && $DEVRUN --no-verify --timeout 590 --tag t60-verify -- 'bash $S/remote_verify_pair.sh t60base t60'"
echo "VERIFY_RC=$?"
