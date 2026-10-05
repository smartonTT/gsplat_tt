#!/bin/bash
# Task #60 second chain (remote trees already synced + built by run_all.sh verify):
# 3-round A/B (one devrun, <400 s) -> 10-view Tracy for both trees.
# Usage (repo root): docs/blend-dyn-t60/run_ab_tracy.sh
set -u
cd "$(git rev-parse --show-toplevel)"
D=docs/blend-dyn-t60
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t60_scripts
scp -q $D/remote_*.sh yyzo-bh-07:$S/
# profiler scripts are not in the synced subset; copy them into both trees
for v in t60base t60; do
  tar -c opt/profiler/*.sh opt/profiler/*.py | ssh -o BatchMode=yes yyzo-bh-07 "tar -x -m -C /localdev/smarton/gstt2-$v"
done
ttp lock p100 -- $DEVRUN --no-verify --timeout 390 --tag t60-ab -- "bash $S/remote_ab.sh"
echo "AB_RC=$?"
for v in t60base t60; do
  ttp lock p100 -- $DEVRUN --no-verify --timeout 380 --tag t60-tracy-$v -- "bash $S/remote_tracy.sh $v"
  echo "TRACY_${v}_RC=$?"
done
echo CHAIN_DONE
