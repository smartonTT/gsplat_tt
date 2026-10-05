#!/bin/bash
# Mac side: sync + build, then one devrun chunk per A/B round, each under the p100 lock.
# Usage: run_all.sh <rev>  (run from the repo root, detached)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07
T=/localdev/smarton/gstt2-t111
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}" || exit $?
for r in 1 2; do
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 400 --tag t111-ab-$r -- "bash $T/docs/blend-fpu-qf-t111/remote_ab.sh $r"
  rc=$?; echo "AB_${r}_RC=$rc"
  [ $rc -eq 75 ] && exit 75
done
echo CHAIN_DONE
