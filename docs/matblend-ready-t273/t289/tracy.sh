#!/bin/bash
# t289: Tracy capture under one ttp lock p100.  tracy.sh <tag> [ENV=V ...]   (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t289
tag=$1
ttp lock p100 -- bash -c "opt/sync_remote.sh $H $T HEAD && $DEVRUN --host $H --no-verify --timeout 600 --tag $tag -- 'bash $T/docs/matblend-ready-t273/t289/remote_tracy.sh $*'"
rc=$?; echo "RUN_RC=$rc"
mkdir -p docs/matblend-ready-t273/t289/out
scp -q -o BatchMode=yes "$H:$T/tmp/t289/tracy-$tag-*" docs/matblend-ready-t273/t289/out/ 2>/dev/null
exit $rc
