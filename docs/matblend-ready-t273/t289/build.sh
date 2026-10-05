#!/bin/bash
# t289: sync + build the fused mat+blend tree on yyzo-bh-07 (one ttp lock p100).
#   build.sh [rev]   (Mac, repo root)
set -u
cd "$(git rev-parse --show-toplevel)"
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t289
ttp lock p100 -- opt/sync_remote.sh $H $T "${1:-HEAD}"; rc=$?
echo "SYNC_RC=$rc"; exit $rc
