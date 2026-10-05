#!/bin/bash
# t298: sync + host build of HEAD into the t298 tree (first build is long).
set -u
cd "$(git rev-parse --show-toplevel)"
ttp lock p100 -- opt/sync_remote.sh yyzo-bh-07 /localdev/smarton/gstt2-t298 HEAD; rc=$?
echo "SYNC_RC=$rc"; exit $rc
