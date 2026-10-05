#!/bin/bash
# t302: paired untraced round, base (51ac7cf, t293) vs HEAD, alternating builds:
# base, head, base, head; 2 arms each. Each step takes its own ttp lock p100.
#   drive2.sh   (Mac, repo root; run via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t289
OUT=docs/matblend-ready-t273/t289/out; mkdir -p $OUT
HEAD_SHA=$(git rev-parse --short HEAD)
echo "=== t302 drive2 head=$HEAD_SHA base=51ac7cf $(date +%T)"
step() {  # round rev arm...
  local r=$1 rev=$2; shift 2
  ttp lock p100 -- bash -c "opt/sync_remote.sh $H $T $rev && $DEVRUN --host $H --no-verify --timeout 400 --tag t302-$r -- 'bash $T/docs/matblend-ready-t273/t289/remote_time.sh $r $*'"
  echo "STEP $r $rev RC=$?"
  scp -q -o BatchMode=yes "$H:$T/tmp/t289/run-r$r-*.log" "$H:$T/tmp/t289/md5-r$r-*.txt" $OUT/ 2>/dev/null
}
step t302b1 51ac7cf b1:GSPLAT_TT_NOOP=0 b2:GSPLAT_TT_NOOP=1
step t302h1 HEAD    h1:GSPLAT_TT_NOOP=0 h2:GSPLAT_TT_NOOP=1
step t302b2 51ac7cf b3:GSPLAT_TT_NOOP=0 b4:GSPLAT_TT_NOOP=1
step t302h2 HEAD    h3:GSPLAT_TT_NOOP=0 h4:GSPLAT_TT_NOOP=1
echo "=== t302 drive2 done $(date +%T)"
