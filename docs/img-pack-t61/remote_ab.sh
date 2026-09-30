#!/bin/bash
# N interleaved A/B rounds (default 3), bicycle 30 views, one device session.
# Usage: remote_ab.sh <base_variant> <cand_variant> [rounds]
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
source /localdev/smarton/gstt2/.venv/bin/activate
for r in $(seq 1 ${3:-3}); do for v in $1 $2; do
  cd /localdev/smarton/gstt2-$v; echo "=== AB round=$r variant=$v $(date +%T)"
  TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-$v timeout 90 python3 render/run.py --no-ref --iter-dir t61-ab-$r 2>&1 | grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL"
done; done
echo "=== end $(date +%T)"
