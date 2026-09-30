#!/bin/bash
# 3 interleaved A/B rounds, knob 0 (old) vs 1 (DEST staging), bicycle 30 views, one session.
set -u
source /localdev/smarton/t68_scripts/remote_env.sh
for r in 1 2 3; do for k in 0 1; do
  echo "=== AB round=$r k=$k $(date +%T)"
  GSPLAT_TT_BLEND_COEF_DEST=$k TT_METAL_CACHE_RENDER=$(cache $k) timeout 150 python3 render/run.py --no-ref --iter-dir t68-ab-$r-k$k 2>&1 | grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL"
done; done
echo "=== end $(date +%T)"
