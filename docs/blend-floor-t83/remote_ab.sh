#!/bin/bash
# One interleaved round <r> over the ablation list (default "0 4"), bicycle 30 views, no profiler.
set -u
source /localdev/smarton/t83_scripts/remote_env.sh
r=$1; shift; list=${*:-0 4}
for a in $list; do
  echo "=== AB round=$r a=$a $(date +%T)"
  GSPLAT_TT_BLEND_ABL=$a TT_METAL_CACHE_RENDER=$(cache $a) timeout 150 python3 render/run.py --no-ref --iter-dir t83-ab-$r-a$a 2>&1 | grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL"
done
echo "=== end $(date +%T)"
