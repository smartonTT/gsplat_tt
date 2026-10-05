#!/bin/bash
# One interleaved round <r> over ablations 0 1 2, bicycle 30 views.
set -u
source /localdev/smarton/t78_scripts/remote_env.sh
r=$1
for a in 0 1 2; do
  echo "=== AB round=$r a=$a $(date +%T)"
  GSPLAT_TT_BLEND_ABL=$a TT_METAL_CACHE_RENDER=$(cache $a) timeout 150 python3 render/run.py --no-ref --iter-dir t78-ab-$r-a$a 2>&1 | grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL"
done
echo "=== end $(date +%T)"
