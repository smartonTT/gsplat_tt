#!/bin/bash
# t293: Tracy (fused), md5/timing rounds after the default flip, device screenshot.
# Each step takes its own ttp lock p100 (inside the t289 scripts / screenshot.sh).
#   drive.sh <iter>   (Mac, repo root; run via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
it=${1:?iter}; D=docs/matblend-ready-t273/t293
echo "=== t293 drive $(git rev-parse --short HEAD) $(date +%T)"
bash docs/matblend-ready-t273/t289/tracy.sh t293-fuse GSPLAT_TT_MATBLEND_FUSE=1; echo "TRACY_RC=$?"
bash docs/matblend-ready-t273/t289/run.sh 1200 5 base nofuse:GSPLAT_TT_MATBLEND_FUSE=0 newton0:GSPLAT_TT_PFWC_RECIP_NEWTON=0; echo "R5_RC=$?"
bash docs/matblend-ready-t273/t289/run.sh 900 6 nofuse:GSPLAT_TT_MATBLEND_FUSE=0 base; echo "R6_RC=$?"
NAME=ttw-$it T=/localdev/smarton/gstt2-t289 opt/ttw/screenshot.sh $it HEAD; echo "SHOT_RC=$?"
echo "=== t293 drive done $(date +%T)"
