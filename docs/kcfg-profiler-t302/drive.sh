#!/bin/bash
# t302: Tracy at full defaults (no WRITER_SPLIT=0, no KCFG_EXTRA_KB), then an untraced
# md5/timing round. Each step takes its own ttp lock p100 (inside the t289 scripts).
#   drive.sh   (Mac, repo root; run via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
echo "=== t302 drive $(git rev-parse --short HEAD) $(date +%T)"
bash docs/matblend-ready-t273/t289/tracy.sh t302-def; echo "TRACY_RC=$?"
bash docs/matblend-ready-t273/t289/run.sh 900 t302 a:GSPLAT_TT_NOOP=0 b:GSPLAT_TT_NOOP=1 c:GSPLAT_TT_NOOP=2; echo "ROUND_RC=$?"
echo "=== t302 drive done $(date +%T)"
