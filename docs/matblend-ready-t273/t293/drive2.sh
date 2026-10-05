#!/bin/bash
# t293 run 2: the fused program's CBs overflow L1 by 5888 B under the profiler's auto
# kcfg (+32 KB), so Tracy at KCFG_EXTRA_KB=26 (6 KB back to the CBs); if that fails,
# Tracy with GSPLAT_TT_PFWC_WRITER_SPLIT=0 (kcfg +16). Timing rounds at the 600 s ceiling.
#   drive2.sh   (Mac, repo root; run via ttp detach)
set -u
cd "$(git rev-parse --show-toplevel)"
O=docs/matblend-ready-t273/t289/out
echo "=== t293 drive2 $(git rev-parse --short HEAD) $(date +%T)"
bash docs/matblend-ready-t273/t289/tracy.sh t293-k26 GSPLAT_TT_KCFG_EXTRA_KB=26; echo "TRACY_K26_RC=$?"
if ! grep -q "blend_end" $O/tracy-t293-k26-cores.txt 2>/dev/null; then
  bash docs/matblend-ready-t273/t289/tracy.sh t293-ns GSPLAT_TT_PFWC_WRITER_SPLIT=0; echo "TRACY_NS_RC=$?"
fi
bash docs/matblend-ready-t273/t289/run.sh 600 5 base nofuse:GSPLAT_TT_MATBLEND_FUSE=0; echo "R5_RC=$?"
bash docs/matblend-ready-t273/t289/run.sh 600 6 nofuse:GSPLAT_TT_MATBLEND_FUSE=0 base; echo "R6_RC=$?"
echo "=== t293 drive2 done $(date +%T)"
