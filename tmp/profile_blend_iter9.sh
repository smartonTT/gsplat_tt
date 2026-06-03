#!/usr/bin/env bash
# Device-cycle / wall-time breakdown for fused cull+blend (iter 9 profiling).
set -euo pipefail
cd /localdev/smarton/gstt2
source .venv/bin/activate 2>/dev/null || true
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
export TT_METAL_ARCH_NAME=blackhole
export MESH_DEVICE=P100

BASE="GSPLAT_TT_BLEND_MODE=2 GSPLAT_TT_MB_KERNEL=1 GSPLAT_TT_MB_DEVCULL=1 GSPLAT_TT_DEVICE_PROJECT=1 GSPLAT_TT_RESIDENT_PROJECT=1 GSPLAT_TT_RESIDENT_GATHER=1 GSPLAT_TT_DEVICE_TILE_ASSIGN=1 GSPLAT_TT_RESIDENT_TA_IN=1 GSPLAT_TT_DEVICE_SORT=1 GSPLAT_TT_RESIDENT_PAIRS=1 GSPLAT_TT_RESIDENT_BLEND=1 GSPLAT_TT_SORT_DEVICE_PUBLISH=1 GSPLAT_TT_TA_DEVICE_SCAN=1 GSPLAT_TT_SFPU_CULL=1 GSPLAT_TT_MB_TIMING=1"

run_case() {
  local tag="$1"
  local extra="$2"
  echo ""
  echo "========== CASE: $tag =========="
  eval "export $BASE $extra"
  .venv/bin/python3 scripts/a003_verify.py --views 1 --iter-dir loop-ttw --out "opt/ttw/profile-${tag}.json" 2>&1
  grep -E '^(SUMMARY|\[FUSED_TILE\]|\[BLEND_SPLIT\]|\[CULL_SPLIT\]|\[BLEND_DEVICE\])' "opt/ttw/profile-${tag}.json" 2>/dev/null || true
}

echo "=== HEAD ==="
git rev-parse --short HEAD
echo "=== BUILD CHECK ==="
cmake --build build-tt -j 16 --target gsplat_tt_cpu_cpp 2>&1

run_case "fused_default" "GSPLAT_TT_FUSED_TILE=1"
run_case "split_cull_blend" "GSPLAT_TT_FUSED_TILE=0"
