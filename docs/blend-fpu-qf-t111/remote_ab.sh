#!/bin/bash
# Task #111 A/B on the remote tree (default /localdev/smarton/gstt2-t111), bicycle 30 views.
# Arms: d = default; p0 = BLEND_T_PERIOD=0; q0 = period 0 + GSPLAT_TT_BLEND_FPU_QF_ABL=1.
# q0 - p0 = SFPU time of the conic (upper bound on what an FPU quadratic form can save).
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=${T111_TREE:-/localdev/smarton/gstt2-t111}
cd $T || exit 1
source .venv/bin/activate
cache() { echo /localdev/smarton/.cache/ttmc-$(basename $T)-$1; }
for r in ${1:-1}; do
  for arm in d p0 q0; do
    case $arm in
      d)  envs="" ;;
      p0) envs="BLEND_T_PERIOD=0u" ;;
      q0) envs="BLEND_T_PERIOD=0u GSPLAT_TT_BLEND_FPU_QF_ABL=1" ;;
    esac
    echo "=== AB round=$r arm=$arm $(date +%T)"
    env $envs TT_METAL_CACHE_RENDER=$(cache $arm) timeout 300 python3 render/run.py --no-ref --iter-dir t111-ab-$r-$arm 2>&1 | grep -E "^(SUMMARY|STAGES)|md5|Traceback|TT_THROW|TT_FATAL"
  done
done
echo "=== end $(date +%T)"
