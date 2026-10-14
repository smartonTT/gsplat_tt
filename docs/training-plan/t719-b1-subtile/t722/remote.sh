#!/bin/bash
# t722 (B1'): one device step on the p150 box, from docs/training-plan/d1-scaling/remote.sh.
#   B1=<0|1|2|r> PERIOD=<n|default> remote.sh <tag> <cfg>...   (b2b, 6 passes; raw_md5 from the check pass)
# B1=r (#754): B1 full route mode, GSPLAT_TT_B1=0 GSPLAT_TT_B1_ROUTE=1.
# Each arm has its own JIT cache (ttmc-t722-b<B1>-p<PERIOD>) for opt/ab_jit_check.py.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_CONTRIB_FLOOR_INV BLEND_T_PERIOD GSPLAT_TT_B1_ROUTE
T=${T:?T}; B1=${B1:?B1}; PERIOD=${PERIOD:-default}
if [ "$B1" = r ]; then export GSPLAT_TT_B1=0 GSPLAT_TT_B1_ROUTE=1; else export GSPLAT_TT_B1=$B1; fi
C=/localdev/smarton/.cache
# eth dispatch picks its JIT cache from GSPLAT_TT_ETH_CACHE (render/eth_default.py), not
# TT_METAL_CACHE_RENDER: give each arm its own so opt/ab_jit_check.py can compare them.
export GSPLAT_TT_ETH_CACHE=$C/ttmc-t722-b$B1-p$PERIOD
[ "$PERIOD" = default ] || export BLEND_T_PERIOD=${PERIOD}u
S=$T/tmp/t722; mkdir -p $S
tag0=${1:?tag}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for cfg in "$@"; do
  tag=$tag0-b$B1-p$PERIOD${ROUND:+-r$ROUND}-$cfg; cam=tmp/d1/cams_$cfg.json; camarg="--cameras $cam"
  # cfg 'hero': full scene, default cameras (cameras_v2), so hero_clean.png is the reference hero view
  if [ "$cfg" = hero ]; then camarg=""; else [ -f $cam ] || { echo "missing $cam"; exit 2; }; fi
  echo "=== $tag $(cut -c1-8 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  TT_METAL_CACHE_RENDER=$C/ttmc-t722-b$B1-p$PERIOD timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref \
    $camarg --iter-dir t722-$tag --back-to-back --b2b-passes 6 > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|raw_md5|TTW_TIMING (b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:|tile overflow|re-plan" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  [ "$cfg" = hero ] && [ -f tmp/t722-$tag/hero_clean.png ] && cp tmp/t722-$tag/hero_clean.png $S/hero-b$B1.png
done
echo "=== done $tag0 $(date +%T)"
exit $fail
