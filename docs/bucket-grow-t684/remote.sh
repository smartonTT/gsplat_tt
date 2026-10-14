#!/bin/bash
# t684: one step of the grown-bucket A/B on bh-33 (same tree; arms differ only by env).
#   remote.sh <round> <arm> ...   arms (both: benchmarks/cameras_v2.json orbit, b2b, 20 passes):
#     base  default bucket (32768 records per tile)
#     grow  GSPLAT_TT_TEST_PREGROW_TILE_CAP=1 (65472, as after a #284 overflow grow)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_TEST_PREGROW_TILE_CAP GSPLAT_TT_TEST_TILE_CAP
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t684; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  tag=r$r-$arm
  case $arm in
    base) env=() ;; grow) env=(GSPLAT_TT_TEST_PREGROW_TILE_CAP=1) ;; *) echo "bad arm $arm"; exit 2 ;;
  esac
  cache=$C/ttmc-t684-$arm
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} TT_METAL_CACHE_RENDER=$cache \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras benchmarks/cameras_v2.json \
    --iter-dir t684-$tag --back-to-back > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|bucket (pre-grown|tile stride|grown)|tile overflow|Traceback|TT_THROW|TT_FATAL|error:" \
    $S/run-$tag.log | cut -c1-400 | head -16
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $(date +%T)"
exit $fail
