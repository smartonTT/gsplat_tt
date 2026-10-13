#!/bin/bash
# t649: one step of the L-A + L-A' A/B on bh-33 (same tree, arms by env, one JIT cache).
#   remote.sh <round> <arm> ...   arms:
#     base      GSPLAT_TT_LA_PREFETCH=0 GSPLAT_TT_SORT_IN_FOLD=0 (best-iter-227 behaviour)
#     la        L-A only (LA_PREFETCH=1 SORT_IN_FOLD=0)
#     sif       L-A' only (LA_PREFETCH=0 SORT_IN_FOLD=1)
#     both      the new defaults (no env): L-A, L-A' where L-A did not run
#     lat-base / lat-both   latency mode (no --back-to-back, no --dump-views), secondary only
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t649; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  tag=r$r-$arm
  env=(); args=(--back-to-back)
  case $arm in
    base|lat-base) env=(GSPLAT_TT_LA_PREFETCH=0 GSPLAT_TT_SORT_IN_FOLD=0) ;;
    la) env=(GSPLAT_TT_LA_PREFETCH=1 GSPLAT_TT_SORT_IN_FOLD=0) ;;
    sif) env=(GSPLAT_TT_LA_PREFETCH=0 GSPLAT_TT_SORT_IN_FOLD=1) ;;
    both|lat-both) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  case $arm in lat-*) args=() ;; esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} ${args[*]+${args[*]}} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} TT_METAL_CACHE_RENDER=$C/ttmc-t649 \
    timeout ${RUN_TIMEOUT:-200} python3 render/run.py --no-ref --iter-dir t649-$tag ${args[@]+"${args[@]}"} > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|\[LA\] )|sort_in_fold|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-300 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  # an inert lever must not be measured: count the hook lines per arm
  nla=$(grep -c '\[LA\] prefetch' $S/run-$tag.log); nsif=$(grep -c 'sort_in_fold: next view.s sort enqueued=1' $S/run-$tag.log)
  echo "LEVER $tag la_lines=$nla sif_lines=$nsif overflow=$(grep -c '\[LA\] pair overflow' $S/run-$tag.log)"
  case $arm in
    la|both|lat-both) [ "$nla" -gt 0 ] || { echo "INERT $tag: no [LA] prefetch lines"; fail=1; } ;;
    sif) [ "$nsif" -gt 0 ] || { echo "INERT $tag: no sort_in_fold lines"; fail=1; } ;;
    base|lat-base) [ "$nla" = 0 ] && [ "$nsif" = 0 ] || { echo "BASE_NOT_OFF $tag"; fail=1; } ;;
  esac
done
echo "=== done $r $(date +%T)"
exit $fail
