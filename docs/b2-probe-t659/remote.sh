#!/bin/bash
# t659 (from docs/swapfill-land-t655/remote.sh): one step of the B2 ellipse-probe A/B on bh-33
# (same tree, arms by env; one JIT cache per arm since the knob is a compile define).
#   remote.sh <round> <arm> ...   arms:
#     off       best-iter-229 defaults (no env)
#     probe     GSPLAT_TT_ELLIPSE_PROBE=1 (#538 per-pair ellipse test in K2, pairs unchanged)
#     aa        defaults again with its own JIT cache (same-day A/A noise floor)
#     lat-off / lat-probe   latency mode (no --back-to-back, no --dump-views), secondary only
# After each arm its JIT cache listing goes to $S/<arm>.jit.txt for opt/ab_jit_check.py.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_ELLIPSE_PROBE GSPLAT_TT_K2_PROF
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t659; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  tag=r$r-$arm
  env=(); args=(--back-to-back)
  case $arm in
    off|lat-off) cache=$C/ttmc-t659off ;;
    probe|lat-probe) env=(GSPLAT_TT_ELLIPSE_PROBE=1); cache=$C/ttmc-t659probe ;;
    aa) cache=$C/ttmc-t659aa ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  case $arm in lat-*) args=() ;; esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} ${args[*]+${args[*]}} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} TT_METAL_CACHE_RENDER=$cache \
    timeout ${RUN_TIMEOUT:-200} python3 render/run.py --no-ref --iter-dir t659-$tag ${args[@]+"${args[@]}"} > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-300 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  # an inert lever must not be measured: the probe arm's JIT cache must hold the define, the others not
  n=$(grep -rlE "define K2_ELLIPSE_PROBE" "$cache" 2>/dev/null | wc -l)
  echo "LEVER $tag probe_define_files=$n"
  case $arm in
    probe|lat-probe) [ "$n" -gt 0 ] || { echo "INERT $tag"; fail=1; } ;;
    *) [ "$n" = 0 ] || { echo "BASE_NOT_OFF $tag"; fail=1; } ;;
  esac
  find "$cache" -maxdepth 6 -path '*/kernels/*/*' -type d > $S/${arm#lat-}.jit.txt
done
echo "=== done $r $(date +%T)"
exit $fail
