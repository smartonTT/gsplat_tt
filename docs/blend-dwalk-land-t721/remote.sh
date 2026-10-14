#!/bin/bash
# t721: one step of the BLEND_DWALK landing A/B on the p150 measurement box (bh-49 p150a).
# Two trees: $T2 = best-iter-229 code (= best-iter-230, the p150 re-baseline), $T = the candidate.
# One eth JIT cache per arm (GSPLAT_TT_ETH_CACHE; the knob is a compile define).
#   remote.sh <round> <arm> ...   arms (orbit = benchmarks/cameras_v2.json, 30 views, b2b 20 passes):
#     b230   $T2 defaults (best-iter-230)                aa    b230 again, same cache (A/A)
#     new    $T defaults (BLEND_DWALK 1)                 base  $T with GSPLAT_TT_BLEND_DWALK=0
#     nb230 / nnew   train NN cameras (169 views, 1237x822), b2b 6 passes
#     lb230 / lnew   orbit latency (no --back-to-back, no --dump-views; secondary)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE TT_METAL_CACHE_RENDER \
  GSPLAT_PER_VIEW_STAGES GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL \
  GSPLAT_TT_MATCULL_PROF GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_BLEND_DWALK GSPLAT_TT_AFR GSPLAT_TT_SPLIT
T=${T:?T} T2=${T2:?T2}
C=/localdev/smarton/.cache; S=$T/tmp/t721; mkdir -p $S
r=${1:?round}; shift
fail=0
for arm in "$@"; do
  tag=r$r-$arm
  env=(); args=(--back-to-back); cam=benchmarks/cameras_v2.json
  case $arm in
    b230|aa|nb230|lb230) tree=$T2; cache=$C/ttmc-t721-b230 ;;
    new|nnew|lnew) tree=$T; cache=$C/ttmc-t721-new ;;
    base) tree=$T; cache=$C/ttmc-t721-base; env=(GSPLAT_TT_BLEND_DWALK=0) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  case $arm in
    n*) cam=benchmarks/cameras_train_nn.json; args+=(--b2b-passes 6) ;;
    l*) args=() ;;
  esac
  echo "=== $tag $(cut -c1-8 $tree/SHA) cams=$cam ${env[*]+${env[*]}} ${args[*]+${args[*]}} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  ( cd "$tree" && source .venv/bin/activate && env ${env[@]+"${env[@]}"} GSPLAT_TT_ETH_CACHE=$cache \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t721-$tag \
    ${args[@]+"${args[@]}"} ) > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|\[eth\]|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  # an inert lever must not be measured: the arm's JIT cache must hold the define it asks for
  fills=$(grep -rhoE "define BLEND_DWALK [01]" "$cache" 2>/dev/null | sort -u | tr '\n' ' ')
  echo "LEVER $tag cache_defines=[${fills}]"
  case $arm in
    base) case "$fills" in *"DWALK 1"*|"") echo "BASE_NOT_OFF $tag"; fail=1 ;; esac ;;
    new|nnew|lnew) case "$fills" in *"DWALK 0"*|"") echo "INERT $tag"; fail=1 ;; esac ;;
    *) [ -z "$fills" ] || { echo "B230_HAS_DWALK $tag"; fail=1; } ;;
  esac
done
echo "=== done $r $(date +%T)"
exit $fail
