#!/bin/bash
# t708: one step of the forward-training-flag A/B (from docs/p150-rebaseline-t687/remote.sh).
# Runs on the p150 measurement box.  remote.sh <round> <arm> ...   arms:
#   off   orbit (benchmarks/cameras_v2.json, 30 views) b2b 20 passes, GSPLAT_TT_BLEND_TRAIN unset
#   on    same with GSPLAT_TT_BLEND_TRAIN=1
#   dump  on, latency, view 0 only, --dump-views + GSPLAT_TT_BLEND_TRAIN_DUMP (fp32 train buffer)
# Each arm has its own eth JIT cache (GSPLAT_TT_ETH_CACHE) so the JIT listings are per arm.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_BLEND_TRAIN GSPLAT_TT_BLEND_TRAIN_DUMP GSPLAT_TT_MATBLEND_FUSE
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t708; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
cam=benchmarks/cameras_v2.json
for arm in "$@"; do
  tag=r$r-$arm
  case $arm in
    off) env=(); args=(--back-to-back); c=off ;;
    on) env=(GSPLAT_TT_BLEND_TRAIN=1); args=(--back-to-back); c=on ;;
    dump) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_BLEND_TRAIN_DUMP=$S/train-v0.bin); args=(--dump-views t708-dump --view-range 0:1); c=on ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t708-$c TT_METAL_CACHE_RENDER=$C/ttmc-t708-$c \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t708-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|\[eth\]|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  [ $arm = dump ] && { cp tmp/t708-dump/*.png $S/ 2>/dev/null; ls -l $S/train-v0.bin $S/*.png; }
done
echo "=== done $r $(date +%T)"
exit $fail
