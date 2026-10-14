#!/bin/bash
# t778: one step of the train-flag split (copy of docs/t755/remote.sh). Runs on the p150
# measurement box.  remote.sh <round> <arm> ...   arms (no code change, env knobs only):
#   rnd    orbit (benchmarks/cameras_v2.json, 30 views) b2b, render default (fused, SCHED 2, T_PERIOD 512)
#   unf    rnd + GSPLAT_TT_MATBLEND_FUSE=0 (unfused program, as train mode)
#   s0     unf + GSPLAT_TT_BLEND_SCHED=0 (compiled bodies, as train mode)
#   tp0    s0 + BLEND_T_PERIOD=0u (no T early-out readback, as train mode)
#   tr     GSPLAT_TT_BLEND_TRAIN=1 (adds accurate exp + exact alpha/T rules, fp32 op/colour, fp32 RGBT write)
#   rndnn / trnn   rnd / tr on the first 30 dataset views (cameras_train_nn.json, 1237x822) b2b
#   pvnn   tr on the same 30 dataset views, latency mode, GSPLAT_PER_VIEW_STAGES=1 (per-view stage ms)
# Each kernel config has its own JIT caches (GSPLAT_TT_ETH_CACHE, TT_METAL_CACHE_RENDER) so the JIT
# listings are per arm.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_BLEND_TRAIN GSPLAT_TT_BLEND_TRAIN_DUMP GSPLAT_TT_MATBLEND_FUSE \
  GSPLAT_TT_BLEND_DECODE_AHEAD GSPLAT_TT_BLEND_SCHED BLEND_T_PERIOD BLEND_T_EPS \
  GSPLAT_TT_TRAIN_OPC_BATCH GSPLAT_TT_TRAIN_OPC_SKIPDEAD GSPLAT_TT_PFWC_REC32 GSPLAT_TT_TRAIN_OPC_SIDE GSPLAT_TT_TRAIN_OPC_WRITER GSPLAT_TT_T727_DIAG GSPLAT_TT_TRAIN_OPC_RING3 GSPLAT_TT_BLEND_PROF \
  TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t778; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
orb=benchmarks/cameras_v2.json nn=benchmarks/cameras_train_nn.json
for arm in "$@"; do
  tag=r$r-$arm
  case $arm in
    rnd) env=(); cam=$orb; args=(--back-to-back); c=rnd ;;
    unf) env=(GSPLAT_TT_MATBLEND_FUSE=0); cam=$orb; args=(--back-to-back); c=unf ;;
    s0) env=(GSPLAT_TT_MATBLEND_FUSE=0 GSPLAT_TT_BLEND_SCHED=0); cam=$orb; args=(--back-to-back); c=s0 ;;
    tp0) env=(GSPLAT_TT_MATBLEND_FUSE=0 GSPLAT_TT_BLEND_SCHED=0 BLEND_T_PERIOD=0u); cam=$orb; args=(--back-to-back); c=tp0 ;;
    tr) env=(GSPLAT_TT_BLEND_TRAIN=1); cam=$orb; args=(--back-to-back); c=tr ;;
    rndnn) env=(); cam=$nn; args=(--back-to-back --view-range 0:30); c=rnd ;;
    trnn) env=(GSPLAT_TT_BLEND_TRAIN=1); cam=$nn; args=(--back-to-back --view-range 0:30); c=tr ;;
    pvnn) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_PER_VIEW_STAGES=1); cam=$nn; args=(--view-range 0:30); c=tr ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} $cam ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t778-$c TT_METAL_CACHE_RENDER=$C/ttmc-t778-$c \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t778-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  rm -rf "$T/tmp/t778-$tag"  # per-run PNGs: keep remote disk lean
done
echo "=== done $r $(date +%T)"
exit $fail
