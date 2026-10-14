#!/bin/bash
# t755: one step of the 3-slot op/colour ring A/B (copy of docs/t741/remote.sh).
# Runs on the p150 measurement box.  remote.sh <round> <arm> ...   arms:
#   on      orbit (benchmarks/cameras_v2.json, 30 views) b2b, GSPLAT_TT_BLEND_TRAIN=1 (default)
#   aa      same as on with its own JIT cache (A/A spread)
#   r3      on + GSPLAT_TT_TRAIN_OPC_RING3=1
#   dump*   on / r3, latency, view 0 only, fp32 train buffer (GSPLAT_TT_BLEND_TRAIN_DUMP),
#           then docs/t708/check_dump.py against the run's PNG
# Each arm has its own eth JIT cache (GSPLAT_TT_ETH_CACHE) so the JIT listings are per arm.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_BLEND_TRAIN GSPLAT_TT_BLEND_TRAIN_DUMP GSPLAT_TT_MATBLEND_FUSE \
  GSPLAT_TT_BLEND_DECODE_AHEAD \
  GSPLAT_TT_TRAIN_OPC_BATCH GSPLAT_TT_TRAIN_OPC_SKIPDEAD GSPLAT_TT_PFWC_REC32 GSPLAT_TT_TRAIN_OPC_SIDE GSPLAT_TT_TRAIN_OPC_WRITER GSPLAT_TT_T727_DIAG GSPLAT_TT_TRAIN_OPC_RING3 GSPLAT_TT_BLEND_PROF \
  TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t755; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
cam=benchmarks/cameras_v2.json
for arm in "$@"; do
  tag=r$r-$arm
  case $arm in
    on) env=(GSPLAT_TT_BLEND_TRAIN=1); args=(--back-to-back); c=on ;;
    aa) env=(GSPLAT_TT_BLEND_TRAIN=1); args=(--back-to-back); c=aa ;;
    r3) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_OPC_RING3=1); args=(--back-to-back); c=r3 ;;
    dump) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_BLEND_TRAIN_DUMP=$S/train-v0-on.bin); args=(--dump-views t755-dump --view-range 0:1); c=on ;;
    dumpr3) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_OPC_RING3=1 GSPLAT_TT_BLEND_TRAIN_DUMP=$S/train-v0-r3.bin); args=(--dump-views t755-dumpr3 --view-range 0:1); c=r3 ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t755-$c TT_METAL_CACHE_RENDER=$C/ttmc-t755-$c \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t755-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|\[SORT\] ONELAUNCH|\[eth\]|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  case $arm in dump*)
    ls -l $S/train-v0-*.bin
    b=$S/train-v0-${arm#dump}.bin; [ "$arm" = dump ] && b=$S/train-v0-on.bin
    python3 docs/t708/check_dump.py "$b" "$T/tmp/t755-$arm/view00_hero.png" > $S/check_dump-$arm.txt 2>&1
    tail -1 $S/check_dump-$arm.txt ;;
  esac
done
echo "=== done $r $(date +%T)"
exit $fail
