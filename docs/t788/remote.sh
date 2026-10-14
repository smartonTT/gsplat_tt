#!/bin/bash
# t788: one step of the train-forward lever A/B (from docs/t778/remote.sh). Runs on the p150
# measurement box.  remote.sh <round> <arm> ...   arm = <cfg><set>[h]
#   cfg: tr  GSPLAT_TT_BLEND_TRAIN=1 (baseline: unfused, SCHED 0, no T early-out, v_if clamps)
#        tp  tr + GSPLAT_TT_TRAIN_T_PERIOD=512 (exact T early-out, eps FLT_MIN)
#        fz  tr + GSPLAT_TT_TRAIN_FUSE=1 (fused mat+blend program)
#        mm  tr + GSPLAT_TT_TRAIN_MINMAX=1 (SFPSWAP clamps)
#        all tr + all three
#   set: o = orbit (benchmarks/cameras_v2.json, 30 views incl. hero), n = first 30 dataset views
#        (cameras_train_nn.json, 1237x822)
#   h:   latency mode with GSPLAT_TT_BLEND_TRAIN_HASH=1 (per-forward train-buffer hash, identity);
#        without h: --back-to-back (timing)
# Each cfg has its own JIT caches so the JIT listings are per arm. A hang marks the cfg dead
# ($S/dead-<cfg>), resets the board and goes on with the other arms.
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
  TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE \
  GSPLAT_TT_TRAIN_T_PERIOD GSPLAT_TT_TRAIN_FUSE GSPLAT_TT_TRAIN_MINMAX GSPLAT_TT_BLEND_TRAIN_HASH
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t788; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
orb=benchmarks/cameras_v2.json nn=benchmarks/cameras_train_nn.json
for arm in "$@"; do
  tag=r$r-$arm
  case $arm in *h) h=1; a=${arm%h} ;; *) h=0; a=$arm ;; esac
  set_=${a: -1}; c=${a%?}
  case $c in
    tr) env=(GSPLAT_TT_BLEND_TRAIN=1) ;;
    tp) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_T_PERIOD=512) ;;
    fz) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_FUSE=1) ;;
    mm) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_MINMAX=1) ;;
    all) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_TRAIN_T_PERIOD=512 GSPLAT_TT_TRAIN_FUSE=1 GSPLAT_TT_TRAIN_MINMAX=1) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  case $set_ in o) cam=$orb; vr=() ;; n) cam=$nn; vr=(--view-range 0:30) ;; *) echo "bad arm $arm"; exit 2 ;; esac
  if [ $h = 1 ]; then env+=(GSPLAT_TT_BLEND_TRAIN_HASH=1); args=(--view-range 0:30); else args=(--back-to-back ${vr[@]+"${vr[@]}"}); fi
  if [ -e $S/dead-$c ]; then echo "=== $tag skipped (cfg $c hung earlier)"; continue; fi
  echo "=== $tag $(cut -c1-8 SHA) ${env[*]} $cam ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${env[@]}" GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t788-$c TT_METAL_CACHE_RENDER=$C/ttmc-t788-$c \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t788-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  [ $h = 1 ] && { grep -a "^TRAIN_HASH" $S/run-$tag.log > $S/hash-$tag.txt; echo "hashes $(wc -l < $S/hash-$tag.txt) md5 $(cut -d' ' -f3 $S/hash-$tag.txt | md5sum | cut -c1-8)"; }
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; touch $S/dead-$c; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; fail=1; continue; fi
  [ $rc = 0 ] || fail=1
  rm -rf "$T/tmp/t788-$tag"  # per-run PNGs: keep remote disk lean
done
echo "=== done $r $(date +%T)"
exit $fail
