#!/bin/bash
# t667: one step of the training-view bench on bh-33 (same tree and code; arms are camera files).
#   remote.sh <round> <arm> ...   arms:
#     orbit   benchmarks/cameras_v2.json (headline control, 1024x1024)   b2b, 20 passes
#     orb1237 benchmarks/cameras_orbit_1237.json                          b2b, 20 passes
#     shuf    benchmarks/cameras_train_shuffled.json (169 views)          b2b, 6 passes
#     nn      benchmarks/cameras_train_nn.json (169 views)                b2b, 6 passes
#     test    benchmarks/cameras_test.json (25 views)                     b2b, 20 passes
#     pv-<arm>   latency mode with GSPLAT_PER_VIEW_STAGES=1 (per-view stage spread, no dump)
#     dump    latency --dump-views of the first 3 shuffled-train views (random, seed 0)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t667; mkdir -p $S
r=${1:?round}; shift
fail=0
cd "$T" || exit 1; source .venv/bin/activate
cams() { case $1 in
  orbit) echo benchmarks/cameras_v2.json ;; orb1237) echo benchmarks/cameras_orbit_1237.json ;;
  shuf) echo benchmarks/cameras_train_shuffled.json ;; nn) echo benchmarks/cameras_train_nn.json ;;
  test) echo benchmarks/cameras_test.json ;; *) return 1 ;; esac; }
for arm in "$@"; do
  tag=r$r-$arm
  env=(); base=${arm#pv-}
  case $arm in
    pv-*) env=(GSPLAT_PER_VIEW_STAGES=1); args=(); cam=$(cams $base) || { echo "bad arm $arm"; exit 2; } ;;
    dump) base=shuf; cam=$(cams shuf); args=(--dump-views t667-dump --view-range 0:3) ;;
    *) cam=$(cams $arm) || { echo "bad arm $arm"; exit 2; }; args=(--back-to-back)
       case $arm in shuf|nn) args+=(--b2b-passes 6) ;; esac ;;
  esac
  cache=$C/ttmc-t667-$base
  echo "=== $tag $(cut -c1-8 SHA) cams=$cam ${env[*]+${env[*]}} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env ${env[@]+"${env[@]}"} TT_METAL_CACHE_RENDER=$cache \
    timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam --iter-dir t667-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  [ $arm = dump ] && { mkdir -p $S/dump; cp tmp/t667-dump/*.png $S/dump/ 2>/dev/null; ls $S/dump; }
done
echo "=== done $r $(date +%T)"
exit $fail
