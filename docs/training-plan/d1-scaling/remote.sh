#!/bin/bash
# t691 (D1): one device step on the p150 measurement box (pattern of docs/p150-rebaseline-t687/remote.sh).
#   remote.sh <round> <mode> <cfg>...   mode: b2b (6 passes) | lat (plain latency, per-stage split)
#   cfg: <ntag>_s<scale>, camera file tmp/d1/cams_<cfg>.json from makeproxy.py
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_HTL GSPLAT_TT_HTL_OUT GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_LA_PREFETCH GSPLAT_TT_SORT_IN_FOLD \
  GSPLAT_TT_BLEND_SWAP_FILL GSPLAT_TT_AFR GSPLAT_TT_CONTRIB_FLOOR_INV
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t691; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for cfg in "$@"; do
  tag=r$r-$mode-$cfg; cam=tmp/d1/cams_$cfg.json
  [ -f $cam ] || { echo "missing $cam"; exit 2; }
  case $mode in b2b) args=(--back-to-back --b2b-passes 6) ;; lat) args=() ;; *) echo "bad mode"; exit 2 ;; esac
  echo "=== $tag $(cut -c1-8 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  TT_METAL_CACHE_RENDER=$C/ttmc-t691 timeout ${RUN_TIMEOUT:-240} python3 render/run.py --no-ref --cameras $cam \
    --iter-dir t691-$tag "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|STAGES)|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -14
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
