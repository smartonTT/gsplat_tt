#!/bin/bash
# t633 (from docs/exp1-t584/remote_ab.sh): one step of the B2 ellipse-probe A/B on the measurement box.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | dump   arm: off | probe
#   one tree $T; off = GSPLAT_TT_ELLIPSE_PROBE unset (best-iter-227 defaults, JIT cache ttmc-t633off),
#   probe = GSPLAT_TT_ELLIPSE_PROBE=1 (#538 per-pair test in K2 and, since #633, in the folded K2; ttmc-t633probe)
#   b2b:  render/run.py --no-ref --back-to-back (headline metric, 30 bicycle views)
#   dump: render/run.py --no-ref --dump-views, md5 list vs opt/md5_golden.py (both arms must match:
#         the probe leaves every pair unchanged)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_ELLIPSE_PROBE
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t633; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  case $arm in off) e= ;; probe) e=1 ;; *) echo "bad arm $arm"; exit 2 ;; esac
  cache=$C/ttmc-t633$arm
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t633-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t633-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-8 SHA) ELLIPSE_PROBE=${e:-unset} cache=$cache $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t633-dump-$tag
  env ${e:+GSPLAT_TT_ELLIPSE_PROBE=$e} TT_METAL_CACHE_RENDER=$cache timeout ${RUN_TIMEOUT:-150} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t633-dump-$tag | head -1)
    if [ -n "$d" ]; then
      (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
      python3 $T/opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log; g=$?
      echo "MD5 golden rc=$g ($arm)"
      [ $g != 0 ] && [ $rc = 0 ] && rc=6
      rm -rf "$d"
    else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
  fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
