#!/bin/bash
# t634 (from docs/exp1-t584/remote_ab.sh): one step of the abs-means emit gate on the measurement box.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | dump   arm: base | abl | ctl
#   one tree $T; GSPLAT_TT_OL_ABS_MEANS_ABL = 0 (base), 1 (abl: emit skips both sub_int32),
#   2 (ctl: subtracts run, same absolute means written as abl, so downstream work matches abl).
#   Each arm has its own JIT cache. b2b: render/run.py --no-ref --back-to-back (headline, 30 views).
#   dump (base only): --dump-views md5 list vs opt/md5_golden.py (base must match).
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_BLEND_EXP1 GSPLAT_TT_OL_ABS_MEANS_ABL
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t634; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  case $arm in base) e=0 ;; abl) e=1 ;; ctl) e=2 ;; *) echo "bad arm $arm"; exit 2 ;; esac
  cache=$C/ttmc-t634$arm
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t634-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t634-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-8 SHA) ABS_MEANS_ABL=$e cache=$cache $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t634-dump-$tag
  GSPLAT_TT_OL_ABS_MEANS_ABL=$e TT_METAL_CACHE_RENDER=$cache timeout ${RUN_TIMEOUT:-150} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t634-dump-$tag | head -1)
    if [ -n "$d" ]; then
      (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
      python3 $T/opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log; g=$?
      echo "MD5 golden rc=$g ($arm)"
      [ "$arm" = base ] && [ $g != 0 ] && [ $rc = 0 ] && rc=6
      rm -rf "$d"
    else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
  fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
