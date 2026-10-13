#!/bin/bash
# t606 (copy of docs/pfwc-rec-loop-t599/remote_ab.sh, arm env only): one step of the env A/B.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | dump   arm: base | new | vis
#   one tree $T, a JIT cache per arm (ttmc-t606<arm>):
#   base = defaults (RECFAST=0 VIS_FAST=0, best-iter-227), new = RECFAST=1 VIS_FAST=1,
#   vis = VIS_FAST=1 only (attribution)
#   b2b:  render/run.py --no-ref --back-to-back (headline metric, 30 bicycle views)
#   dump: render/run.py --no-ref --dump-views, md5 list vs opt/md5_golden.py; an off-golden
#         dump (and every base dump) is kept as $S/dump-<tag> for a per-view diff
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_B4A_FOLD GSPLAT_TT_STACK_CHECK GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t606; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  case $arm in base) e=0; v=0 ;; new) e=1; v=1 ;; vis) e=0; v=1 ;; *) echo "bad arm $arm"; exit 2 ;; esac
  cache=$C/ttmc-t606$arm
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t606-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t606-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-8 SHA) RECFAST=$e VIS_FAST=$v cache=$cache $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t606-dump-$tag
  GSPLAT_TT_PFWC_RECFAST=$e GSPLAT_TT_PFWC_VIS_FAST=$v TT_METAL_CACHE_RENDER=$cache timeout ${RUN_TIMEOUT:-150} python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t606-dump-$tag | head -1)
    if [ -n "$d" ]; then
      (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
      python3 $T/opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log; g=$?
      echo "MD5 golden rc=$g ($arm)"
      [ $g != 0 ] && [ $rc = 0 ] && rc=6
      if [ $g != 0 ] || [ $arm = base ]; then rm -rf $S/dump-$tag; mv "$d" $S/dump-$tag; else rm -rf "$d"; fi
    else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
  fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
