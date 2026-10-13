#!/bin/bash
# t644 (from docs/exp1-t584/remote_ab.sh): one step of the pfwc unit-gather probe on the measurement box.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | lat | dump   arm: base | c1k | u64 | u64id
#   base  today's order (shipping path)
#   c1k   GSPLAT_TT_CHUNK_CULL=1: Morton + chunk-1024 cull (t470 lever, reference point)
#   u64   + GSPLAT_TT_CHUNK_UNIT=64: 64-gaussian cull units, 16 per pfwc chunk (the probe)
#   u64id u64 cull on the host, chunks = the 1024-tiles holding a surviving unit, lanes as c1k,
#         each plane still read as 16 x 256 B (gather cost at c1k's data: md5 must match c1k)
#   all arms GSPLAT_TT_B4A_FOLD=1 (b2b config), each its own JIT cache
#   b2b:  render/run.py --no-ref --back-to-back (headline metric, 30 bicycle views)
#   lat:  latency mode, no dump, GSPLAT_TT_XVIEW_OVERLAP=0 (pfwc serialised: project stage = pfwc)
#   dump: --dump-views md5 list (base must match opt/md5_golden.py)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_STACK_CHECK GSPLAT_TT_CHUNK_CULL GSPLAT_TT_CHUNK_UNIT GSPLAT_TT_CHUNK_SKIP GSPLAT_TT_CHUNK_REORDER GSPLAT_TT_CHUNK_UNIT_ID
export GSPLAT_TT_B4A_FOLD=1 GSPLAT_B2B_ALL_STAGES=1
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t644; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
cd "$T" || exit 1; source .venv/bin/activate
for arm in "$@"; do
  case $arm in
    base) e=() ;;
    c1k) e=(GSPLAT_TT_CHUNK_CULL=1) ;;
    u64) e=(GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_UNIT=64) ;;
    u64id) e=(GSPLAT_TT_CHUNK_CULL=1 GSPLAT_TT_CHUNK_UNIT=64 GSPLAT_TT_CHUNK_UNIT_ID=1) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  cache=$C/ttmc-t644$arm
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t644-$tag)
  x=(GSPLAT_TT_CHUNK_LOG=0)
  case $mode in
    b2b) args+=(--back-to-back) ;;
    lat) x+=(GSPLAT_TT_XVIEW_OVERLAP=0) ;;
    dump) args+=(--dump-views t644-dump-$tag); x=(GSPLAT_TT_CHUNK_LOG=1) ;;
  esac
  echo "=== $tag $(cut -c1-8 SHA) ${e[*]} ${x[*]} cache=$cache $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t644-dump-$tag
  env "${e[@]}" "${x[@]}" TT_METAL_CACHE_RENDER=$cache timeout ${RUN_TIMEOUT:-150} \
    python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |B2B_STAGES|SUMMARY|STAGES|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame|stage_[a-z_]+)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-700 | head -40
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t644-dump-$tag | head -1)
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
