#!/bin/bash
# t554 (B4a fold: K2(N+1) on the fused mat+blend(N) BRISC, default on, GSPLAT_TT_B4A_FOLD=0 off): one round of the A/B under the back-to-back headline metric.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | lat | dump | vb2b | vlat   arm: off | on
#   vb2b / vlat: b2b / latency with GSPLAT_TT_B4A_VERIFY=3 (the first 3 folded views: keep the
#   folded K2 outputs, rerun the real K2 and compare; prints the [B4A] fold verify lines)
#   on = flag unset (default since #554), off = GSPLAT_TT_B4A_FOLD=0; an on arm must log
#   "[B4A] fold: view K2 from the previous blend" (the fold ran) or it fails
#   otherwise defaults (env as probe464.sh).
#   b2b: render/run.py --no-ref --back-to-back (1 warm-up + 20 timed passes, run.py default over the 30 views)
#   lat: render/run.py --no-ref (latency, no --dump-views)
#   dump: render/run.py --no-ref --dump-views, md5 list checked with opt/md5_golden.py
#   (run on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OPACITY_PRUNE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF GSPLAT_TT_B4A_SHADOW GSPLAT_TT_B4A_VERIFY GSPLAT_TT_B4A_FOLD GSPLAT_TT_K2_EXTRA
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t554; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
for arm in "$@"; do
  case $arm in off) k=0 ;; on) k= ;; *) echo "bad arm $arm"; exit 2 ;; esac
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t554-$tag)
  v=
  case $mode in vb2b|vlat) v=3 ;; esac
  case $mode in b2b|vb2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t554-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-7 SHA) B4A_FOLD=${k:-unset(default on)} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t554-dump-$tag
  env ${k:+GSPLAT_TT_B4A_FOLD=$k} ${v:+GSPLAT_TT_B4A_VERIFY=$v} TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t513 timeout ${RUN_TIMEOUT:-150} \
    python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |B2B_STAGES|STAGES|SUMMARY|\[DEV\]|\[run\] opacity prune)|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  grep -E "^\[B4A\]" $S/run-$tag.log | head -40
  if [ -n "$v" ]; then
    grep -q "^\[B4A\] fold verify view .* OK" $S/run-$tag.log && ! grep -q "MISMATCH" $S/run-$tag.log \
      || { echo "B4A_VERIFY_FAIL $tag"; [ $rc = 0 ] && rc=8; }
  fi
  if [ -z "$k" ] && ! grep -q "^\[B4A\] fold: view K2 from the previous blend" $S/run-$tag.log; then
    echo "FOLD_NOT_RUN $tag"; [ $rc = 0 ] && rc=9
  fi
  if [ "$k" = 0 ] && grep -q "^\[B4A\] fold" $S/run-$tag.log; then
    echo "FOLD_NOT_OFF $tag (kill switch GSPLAT_TT_B4A_FOLD=0 ignored)"; [ $rc = 0 ] && rc=10
  fi
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t554-dump-$tag | head -1)
    if [ -n "$d" ]; then
      (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
      python3 opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log || { [ $rc = 0 ] && rc=6; }
      rm -rf "$d"
    else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
  fi
  [ $rc = 0 ] || fail=1
done
echo "=== done $r $mode $(date +%T)"
exit $fail
