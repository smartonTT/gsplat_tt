#!/bin/bash
# t555 (B4' step 3b step 2: sort fused into mat+blend): one round of the A/B under the back-to-back headline metric.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | lat | dump   arm: base | nt1 | solo | fuse
#   base = defaults (iter 226), nt1 = GSPLAT_TT_OL_TOWN_NT=1, solo = nt1 + GSPLAT_TT_OL_SOLO=1 (one sort stream, BRISC),
#   fuse = solo + GSPLAT_TT_SORT_IN_BLEND=1 (sort(N+1) inside mat+blend(N)); otherwise defaults (env as probe464.sh).
#   b2b: render/run.py --no-ref --back-to-back (1 warm-up + 20 timed passes, run.py default over the 30 views)
#   lat: render/run.py --no-ref (latency, no --dump-views)
#   dump: render/run.py --no-ref --dump-views, md5 list checked with opt/md5_golden.py
#   (run on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OPACITY_PRUNE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF GSPLAT_TT_BLEND_ALIAS_CUT GSPLAT_TT_KCFG_EXTRA_KB GSPLAT_TT_OL_TOWN_NT GSPLAT_TT_OL_SOLO GSPLAT_TT_SORT_IN_BLEND
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t555; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
for arm in "$@"; do
  case $arm in base) k= ;; nt1) k=GSPLAT_TT_OL_TOWN_NT=1 ;; solo) k="GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1" ;;
    fuse) k="GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1" ;;
    fuse34) k="GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_SORT_IN_BLEND=1 GSPLAT_TT_KCFG_EXTRA_KB=34" ;;  # t565: KCFG probe
    solo48) k="GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_SOLO=1 GSPLAT_TT_KCFG_EXTRA_KB=48" ;;  # t565: does KCFG +48 KB alone hang?
    *) echo "bad arm $arm"; exit 2 ;; esac
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t555-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t555-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-7 SHA) env=[${k}] $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t555-dump-$tag
  env $k TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t513 timeout ${RUN_TIMEOUT:-150} \
    python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |B2B_STAGES|STAGES|SUMMARY|\[SORT\] ONELAUNCH|\[DEV\]|\[run\] opacity prune)|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t555-dump-$tag | head -1)
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
