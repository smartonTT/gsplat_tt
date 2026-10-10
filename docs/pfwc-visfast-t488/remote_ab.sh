#!/bin/bash
# t488: one round of the PFWC_VIS_FAST A/B under the back-to-back headline metric.
#   remote_ab.sh <round> <mode> <arm> ...   mode: b2b | lat | dump   arm: off | on
#   off = GSPLAT_TT_PFWC_VIS_FAST=0, on = =1 (both arms GSPLAT_TT_PFWC_SKIP_RGB=1), def = unset (the default); otherwise
#   defaults (env as probe464.sh).
#   b2b: render/run.py --no-ref --back-to-back (1 warm-up + 20 timed passes, run.py default over the 30 views)
#   lat: render/run.py --no-ref (latency, no --dump-views)
#   dump: render/run.py --no-ref --dump-views, md5 list checked with opt/md5_golden.py
#   (run on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_SKIP_RGB GSPLAT_TT_PFWC_VIS_FAST
T=${T:-/localdev/smarton/gstt2-t469}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t488; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; shift 2
fail=0
for arm in "$@"; do
  case $arm in off) k=0 ;; on) k=1 ;; def) k= ;; *) echo "bad arm $arm"; exit 2 ;; esac
  tag=r$r-$mode-$arm
  args=(--no-ref --iter-dir t488-$tag)
  case $mode in b2b) args+=(--back-to-back) ;; dump) args+=(--dump-views t488-dump-$tag) ;; esac
  echo "=== $tag $(cut -c1-7 SHA) VIS_FAST=${k:-unset} SKIP_RGB=1 $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  rm -rf tmp/t488-dump-$tag
  env GSPLAT_TT_PFWC_SKIP_RGB=1 ${k:+GSPLAT_TT_PFWC_VIS_FAST=$k} TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t488 timeout ${RUN_TIMEOUT:-150} \
    python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |B2B_STAGES|STAGES|SUMMARY|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
  if [ "$mode" = dump ]; then
    d=$(find . -maxdepth 3 -type d -name t488-dump-$tag | head -1)
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
