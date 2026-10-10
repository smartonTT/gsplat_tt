#!/bin/bash
# t485: one round on the measurement box (defaults, env as t469's remote_ab.sh).
#   remote.sh <round> <mode> <arm>   mode: b2b | dump; arm: A (GSPLAT_TT_OUT_ZEROCOPY_SLOTS=4) | B (unset)
#   b2b: render/run.py --no-ref --back-to-back with GSPLAT_B2B_ALL_STAGES=1 (B2B_STAGES per pass)
#   dump: render/run.py --no-ref --back-to-back --b2b-passes 2 --dump-views (check-pass frames),
#         md5 list checked with opt/md5_golden.py; hero_clean.png kept
#   (run through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OUT_ZEROCOPY_SLOTS
T=${T:-/localdev/smarton/gstt2-t485}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t485; mkdir -p $S
r=${1:?round}; mode=${2:?mode}; arm=${3:?arm}
case $arm in A) export GSPLAT_TT_OUT_ZEROCOPY_SLOTS=4 ;; B) ;; *) echo "bad arm $arm"; exit 2 ;; esac
tag=r$r-$mode-$arm
args=(--no-ref --iter-dir t485-$tag)
case $mode in b2b) args+=(--back-to-back); export GSPLAT_B2B_ALL_STAGES=1 ;;
  dump) args+=(--back-to-back --b2b-passes 2 --dump-views t485-dump-$tag) ;; *) echo "bad mode $mode"; exit 2 ;; esac
echo "=== $tag $(cut -c1-7 SHA) $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
rm -rf tmp/t485-dump-$tag
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t485 timeout ${RUN_TIMEOUT:-150} \
  python3 render/run.py "${args[@]}" > $S/run-$tag.log 2>&1
rc=$?
echo "run rc=$rc out_ring_new_slots=$(grep -c 'OUT_RING new pinned slot' $S/run-$tag.log) last: $(grep 'OUT_RING new pinned slot' $S/run-$tag.log | tail -1)"
grep -E "^(B2B |\[run\] b2b keep|B2B_STAGES|SORT_STAGES|SUMMARY|\[DEV\])|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -20
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
if [ "$mode" = dump ]; then
  d=$(find . -maxdepth 3 -type d -name t485-dump-$tag | head -1)
  if [ -n "$d" ]; then
    (cd "$d" && md5sum * | sort -k2) > $S/md5-$tag.txt
    python3 opt/md5_golden.py $S/md5-$tag.txt $S/run-$tag.log || { [ $rc = 0 ] && rc=6; }
    rm -rf "$d"
  else echo "NO_DUMP_DIR"; [ $rc = 0 ] && rc=7; fi
fi
cp -f tmp/t485-$tag/hero_clean.png $S/hero-$tag.png 2>/dev/null
echo "=== done $r $mode $(date +%T)"
exit $rc
