#!/bin/bash
# t528 (remote, measurement box bh-33, through devrun.sh inside drive.sh's ttp lock p100):
# per-stage b2b profile of best-iter-224 defaults.
#   bench.sh b2b <round>            render/run.py --no-ref --back-to-back (1 check + 1 warm-up + 20 passes,
#                                   30 views), GSPLAT_B2B_ALL_STAGES=1 (prints every host stage timer; no
#                                   effect on the device work)
#   bench.sh counts                 latency loop (no dumps) with GSPLAT_VIEW_COUNTS=1: N, M (visible), P (pairs)
#   bench.sh chunk <tag> <a> <b>    untraced b2b 1-pass of views a:b (same shape as a Tracy chunk; raw_md5 ref)
#   bench.sh tracy <tag> <a> <b> [ENV=V ...]   Tracy b2b capture of views a:b (--b2b-passes 1 --b2b-warmup 0,
#                                   profiler read once after the 11th render: GSPLAT_TT_PROFILE_READ_EVERY=11)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF TT_METAL_DEVICE_PROFILER \
  GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR GSPLAT_TT_K2_PROF GSPLAT_TT_OL_EMIT_PROF \
  GSPLAT_TT_BLEND_PROF GSPLAT_VIEW_COUNTS GSPLAT_B2B_ALL_STAGES
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t528; mkdir -p $S
CACHE=/localdev/smarton/.cache/ttmc-gstt2-t513
hang() { echo "HANG in $1: tt-smi -r"; tt-smi -r > $S/reset-$1.log 2>&1; echo "reset rc=$?"; exit 124; }
SYS='/sys/class/tenstorrent/tenstorrent!0'
hdr() { echo "=== $1 $(cut -c1-8 SHA) $(date +%T) card=$(cat "$SYS/tt_card_type" 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"; }
pat="^(B2B |B2B_STAGES pass=1[0-9]|SUMMARY|VIEW_COUNTS i=0 |\[DEV\] dispatch)|Traceback|TT_THROW|TT_FATAL|error:"
mode=${1:?mode}; shift
case $mode in
b2b)
  tag=b2b-r${1:?round}; hdr $tag
  GSPLAT_B2B_ALL_STAGES=1 TT_METAL_CACHE_RENDER=$CACHE timeout ${RUN_TIMEOUT:-300} \
    python3 render/run.py --no-ref --back-to-back --iter-dir t528-$tag > $S/run-$tag.log 2>&1
  rc=$?; echo "run rc=$rc"; grep -E "$pat" $S/run-$tag.log | cut -c1-300 | head -6
  { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
  rm -rf tmp/t528-$tag; exit $rc ;;
counts)
  tag=counts; hdr $tag
  GSPLAT_VIEW_COUNTS=1 TT_METAL_CACHE_RENDER=$CACHE timeout ${RUN_TIMEOUT:-300} \
    python3 render/run.py --no-ref --iter-dir t528-$tag > $S/run-$tag.log 2>&1
  rc=$?; echo "run rc=$rc"; grep -E "$pat" $S/run-$tag.log | cut -c1-300 | head -6
  echo "VIEW_COUNTS lines $(grep -c '^VIEW_COUNTS' $S/run-$tag.log)"
  { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
  rm -rf tmp/t528-$tag; exit $rc ;;
chunk)
  tag=${1:?tag}; a=${2:?a}; b=${3:?b}; hdr "chunk $tag $a:$b"
  TT_METAL_CACHE_RENDER=$CACHE timeout ${RUN_TIMEOUT:-300} python3 render/run.py --no-ref --iter-dir t528-$tag \
    --view-range $a:$b --back-to-back --b2b-passes 1 --b2b-warmup 0 > $S/run-$tag.log 2>&1
  rc=$?; echo "run rc=$rc"; grep -E "$pat" $S/run-$tag.log | cut -c1-300 | head -6
  { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
  rm -rf tmp/t528-$tag; exit $rc ;;
tracy)
  tag=${1:?tag}; a=${2:?a}; b=${3:?b}; shift 3; e="$*"
  PR=$S/prof-$tag; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
env $e TT_METAL_CACHE_RENDER=$CACHE-prof GSPLAT_TT_PROFILE_READ_EVERY=11 \
  python3 render/run.py --no-ref --iter-dir t528-$tag --view-range $a:$b --back-to-back --b2b-passes 1 --b2b-warmup 0
IN
  chmod +x $PR/inner.sh
  hdr "tracy $tag $a:$b [${e:-plain}]"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $S/T-$tag.log 2>&1
  rc=$?; echo "tracy $tag rc=$rc $(date +%T)"
  { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
  grep -E "^(B2B |SUMMARY)|Traceback|TT_FATAL|DRAM buffers were full" $S/T-$tag.log | cut -c1-300 | head -8
  dl=$(find $PR -name profile_log_device.csv | head -1)
  [ -n "$dl" ] || { echo "no device csv $tag"; exit 2; }
  head -1 "$dl" | cut -c1-200
  gzip -c "$dl" > $S/dev-$tag.csv.gz
  echo "device csv $tag rows $(wc -l < "$dl")"
  rm -rf $PR tmp/t528-$tag
  exit $rc ;;
*) echo "bad mode $mode"; exit 2 ;;
esac
