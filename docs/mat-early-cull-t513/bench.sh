#!/bin/bash
# t519 (remote, measurement box, through devrun.sh inside drive.sh's ttp lock p100).
#   bench.sh b2b <round> <arm> ...      render/run.py --no-ref --back-to-back (1 warm-up + 20 passes, 30 views)
#   bench.sh tracy <arm> <a> <b>        GSPLAT_TT_MATCULL_PROF=1 Tracy b2b capture of views a:b
#                                       (--b2b-passes 1 --b2b-warmup 0, profiler read once after the
#                                       11th render: GSPLAT_TT_PROFILE_READ_EVERY=11)
#   arm: A = GSPLAT_TT_MAT_EARLY_CULL unset, B = GSPLAT_TT_MAT_EARLY_CULL=1; otherwise defaults.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF TT_METAL_DEVICE_PROFILER \
  GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t519; mkdir -p $S
CACHE=/localdev/smarton/.cache/ttmc-gstt2-t513
armenv() { case $1 in A) echo "" ;; B) echo "GSPLAT_TT_MAT_EARLY_CULL=1" ;; *) echo BAD ;; esac; }
mode=${1:?mode}; shift
hang() { echo "HANG in $1: tt-smi -r"; tt-smi -r > $S/reset-$1.log 2>&1; echo "reset rc=$?"; exit 124; }
case $mode in
b2b)
  r=${1:?round}; shift; fail=0
  for arm in "$@"; do
    e=$(armenv $arm); [ "$e" = BAD ] && { echo "bad arm $arm"; exit 2; }
    tag=r$r-$arm
    echo "=== $tag $(cut -c1-8 SHA) [${e:-unset}] $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    env $e TT_METAL_CACHE_RENDER=$CACHE timeout ${RUN_TIMEOUT:-300} \
      python3 render/run.py --no-ref --back-to-back --iter-dir t519-$tag > $S/run-$tag.log 2>&1
    rc=$?; echo "run rc=$rc load=$(cut -d' ' -f1-3 /proc/loadavg)"
    grep -E "^(B2B |B2B_STAGES|SUMMARY|\[DEV\] (sort|mat))|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -12
    { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
    rm -rf tmp/t519-$tag; [ $rc = 0 ] || fail=1
  done
  exit $fail ;;
tracy)
  arm=${1:?arm}; a=${2:?a}; b=${3:?b}; e=$(armenv $arm); tag=M$arm-$a
  PR=$S/prof-$tag; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
env $e TT_METAL_CACHE_RENDER=$CACHE-prof GSPLAT_TT_MATCULL_PROF=1 GSPLAT_TT_PROFILE_READ_EVERY=11 \
  python3 render/run.py --no-ref --iter-dir t519-$tag --view-range $a:$b --back-to-back --b2b-passes 1 --b2b-warmup 0
IN
  chmod +x $PR/inner.sh
  echo "=== tracy $tag [${e:-unset}] $(cut -c1-8 SHA) $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $S/T-$tag.log 2>&1
  rc=$?; echo "tracy $tag rc=$rc $(date +%T)"
  { [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
  grep -E "^(B2B |SUMMARY|STAGES)|Traceback|TT_FATAL|DRAM buffers were full" $S/T-$tag.log | cut -c1-300 | head -8
  dl=$(find $PR -name profile_log_device.csv | head -1)
  [ -n "$dl" ] || { echo "no device csv $tag"; exit 2; }
  gzip -c "$dl" > $S/dev-$tag.csv.gz
  echo "device csv $tag rows $(wc -l < "$dl") mj_zones $(grep -c ',mj_' "$dl") sel_zones $(grep -c 'mat_ol_sel' "$dl")"
  rm -rf $PR tmp/t519-$tag
  exit $rc ;;
*) echo "bad mode $mode"; exit 2 ;;
esac
