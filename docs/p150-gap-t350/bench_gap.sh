#!/bin/bash
# t350: remote half of the p150 gap attribution on bh-30 (the viewer is stopped by drive.sh).
# Same tree, venv, tt-metal build and command as #346 (best-iter-207 in $P/tree, the viewer's
# tt-metal 437bc366 used read-only), plus GSPLAT_PER_VIEW_STAGES=1.
#   bench_gap.sh bench <tag> [U|P]...  one 30-view round per arm: U = unpinned,
#                                      P = taskset -c $PIN (cores 18-23, two Zen 2 CCXs)
#                                      C = GSPLAT_TT_MAT_CQ1=0
#   [EMIT=1] bench_gap.sh tracy <tag>  one 30-view Tracy capture with device zones
# While each run is in flight it samples mutagen-agent (the user's devsync; read only, never
# touched) and the box's CPU use every 0.5 s into $O/<tag>-<arm>.mutagen.
set -u
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal   # read-only use of the viewer's build
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export GSPLAT_PER_VIEW_STAGES=1
cd $P/tree || exit 1; source .venv/bin/activate
O=$P/out350; mkdir -p $O
PIN=${PIN:-18-23}
REF=$P/tree/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
MODE=$1; TAG=$2; shift 2

sampler() {  # $1 = out file; mutagen utime+stime (ticks), its threads' CPUs, box /proc/stat line
  local m; m=$(pgrep -u "$USER" -x mutagen-agent | head -1)
  while :; do
    echo "$(date +%s.%N) mut=$( [ -n "$m" ] && awk '{print $14+$15}' /proc/$m/stat) \
psr=$( [ -n "$m" ] && ps -L -o psr= -p $m | tr -s ' \n' ',' ) \
cpu=$(head -1 /proc/stat | cut -d' ' -f3-) load=$(cut -d' ' -f1 /proc/loadavg)"
    sleep 0.5
  done > "$1"
}

run_one() {  # $1 = arm, rest = command prefix
  local arm=$1; shift; local t=t350-$TAG-$arm
  rm -rf tmp/$t tmp/$t-dump
  sampler $O/$TAG-$arm.mutagen & local sp=$!
  echo "=== $TAG $arm start $(date -u +%T) prefix='$*'"
  TT_METAL_CACHE_RENDER=$P/cache timeout ${RUN_TIMEOUT:-330} "$@" \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$TAG-$arm.log 2>&1
  local rr=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  echo "run $TAG $arm rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/$TAG-$arm.log | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/$TAG-$arm.reset.log 2>&1; return 124; fi
  [ $rr = 0 ] || return $rr
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/$TAG-$arm.md5; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/$TAG-$arm.hero.png
  diff -q $REF $O/$TAG-$arm.md5 > /dev/null && echo "ALL_VIEWS_IDENTICAL $TAG $arm" \
    || echo "VIEWS_DIFFER $TAG $arm ($(diff $REF $O/$TAG-$arm.md5 | grep -c '^>'))"
}

rc=0
case $MODE in
  bench)
    for arm in "$@"; do
      case $arm in
        U) run_one U || rc=$? ;;
        P) run_one P taskset -c $PIN || rc=$? ;;
        C) run_one C env GSPLAT_TT_MAT_CQ1=0 || rc=$? ;;   # #351: K2 rows / totals on CQ0
      esac
      [ $rc = 0 ] || break
    done ;;
  tracy)
    # Device + host Tracy zones, as opt/profiler/capture_tracy.sh (mid-run device dump), but all
    # output under $P (TT_METAL_PROFILER_DIR), nothing written into the viewer's tt-metal tree.
    export TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1
    [ "${EMIT:-0}" = 1 ] && export GSPLAT_TT_OL_EMIT_PROF=1   # emit sub-zones for emit_cores.py
    export TT_METAL_PROFILER_DIR=$P/prof350/$TAG PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-}
    rm -rf $P/prof350/$TAG; mkdir -p $P/prof350/$TAG
    cat > $P/prof350/inner.sh <<EOF
#!/bin/bash
cd $P/tree; source .venv/bin/activate
TT_METAL_CACHE_RENDER=$P/cache-prof python3 render/run.py --no-ref --iter-dir t350-$TAG-T
EOF
    chmod +x $P/prof350/inner.sh
    sampler $O/$TAG-T.mutagen & sp=$!
    echo "=== $TAG tracy start $(date -u +%T)"
    timeout ${RUN_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run \
      -o $P/prof350/$TAG $P/prof350/inner.sh > $O/$TAG-T.log 2>&1
    rc=$?
    kill $sp 2>/dev/null; wait $sp 2>/dev/null
    echo "tracy rc=$rc $(date -u +%T)"
    grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL" $O/$TAG-T.log | head -5
    find $P/prof350/$TAG -name '*.tracy' -o -name 'profile_log_device.csv' | xargs -r ls -la ;;
esac
grep -m2 -h -E "firmware bundle version|KMD version" $O/$TAG-*.log 2>/dev/null
echo "=== end $TAG rc=$rc $(date -u +%FT%TZ)"
exit $rc
