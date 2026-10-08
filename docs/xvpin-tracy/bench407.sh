#!/bin/bash
# t407: remote half on bh-30 (viewer stopped by drive_bh30.sh; this script ALWAYS restarts it on
# exit, same steps as #394). tree407 = origin/smarton/tt-project-opt 14f42fda + the
# GSPLAT_TT_PROFILE_READ_EVERY hook; default config (xvpin is the default there, no env flags).
# 1) 3 untraced 30-view rounds (as #394: --dump-views, md5 vs the 906e0435 list, xview_hits 29).
# 2) Tracy in 3 chunks of 10 views (0:10 10:20 20:30), each its own process (1 warmup + 10 views).
#    The device profiler is read ONCE per process, after its 11th render
#    (GSPLAT_TT_PROFILE_READ_EVERY=11), so nothing stalls the device between views and the
#    cross-view overlap stays live. --dump-device-data-mid-run is kept only so that single read is
#    written to profile_log_device.csv (render_clean never closes the device).
set -u
P=/localdev/smarton/p150bench; T=$P/tree407; V=/localdev/smarton/viewer; O=$P/out407; ROUNDS=${1:-3}
restart_viewer() {
  ( cd $V/tree || exit 1
    pid=$(cat $V/viewer.pid 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "viewer already running $pid"; exit 0; }
    export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TT_METAL_RUNTIME_ROOT=$V/tt-metal GSPLAT_SHA=$(cat SHA)
    export TT_METAL_CACHE=$V/tt-metal-cache NUMPY_MADVISE_HUGEPAGE=0
    unset TTW_DEVRUN PYTHONDONTWRITEBYTECODE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OUT_PINNED VIRTUAL_ENV
    unset TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR
    [ -f $V/viewer.log ] && mv -f $V/viewer.log $V/viewer.prev.log
    rm -f $V/viewer.pid $V/viewer.stop
    setsid nohup bash opt/viewer/supervise.sh $V 8080 > $V/viewer.log 2>&1 < /dev/null &
    for _ in $(seq 25); do [ -s $V/viewer.pid ] && break; sleep 0.2; done
    echo "=== viewer restarted pid $(cat $V/viewer.pid 2>/dev/null) $(date -u +%FT%TZ)" )
}
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; restart_viewer; echo $rc > $O/bench.rc' EXIT
rc=99
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal TT_METAL_ARCH_NAME=blackhole
export TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OUT_PINNED
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
hits_of() { grep -m1 "^STAGES " "$1" | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p'; }
rc=0
for r in $(seq 1 $ROUNDS); do
  t=t407-r$r; rm -rf tmp/$t tmp/$t-dump
  echo "=== r$r start $(date -u +%T)"
  TT_METAL_CACHE_RENDER=$P/cache407 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/r$r.log 2>&1
  rr=$?; echo "run r$r rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/r$r.log | cut -c1-400 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-r$r.log 2>&1; rc=124; exit; fi
  [ $rr = 0 ] || { rc=$rr; exit; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-r$r.txt; rm -rf tmp/$t-dump
  [ $r = 1 ] && cp tmp/$t/hero_clean.png $O/hero-r1.png
  n=$(wc -l < $O/md5-r$r.txt)
  diff -q $REF $O/md5-r$r.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$r ($n)" \
    || { echo "VIEWS_DIFFER r$r ($(diff $REF $O/md5-r$r.txt | grep -c '^>') of $n)"; rc=6; }
  h=$(hits_of $O/r$r.log); [ "${h:-x}" = $((n - 1)) ] && echo "XVIEW_HITS_OK r$r $h/$n" || { echo "XVIEW_HITS_BAD r$r ${h:-none}/$n"; rc=7; }
done
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
for a in 0 10 20; do
  b=$((a + 10)); PR=$P/prof407/c$a; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
TT_METAL_CACHE_RENDER=$P/cache407-prof GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t407-T$a --view-range $a:$b
IN
  chmod +x $PR/inner.sh
  echo "=== tracy c$a start $(date -u +%T)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T$a.log 2>&1
  tr=$?; echo "tracy c$a rc=$tr $(date -u +%T)"; [ $tr = 0 ] || { rc=$tr; [ $tr = 124 ] && { tt-smi -r > $O/reset-T$a.log 2>&1; exit; }; }
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL|DRAM buffers were full" $O/T$a.log | cut -c1-400 | head -6
  echo "XVIEW_HITS c$a $(hits_of $O/T$a.log)/10"
  trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
  [ -n "$trf" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-c$a.csv.gz
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev-c$a.csv.gz && echo "device csv c$a rows $(wc -l < "$dl")"
done
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1.log 2>/dev/null
ls -la $O | cut -c25-
