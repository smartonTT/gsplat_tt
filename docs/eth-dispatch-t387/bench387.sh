#!/bin/bash
# t387: remote half on bh-30 (viewer stopped by drive_bh30.sh). tree387 = opt tip + t383
# (GSPLAT_TT_DISPATCH). 3 alternating untraced 30-view rounds, worker vs ETH dispatch
# (r1 W,E  r2 E,W  r3 W,A; A = auto, which must resolve to eth on the p150b), each with
# GSPLAT_PER_VIEW_STAGES=1 and md5 of the 30 dumped views vs the 906e0435 golden list.
# Then ONE 30-view Tracy capture on eth (per-core mat+blend busy on the 12x10 grid), then a
# short live-viewer check from tree387 with GSPLAT_TT_DISPATCH=eth on port 8080 (READY,
# HTTP 200, one websocket pose -> frames), stopped by its own pid. Viewer's tt-metal read-only.
set -u
P=/localdev/smarton/p150bench; T=$P/tree387; V=/localdev/smarton/viewer; O=$P/out387; ROUNDS=${1:-3}
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal TT_METAL_ARCH_NAME=blackhole
export TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 GSPLAT_PER_VIEW_STAGES=1
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
rc=0
run() {  # run <tag> <dispatch>
  local t=t387-$1 rr; rm -rf tmp/$t tmp/$t-dump
  echo "=== $1 dispatch=$2 start $(date -u +%T)"
  GSPLAT_TT_DISPATCH=$2 TT_METAL_CACHE_RENDER=$P/cache387 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$1.log 2>&1
  rr=$?; echo "run $1 rc=$rr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_THROW|TT_FATAL" $O/$1.log | cut -c1-300 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$1.log 2>&1; rc=124; return 1; fi
  [ $rr = 0 ] || { rc=$rr; return 1; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$1.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$1.png
  diff -q $REF $O/md5-$1.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL $1 ($(wc -l < $O/md5-$1.txt))" \
    || echo "VIEWS_DIFFER $1 ($(diff $REF $O/md5-$1.txt | grep -c '^>'))"
}
for r in $(seq 1 $ROUNDS); do
  case $r in
    1) run r1-W worker && run r1-E eth ;;
    2) run r2-E eth && run r2-W worker ;;
    *) run r$r-W worker && run r$r-A auto ;;
  esac || break
done
if [ $rc = 0 ]; then
  PR=$P/prof387; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
GSPLAT_TT_DISPATCH=eth TT_METAL_CACHE_RENDER=$P/cache387-prof python3 render/run.py --no-ref --iter-dir t387-T
IN
  chmod +x $PR/inner.sh
  echo "=== tracy (eth) start $(date -u +%T)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T.log 2>&1
  tr=$?; echo "tracy rc=$tr $(date -u +%T)"; [ $tr = 0 ] || rc=$tr
  grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL" $O/T.log | cut -c1-300 | head -5
  trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
  CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
  [ -n "$trf" ] && $CSVX -u "$trf" | gzip > $O/tracy-u.csv.gz
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev.csv.gz && echo "device csv rows $(wc -l < "$dl")"
fi
if [ $rc = 0 ]; then
  echo "=== viewer eth check start $(date -u +%T)"
  GSPLAT_TT_DISPATCH=eth TT_METAL_CACHE=$P/cache387-viewer NUMPY_MADVISE_HUGEPAGE=0 GSPLAT_SHA=$(cat SHA) \
    setsid python3 opt/viewer/viewer_clean.py scenes/bicycle.ply --port 8080 > $O/viewer-eth.log 2>&1 < /dev/null &
  vp=$!
  for _ in $(seq 120); do grep -q READY $O/viewer-eth.log && break; kill -0 $vp 2>/dev/null || break; sleep 2; done
  grep -E "^\[DEV\] dispatch|SELFTEST|READY|Traceback|TT_FATAL" $O/viewer-eth.log | cut -c1-300
  echo "http $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8080/)"
  timeout 30 python3 $P/ws_render_check.py ws://localhost:8080 10 2>&1 | tail -2
  kill -TERM $vp 2>/dev/null; for _ in $(seq 20); do kill -0 $vp 2>/dev/null || break; sleep 1; done
  kill -0 $vp 2>/dev/null && kill -KILL $vp
  echo "=== viewer eth check end $(date -u +%T)"
fi
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1-W.log 2>/dev/null
ls -la $O | cut -c25-
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc
