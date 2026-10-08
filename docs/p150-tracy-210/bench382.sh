#!/bin/bash
# t382: remote half on bh-30 (viewer stopped by drive_bh30.sh). best-iter-210 tree (tree382):
# 3 untraced 30-view rounds (GSPLAT_PER_VIEW_STAGES=1, md5 of the 30 dumped views vs the
# 906e0435 golden list), then ONE 30-view Tracy capture (device + host zones, GSPLAT_TT_PROFILE=1,
# --dump-device-data-mid-run) and csvexport -u. Viewer's tt-metal used read-only.
set -u
P=/localdev/smarton/p150bench; T=$P/tree382; V=/localdev/smarton/viewer; O=$P/out382; ROUNDS=${1:-3}
export TT_METAL_HOME=$V/tt-metal TT_METAL_RUNTIME_ROOT=$V/tt-metal TT_METAL_ARCH_NAME=blackhole
export TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 GSPLAT_PER_VIEW_STAGES=1
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
REF=$T/docs/matblend-ready-t273/t289/md5-golden-906e0435.txt
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
rc=0
for r in $(seq 1 $ROUNDS); do
  t=t382-r$r; rm -rf tmp/$t tmp/$t-dump
  echo "=== r$r start $(date -u +%T)"
  TT_METAL_CACHE_RENDER=$P/cache382 timeout ${RUN_TIMEOUT:-330} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/r$r.log 2>&1
  rr=$?; echo "run r$r rc=$rr $(date -u +%T)"
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|tile stride|Traceback|TT_THROW|TT_FATAL" $O/r$r.log | cut -c1-300 | head -6
  if [ $rr = 124 ] || [ $rr = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-r$r.log 2>&1; rc=124; break; fi
  [ $rr = 0 ] || { rc=$rr; break; }
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-r$r.txt; rm -rf tmp/$t-dump
  [ $r = 1 ] && cp tmp/$t/hero_clean.png $O/hero-r1.png
  diff -q $REF $O/md5-r$r.txt > /dev/null && echo "ALL_VIEWS_IDENTICAL r$r ($(wc -l < $O/md5-r$r.txt))" \
    || echo "VIEWS_DIFFER r$r ($(diff $REF $O/md5-r$r.txt | grep -c '^>'))"
done
if [ $rc = 0 ]; then
  PR=$P/prof382; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
TT_METAL_CACHE_RENDER=$P/cache382-prof python3 render/run.py --no-ref --iter-dir t382-T
IN
  chmod +x $PR/inner.sh
  echo "=== tracy start $(date -u +%T)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T.log 2>&1
  tr=$?; echo "tracy rc=$tr $(date -u +%T)"; [ $tr = 0 ] || rc=$tr
  grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL" $O/T.log | cut -c1-300 | head -5
  trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
  CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
  [ -n "$trf" ] && $CSVX -u "$trf" | gzip > $O/tracy-u.csv.gz && cp "$trf" $O/t382.tracy
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev.csv.gz && echo "device csv rows $(wc -l < "$dl")"
fi
grep -m2 -h -E "firmware bundle version|KMD version" $O/r1.log 2>/dev/null
ls -la $O | cut -c25-
echo "=== bench end rc=$rc $(date -u +%FT%TZ)"
echo $rc > $O/bench.rc
