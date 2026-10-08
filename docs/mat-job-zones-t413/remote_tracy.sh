#!/bin/bash
# t413: Tracy capture of the #407 xvpin run with per-job mat zones (GSPLAT_TT_MATCULL_PROF=1:
# mj_wait / mj_job on the TRISCs, mat_ol_* / mat_cull_wait per subchunk on the movers), in
# 3 chunks of 10 views (+1 warmup each), one process per chunk. The device profiler is read
# once per process after its 11th render (GSPLAT_TT_PROFILE_READ_EVERY=11), so the cross-view
# overlap stays live; --dump-device-data-mid-run makes that read reach profile_log_device.csv.
#   remote_tracy.sh <chunk start>   (on the measurement box through devrun.sh, inside drive.sh's
#   ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2-t413}; cd "$T" || exit 1; source .venv/bin/activate
a=${1:?chunk start}; b=$((a + 10)); O=$T/tmp/t413; PR=$O/prof-c$a; rm -rf $PR; mkdir -p $PR
cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t413-prof GSPLAT_TT_MATCULL_PROF=1 \
  GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t413-T$a --view-range $a:$b
IN
chmod +x $PR/inner.sh
echo "=== tracy c$a $(cut -c1-8 SHA) $(date +%T)"
TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
  timeout ${TRACY_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T$a.log 2>&1
rc=$?; echo "tracy c$a rc=$rc $(date +%T)"
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-T$a.log 2>&1; exit $rc; fi
grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL|DRAM buffers were full|MATBLEND_PROGRAM" $O/T$a.log | cut -c1-300 | head -8
echo "XVIEW_HITS c$a $(grep -m1 '^STAGES ' $O/T$a.log | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p')/10"
CSVX=$(ls $TT_METAL_HOME/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
[ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-c$a.csv.gz
[ -n "$dl" ] || { echo "no device csv c$a"; exit 2; }
gzip -c "$dl" > $O/dev-c$a.csv.gz
echo "device csv c$a rows $(wc -l < "$dl") mj_zones $(grep -c ',mj_' "$dl")"
exit $rc
