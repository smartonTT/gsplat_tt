#!/bin/bash
# t479: Tracy capture (as t418/#413) of one arm of the chunk cull A/B:
#   remote_tracy.sh <chunk start> <arm> "ENV=V ENV=V"   e.g. 0 cull "GSPLAT_TT_CHUNK_CULL=1"
# Runs on the measurement box through devrun.sh, inside drive479.sh's ttp lock p100.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1
T=${T:-/localdev/smarton/gstt2-t479}; cd "$T" || exit 1; source .venv/bin/activate
a=${1:?chunk start}; ARM=${2:-cull}; XENV=${3:-}; b=$((a + 10)); O=$T/tmp/t479; PR=$O/prof-$ARM-c$a; rm -rf $PR; mkdir -p $PR
cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t479-prof $XENV \
  GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t479-$ARM-T$a --view-range $a:$b
IN
chmod +x $PR/inner.sh
echo "=== tracy $ARM c$a $(cut -c1-8 SHA) $(date +%T)"
TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
  timeout ${TRACY_TIMEOUT:-420} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T-$ARM-$a.log 2>&1
rc=$?; echo "tracy $ARM c$a rc=$rc $(date +%T)"
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$ARM-T$a.log 2>&1; exit $rc; fi
grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL|DRAM buffers were full|MATBLEND_PROGRAM" $O/T-$ARM-$a.log | cut -c1-300 | head -8
echo "XVIEW_HITS c$a $(grep -m1 '^STAGES ' $O/T-$ARM-$a.log | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p')/10"
CSVX=$(ls $TT_METAL_HOME/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
[ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-$ARM-c$a.csv.gz
[ -n "$dl" ] || { echo "no device csv c$a"; exit 2; }
gzip -c "$dl" > $O/dev-$ARM-c$a.csv.gz
echo "device csv c$a rows $(wc -l < "$dl") mj_zones $(grep -c ',mj_' "$dl")"
exit $rc
