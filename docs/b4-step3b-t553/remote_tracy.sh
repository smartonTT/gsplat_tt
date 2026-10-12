#!/bin/bash
# t553: Tracy capture of one 10-view chunk (+1 warmup) with mover (BRISC/NCRISC) counters on.
#   remote_tracy.sh <arm> <chunk start>   arm: def | nt1
#   def: default config + GSPLAT_TT_MV_PROF=1 (fused blend movers "mv_pc")
#   nt1: + GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1 (town "town_pc" on TRISC 2 only)
# The device profiler is read once per process after its 11th render (GSPLAT_TT_PROFILE_READ_EVERY=11).
# (on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_OL_TOWN_NT GSPLAT_TT_OL_EMIT_PROF GSPLAT_TT_MATCULL_PROF GSPLAT_TT_BLEND_PROF
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
arm=${1:?arm}; a=${2:?chunk start}; b=$((a + 10)); O=$T/tmp/t553; PR=$O/prof-$arm-c$a; rm -rf $PR; mkdir -p $PR
case $arm in def) X= ;; nt1) X="GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1" ;; *) echo "bad arm"; exit 2 ;; esac
cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
env TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t553-mvprof GSPLAT_TT_MV_PROF=1 $X \
  GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t553-$arm-T$a --view-range $a:$b
IN
chmod +x $PR/inner.sh
echo "=== tracy $arm c$a $(cut -c1-8 SHA) $X $(date +%T)"
# bh-33's venv (2026-10-12) lacks the tracy wrapper's python deps (loguru, pandas, click, ...). Only
# profile_log_device.csv is needed, which tt-metal writes itself on ReadMeshDeviceProfilerResults, so
# without them run inner.sh directly under the device profiler (no host Tracy stream, no ops report).
if python3 -c "import loguru, pandas, click" 2>/dev/null; then
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-500} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/$arm-T$a.log 2>&1
else
  echo "tracy wrapper deps missing: device profiler only"
  # the wrapper's --dump-device-data-mid-run is what makes ReadMeshDeviceProfilerResults write the csv
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_NO_CACHE_OP_INFO=1 TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 timeout ${TRACY_TIMEOUT:-500} $PR/inner.sh > $O/$arm-T$a.log 2>&1
fi
rc=$?; echo "tracy $arm c$a rc=$rc $(date +%T)"
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$arm-T$a.log 2>&1; exit $rc; fi
grep -E "^(SUMMARY|STAGES|SORT_STAGES)|Traceback|TT_FATAL|TT_THROW|error:|DRAM buffers were full" $O/$arm-T$a.log | cut -c1-300 | head -8
dl=$(find $PR $TT_METAL_HOME/generated/profiler -newer $PR/inner.sh -name profile_log_device.csv 2>/dev/null | head -1)
[ -n "$dl" ] || { echo "no device csv $arm c$a"; exit 2; }
gzip -c "$dl" > $O/dev-$arm-c$a.csv.gz
echo "device csv $arm c$a rows $(wc -l < "$dl") mv_pc $(grep -c ',mv_pc,' "$dl") town_pc $(grep -c ',town_pc,' "$dl")"
exit $rc
