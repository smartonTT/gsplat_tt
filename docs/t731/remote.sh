#!/bin/bash
# t731: one device-profiler (zone) run of train-mode b2b on the p150 measurement box.
#   remote.sh <arm>   arms:
#     prof   orbit (benchmarks/cameras_v2.json) views 0:4, --back-to-back, GSPLAT_TT_BLEND_TRAIN=1,
#            GSPLAT_TT_BLEND_PROF=1 (per-tile blend zones), TT_METAL_DEVICE_PROFILER=1.
#            render/eth_default.py (t731) switches to the 36 KB profiler overlay $T/tmp/ttm-eth12p36
#            and its own JIT cache by itself; nothing profiler-specific is set here.
#            Run under `python -m tracy --dump-device-data-mid-run` (as #427/#479): gsplat never
#            closes the device, so a plain run.py reads the zones but never writes the CSV
#            (first t731 run: rc=0, NO_DEVICE_CSV). 1 check + 2 timed passes x 4 views = 12
#            renders, device buffers read once per pass (GSPLAT_TT_PROFILE_READ_EVERY=4).
# The device CSV (profile_log_device.csv) is gzipped to $S/<arm>-device.csv.gz and
# opt/profiler/analyze_zones.py's table (12 renders) goes to $S/<arm>-zones.txt.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_MATCULL_PROF GSPLAT_TT_BLEND_TRAIN GSPLAT_TT_BLEND_TRAIN_DUMP \
  GSPLAT_TT_BLEND_PROF TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR
T=${T:?T}
C=/localdev/smarton/.cache; S=$T/tmp/t731; mkdir -p $S
fail=0
cd "$T" || exit 1; source .venv/bin/activate
cam=benchmarks/cameras_v2.json
for arm in "$@"; do
  case $arm in
    prof) env=(GSPLAT_TT_BLEND_TRAIN=1 GSPLAT_TT_BLEND_PROF=1 TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1)
          args=(--back-to-back --view-range 0:4 --b2b-passes 2 --b2b-warmup 0) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  PR=$S/prof-$arm; rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t731 TT_METAL_CACHE_RENDER=$C/ttmc-t731-$arm GSPLAT_TT_PROFILE_READ_EVERY=4 \
  python3 render/run.py --no-ref --cameras $cam --iter-dir t731-$arm ${args[*]}
IN
  chmod +x $PR/inner.sh
  echo "=== $arm $(cut -c1-8 SHA) ${env[*]} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${env[@]}" TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-} \
    timeout ${RUN_TIMEOUT:-400} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $S/run-$arm.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|\[eth\]|^\[DEV\] dispatch|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|overflows|error:|DRAM buffers were full" $S/run-$arm.log | cut -c1-400 | head -16
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $arm: tt-smi -r"; tt-smi -r > $S/reset-$arm.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  cp $T/tmp/ttm-eth12p36/.gsplat-eth-overlay $S/overlay-marker.txt 2>/dev/null
  f=$(find $PR -name profile_log_device.csv | head -1)
  if [ -n "$f" ] && [ -s "$f" ]; then
    echo "device csv rows $(wc -l < "$f") blend zones $(grep -c ',bl' "$f")"
    gzip -c "$f" > $S/$arm-device.csv.gz
    python3 opt/profiler/analyze_zones.py "$f" 12 > $S/$arm-zones.txt 2>&1; echo "zones rc=$?"; head -40 $S/$arm-zones.txt
  else echo "NO_DEVICE_CSV"; fail=1; fi
  rm -rf $PR
done
echo "=== done $(date +%T)"
exit $fail
