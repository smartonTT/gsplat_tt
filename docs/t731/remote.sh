#!/bin/bash
# t731: one device-profiler (zone) run of train-mode b2b on the p150 measurement box.
#   remote.sh <arm>   arms:
#     prof   orbit (benchmarks/cameras_v2.json) views 0:4, --back-to-back, GSPLAT_TT_BLEND_TRAIN=1,
#            GSPLAT_TT_BLEND_PROF=1 (per-tile blend zones), TT_METAL_DEVICE_PROFILER=1.
#            render/eth_default.py (t731) switches to the 36 KB profiler overlay $T/tmp/ttm-eth12p36
#            and its own JIT cache by itself; nothing profiler-specific is set here.
# The device CSV (profile_log_device.csv) is gzipped to $S/<arm>-device.csv.gz.
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
          args=(--back-to-back --view-range 0:4) ;;
    *) echo "bad arm $arm"; exit 2 ;;
  esac
  # Profiler output: tt-metal writes under <runtime root>/generated/profiler; the overlay's
  # generated/ is a symlink into $TT_METAL_HOME, so clear the old CSV there first.
  P=$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv
  rm -f "$P" $T/tmp/ttm-eth12p36/generated/profiler/.logs/profile_log_device.csv 2>/dev/null
  echo "=== $arm $(cut -c1-8 SHA) ${env[*]} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  env "${env[@]}" GSPLAT_TT_ETH_CACHE=$T/tmp/ttm-eth12-cache-t731 TT_METAL_CACHE_RENDER=$C/ttmc-t731-$arm \
    timeout ${RUN_TIMEOUT:-400} python3 render/run.py --no-ref --cameras $cam --iter-dir t731-$arm "${args[@]}" > $S/run-$arm.log 2>&1
  rc=$?
  echo "run rc=$rc"
  grep -E "^(B2B |SUMMARY)|\[eth\]|^\[DEV\] dispatch|TTW_TIMING (ms_view|b2b_ms_frame|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|overflows|error:" $S/run-$arm.log | cut -c1-400 | head -16
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $arm: tt-smi -r"; tt-smi -r > $S/reset-$arm.log 2>&1; echo "reset rc=$?"; exit 124; fi
  [ $rc = 0 ] || fail=1
  cp $T/tmp/ttm-eth12p36/.gsplat-eth-overlay $S/overlay-marker.txt 2>/dev/null
  for f in "$P" $T/tmp/ttm-eth12p36/generated/profiler/.logs/profile_log_device.csv; do
    if [ -s "$f" ]; then ls -l --time-style=+%T "$f"; echo "device csv rows $(wc -l < "$f")"; gzip -c "$f" > $S/$arm-device.csv.gz; break; fi
  done
  [ -s $S/$arm-device.csv.gz ] || { echo "NO_DEVICE_CSV"; fail=1; }
done
echo "=== done $(date +%T)"
exit $fail
