#!/bin/bash
# t484: one arm of the pfwc b2b profile on the measurement box (via devrun.sh, inside drive.sh's
# ttp lock p100).  remote_prof.sh <arm>
#   u   untraced b2b headline (1 check + 1 warm-up + 20 passes), GSPLAT_B2B_ALL_STAGES=1
#   z   device profiler, kernel zones only, b2b --b2b-passes 1 --b2b-warmup 0 (90 views)
#   s   as z + GSPLAT_TT_PFWC_STEPCYC=1 on all five RISCs (per-step wall cycles, task #197)
#   sr  as s + GSPLAT_TT_PFWC_SKIP_RGB=1
# Profiler arms read the device profiler after every view (GSPLAT_TT_PROFILE=1), so their period
# is not the headline; only the per-core in-kernel split is used from them.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_SKIP_RGB GSPLAT_TT_PFWC_STEPCYC \
  TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE GSPLAT_TT_KCFG_EXTRA_KB
T=${T:-/localdev/smarton/gstt2-t469}; cd "$T" || exit 1; source .venv/bin/activate
export PYTHONPATH=$TT_METAL_HOME/tools:${PYTHONPATH:-}
S=$T/tmp/t484; mkdir -p $S
arm=${1:?arm}
args=(--no-ref --back-to-back --iter-dir t484-$arm)
envs=(GSPLAT_B2B_ALL_STAGES=1)
case $arm in
  u) cache=ttmc-gstt2-t481 ;;
  z|s|sr)
    python3 opt/profiler/zone_hash_check.py --repo "$T" || { echo "ZONE_HASH_COLLISION"; exit 8; }
    cache=ttmc-gstt2-t484-prof
    args+=(--b2b-passes 1 --b2b-warmup 0)
    rm -rf $S/prof-$arm; mkdir -p $S/prof-$arm
    # The device CSV only reaches disk through the Tracy mid-run dump (as opt/profiler/capture_tracy.sh):
    # render_clean never closes the device. Output: $S/prof-$arm/.logs/profile_log_device.csv.
    envs+=(TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1)
    # KCFG: host default with the profiler is +32 KB; the fused programs' static CBs overflow L1
    # above ~+34 KB, so only KX (one retry on a too-large pfwc) overrides it.
    case $arm in s|sr) envs+=(GSPLAT_TT_PFWC_STEPCYC=1 GSPLAT_TT_PFWC_STEPRISC=${SR:-9})
      [ -n "${KX:-}" ] && envs+=(GSPLAT_TT_KCFG_EXTRA_KB=$KX) ;; esac
    [ $arm = sr ] && envs+=(GSPLAT_TT_PFWC_SKIP_RGB=1) ;;
  *) echo "bad arm $arm"; exit 2 ;;
esac
echo "=== $arm $(cut -c1-7 SHA) ${envs[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
if [ $arm = u ]; then
  env "${envs[@]}" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/$cache timeout ${RUN_TIMEOUT:-500} \
    python3 render/run.py "${args[@]}" > $S/run-$arm.log 2>&1
  rc=$?
else
  printf '#!/bin/bash\ncd %q && exec python3 render/run.py %s\n' "$T" "${args[*]}" > $S/inner-$arm.sh
  chmod +x $S/inner-$arm.sh
  # Own session so a lingering capture-release is stopped by our pgid, never by name.
  env "${envs[@]}" TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/$cache setsid timeout ${RUN_TIMEOUT:-500} \
    python3 -m tracy -r -p -v --dump-device-data-mid-run -o $S/prof-$arm $S/inner-$arm.sh \
    > $S/run-$arm.log 2>&1 < /dev/null &
  pg=$!; wait $pg; rc=$?
  kill -- -$pg 2>/dev/null
  grep -q 'Traceback' $S/run-$arm.log && rc=1
fi
echo "run rc=$rc"
grep -E "^(B2B |B2B_STAGES|SUMMARY)|\[run\] B2B pass|TTW_TIMING (ms_view|b2b_ms_frame)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$arm.log | cut -c1-600 | head -40
if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $arm: tt-smi -r"; tt-smi -r > $S/reset-$arm.log 2>&1; echo "reset rc=$?"; exit 124; fi
if [ -d $S/prof-$arm ]; then
  C=$(find $S/prof-$arm -name profile_log_device.csv | head -1)
  if [ -n "$C" ]; then echo "csv rows $(wc -l < $C)"; gzip -c "$C" > $S/dev-$arm.csv.gz; rm -rf $S/prof-$arm
  else echo "NO_DEVICE_CSV"; ls -R $S/prof-$arm | head; [ $rc = 0 ] && rc=7; fi
fi
echo "=== done $arm rc=$rc $(date +%T)"
exit $rc
