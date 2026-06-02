#!/usr/bin/env bash
# Tracy DEVICE-timeline capture for the FULL L1-RESIDENT bucket path (NOT the
# production FUSED_TILE path). Same supported `python -m tracy` wrapper +
# never-closes-the-device fix as run_wrapper_capture.sh; only the gsplat flag
# set differs (see L1-RESIDENT CONFIG below).
#
# Runs UNDER devrun.sh (TTW_DEVRUN=1 exported, cwd == remote_root, device flock
# held for the whole job).
#
# WHY a separate capture: production streams the per-candidate gather continuously
# on the data-movers (BRISC/NCRISC). The L1-resident "bucket" path instead loads
# each tile's full splat records ONCE (bulk contiguous DRAM->L1) then sorts+culls+
# blends in L1 with NO per-candidate gather. The Tracy device timeline should show
# the movers spike-then-go-quiet (load-once) rather than continuously streaming.
#
# ROOT CAUSE of the earlier "markers-as-data-but-no-device-timeline" capture:
#   gsplat's maybe_dump_device_profiler() (GSPLAT_TT_PROFILE=1) calls
#   tt_metal::ReadMeshDeviceProfilerResults(NORMAL) after each frame. In tt-metal
#   (tt_metal_profiler.cpp), a NORMAL read only pushes Tracy device zones when
#   get_profiler_mid_run_dump() is true; otherwise that push happens only at device
#   CLOSE -- and gsplat never closes the device.
# FIX: enable the wrapper's --dump-device-data-mid-run (TT_METAL_PROFILER_MID_RUN_DUMP=1)
#   so the per-frame ReadMeshDeviceProfilerResults(NORMAL) ALSO streams the device
#   zones (with clock-domain calibration) into the live Tracy stream.
set -uo pipefail

export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
export TT_METAL_ARCH_NAME=blackhole
export MESH_DEVICE=P100
# `python -m tracy` needs the tracy package's parent dir on the path.
export PYTHONPATH=/localdev/smarton/tt-metal/tools:${PYTHONPATH:-}

# gsplat device-resident pipeline knobs (mirror ttw.toml [run].verify_cmd).
export GSPLAT_TT_BLEND_MODE=2
export GSPLAT_TT_MB_KERNEL=1 GSPLAT_TT_DEVICE_PROJECT=1
export GSPLAT_TT_RESIDENT_PROJECT=1 GSPLAT_TT_RESIDENT_GATHER=1 GSPLAT_TT_DEVICE_TILE_ASSIGN=1
export GSPLAT_TT_RESIDENT_TA_IN=1 GSPLAT_TT_DEVICE_SORT=1 GSPLAT_TT_RESIDENT_PAIRS=1
export GSPLAT_TT_RESIDENT_BLEND=1 GSPLAT_TT_SORT_DEVICE_PUBLISH=1 GSPLAT_TT_TA_DEVICE_SCAN=1
export GSPLAT_TT_PROJ_DEVICE_SCAN=1 GSPLAT_TT_SFPU_CULL=1 GSPLAT_TT_MB_TIMING=1

# ── L1-RESIDENT CONFIG (the cull-fold winning config, blend-data-movement-plan §13,
#    commit 2d1f93f; lessons.md 2026-06-01) ────────────────────────────────────
#   - GSPLAT_TT_FUSED_TILE is NOT set (production path OFF).
#   - GSPLAT_TT_TILE_BUCKET=1 + GSPLAT_TT_BUCKET_FIT=8192 select the per-tile
#     contiguous full-record bucket path (load-once into L1, sort+cull+blend in L1).
#   - blend_aos / proj_m_blendrec (the AoS record TILE_BUCKET requires) is enabled
#     automatically: blend_aos = (SFPU_CULL && GSPLAT_TT_BLEND_AOS != 0); SFPU_CULL=1
#     above and BLEND_AOS is left default-on, so proj_m_blendrec is emitted (sort_device.cpp
#     errors "TILE_BUCKET set but proj_m_blendrec missing" otherwise).
#   - GSPLAT_TT_BUCKET_MASK is deliberately NOT set: dropping it means the single
#     standalone SFPU cull (CULL_SPLIT) fills cull_masks and the bucket reader shares
#     it from L1 (cull_base+k) -- no redundant 2nd cull pass (sort 99->26 ms).
#   - MB_BUCKET_CB_FENCE auto-forces ON whenever the bucket path is active (the
#     proven-necessary fast-producer-race fix); no flag needed.
export GSPLAT_TT_TILE_BUCKET=1
export GSPLAT_TT_BUCKET_FIT=8192

OUTDIR=/localdev/smarton/gstt2/opt/profiler/wrap_out_l1
TRACY="$OUTDIR/.logs/tracy_profile_log_host.tracy"
TRACY_IDEAL_MAC=/localdev/smarton/gstt2/opt/profiler/render-ideal-labeled.tracy
CSV="$TT_METAL_HOME/build/tools/profiler/bin/csvexport-release"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

source .venv/bin/activate 2>/dev/null || true

# Device-timeline triggers (BOTH required):
#   GSPLAT_TT_PROFILE=1            -> gsplat calls ReadMeshDeviceProfilerResults() per frame.
#   TT_METAL_DEVICE_PROFILER=1     -> profiler enabled at device init (wrapper also sets this).
# --dump-device-data-mid-run sets TT_METAL_PROFILER_MID_RUN_DUMP=1 so the per-frame
# read PUSHES the device zones to Tracy.
export TT_METAL_DEVICE_PROFILER=1
export GSPLAT_TT_PROFILE=1

echo "[tracy-wrap-l1] L1-RESIDENT flags: FUSED_TILE=unset TILE_BUCKET=$GSPLAT_TT_TILE_BUCKET BUCKET_FIT=$GSPLAT_TT_BUCKET_FIT SFPU_CULL=$GSPLAT_TT_SFPU_CULL BLEND_AOS=${GSPLAT_TT_BLEND_AOS:-default-on} BUCKET_MASK=${GSPLAT_TT_BUCKET_MASK:-unset}"
echo "[tracy-wrap-l1] CMD: python3 -m tracy -r -p -v --dump-device-data-mid-run -o $OUTDIR opt/profiler/_render_inner.sh"
python3 -m tracy -r -p -v --dump-device-data-mid-run -o "$OUTDIR" opt/profiler/_render_inner.sh
RC=$?
echo "[tracy-wrap-l1] wrapper exited rc=$RC"

# Make sure no capture-release lingers and holds the devrun ssh pipe open.
pkill -f 'profiler/bin/capture[-]release' 2>/dev/null || true

if [[ -s "$TRACY" ]]; then
  cp -f "$TRACY" "$TRACY_IDEAL_MAC"
  echo "[tracy-wrap-l1] OK tracy: $(ls -la "$TRACY")"
  echo "[tracy-wrap-l1] copied -> $TRACY_IDEAL_MAC (scp to Mac opt/profiler/render-ideal-labeled.tracy)"
  echo "[tracy-wrap-l1] === host-zone export via csvexport-release (CPU zones; device zones are GPU zones, see below) ==="
  "$CSV" "$TRACY" > "$OUTDIR/zones.csv" 2>"$OUTDIR/zones.err" || true
  echo "[tracy-wrap-l1] csv zone rows: $(($(wc -l < "$OUTDIR/zones.csv" 2>/dev/null) - 1))"
  echo "[tracy-wrap-l1] === DEVICE GPU-context probe (TracyTTContext names embedded in the .tracy) ==="
  # pushTracyDeviceResults names each per-core device timeline context
  # "Device: <id>, Logical (x,y) Physical (x,y)"; presence => rendered device track.
  DEVCTX=$(strings -n 8 "$TRACY" | grep -c 'Device: [0-9]*, Logical')
  echo "[tracy-wrap-l1] device GPU-context (per-core timeline track) count: $DEVCTX"
  echo "[tracy-wrap-l1] sample device context names:"
  strings -n 8 "$TRACY" | grep 'Device: [0-9]*, Logical' | sort -u | head -8
  echo "[tracy-wrap-l1] device RISC zone-name strings present:"
  strings -n 3 "$TRACY" | grep -E '^(BRISC|NCRISC|TRISC|ERISC|TRISC0|TRISC1|TRISC2)$' | sort | uniq -c | head
  echo "[tracy-wrap-l1] === per-RISC device markers in profile_log_device.csv (data count) ==="
  DLOG="$OUTDIR/.logs/profile_log_device.csv"
  [[ -f "$DLOG" ]] && echo "[tracy-wrap-l1] profile_log_device.csv data rows: $(($(wc -l < "$DLOG") - 1))" || echo "[tracy-wrap-l1] (no profile_log_device.csv)"
else
  echo "[tracy-wrap-l1] FAIL: $TRACY missing/empty; listing $OUTDIR:"
  ls -laR "$OUTDIR" 2>&1
fi
echo "[tracy-wrap-l1] DONE rc=$RC tracy=$TRACY"
exit "$RC"
