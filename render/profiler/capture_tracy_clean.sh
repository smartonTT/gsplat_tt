#!/usr/bin/env bash
# ============================================================================
# Tracy device-timeline capture for render_clean (the render/ clean extract),
# the analog of opt/profiler/capture_tracy.sh but driving render/run.py instead
# of the production a003_verify.py. Profiles render_clean SPECIFICALLY.
#
#   bash render/profiler/capture_tracy_clean.sh        # -> opt/profiler/render-clean/render.tracy
#
# Must run UNDER devrun.sh (device flock held, cwd == remote repo root). Uses the
# proven device-timeline method: `python -m tracy --dump-device-data-mid-run`
# (TT_METAL_PROFILER_MID_RUN_DUMP=1) so render_clean's per-frame
# maybe_dump_device_profiler() (GSPLAT_TT_PROFILE) pushes each of the 30 views'
# device zones into the live stream (gsplat never close()s the device, so a
# normal capture-release would only see the host/JIT-warmup zone stream).
#
# render_clean's config is fully BAKED — it ignores GSPLAT_TT_* flags — so unlike
# the production capture we only set the two device-profiler triggers below.
# ============================================================================
set -uo pipefail

export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
export TT_METAL_ARCH_NAME=blackhole
export MESH_DEVICE=P100
# `python -m tracy` needs the tracy package's parent dir on the path.
export PYTHONPATH=/localdev/smarton/tt-metal/tools:${PYTHONPATH:-}

# Device-timeline triggers:
#   TT_METAL_DEVICE_PROFILER=1 -> profiler enabled at device init.
#   GSPLAT_TT_PROFILE=1        -> render_clean reads device profiler results per frame.
#   --dump-device-data-mid-run -> TT_METAL_PROFILER_MID_RUN_DUMP=1 (push mid-run).
export TT_METAL_DEVICE_PROFILER=1
export GSPLAT_TT_PROFILE=1

REPO=/localdev/smarton/gstt2
OUTDIR="$REPO/opt/profiler/wrap_out_render-clean"
TRACY="$OUTDIR/.logs/tracy_profile_log_host.tracy"
DLOG="$OUTDIR/.logs/profile_log_device.csv"
DST="$REPO/opt/profiler/render-clean/render.tracy"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$(dirname "$DST")"

source .venv/bin/activate 2>/dev/null || true

echo "[capture_clean] render_clean FULL 30-view capture -> $DST"
echo "[capture_clean] CMD: python3 -m tracy -r -p -v --dump-device-data-mid-run -o $OUTDIR render/profiler/_render_clean_inner.sh"
python3 -m tracy -r -p -v --dump-device-data-mid-run -o "$OUTDIR" render/profiler/_render_clean_inner.sh
RC=$?
echo "[capture_clean] wrapper exited rc=$RC"

# Make sure no capture-release lingers and holds the devrun ssh pipe open.
pkill -f 'profiler/bin/capture[-]release' 2>/dev/null || true

if [[ -s "$TRACY" ]]; then
  cp -f "$TRACY" "$DST"
  echo "[capture_clean] OK tracy: $(ls -la "$DST")"
  if [[ -f "$DLOG" ]]; then
    rows=$(($(wc -l < "$DLOG") - 1))
    echo "[capture_clean] device profiler CSV data rows: $rows (1-view baseline ~65600; ~30x => full 30-view)"
    echo "[capture_clean] per-zone-hash device marker counts (each zone repeats ~30x across views):"
    awk -F, 'NR>1 {print $5}' "$DLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -25
  else
    echo "[capture_clean] WARN: no $DLOG (cannot prove device-zone coverage)"
  fi
else
  echo "[capture_clean] FAIL: $TRACY missing/empty; listing $OUTDIR:"
  ls -laR "$OUTDIR" 2>&1
fi
echo "[capture_clean] DONE rc=$RC tracy=$DST"
exit "$RC"
