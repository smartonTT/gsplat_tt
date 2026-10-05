#!/usr/bin/env bash
# ============================================================================
# CANONICAL 30-view Tracy DEVICE-zone capture for render_clean (render/run.py).
# Reconstructed per ttw.toml [profile] (tracy_capture_cmd / tracy_views = 30).
#
#   bash opt/profiler/capture_tracy.sh <iter-dir>     # e.g. ttw-104
#   -> opt/profiler/<iter-dir>/render.tracy  (+ profile_log_device.csv)
#
# MUST run UNDER devrun.sh (device flock held; cwd == remote repo root). Uses the
# proven device-timeline method `python -m tracy --dump-device-data-mid-run`
# (TT_METAL_PROFILER_MID_RUN_DUMP=1): gsplat/render_clean never close()s the
# device, so a normal capture-release would see only the host/JIT-warmup zone
# stream — the mid-run dump is the ONLY path that streams each of the 30 views'
# per-stage DEVICE zones (proj/ta/sort/cull/blend). Renders the FULL 30-view
# bench (run.py default) with the SAME gsplat flags as ttw.toml [run] verify_cmd
# (--iter-dir), plus --no-ref so the trace holds ONLY render_clean device zones
# (the cpu_cpp reference is CPU-only). Mirrors render/profiler/capture_tracy_clean.sh.
# ============================================================================
set -uo pipefail

ITER_DIR="${1:?usage: capture_tracy.sh <iter-dir> [START:END]}"
# Optional view chunk (python slice of the 30-view order). Each chunk is its own
# devrun job so a long capture stays under the 600 s ceiling; stitch the chunk
# CSVs with opt/profiler/stitch_device_csv.py (see capture_tracy_chunked.sh).
VIEW_RANGE="${2:-}"
export TTW_ITER_DIR="$ITER_DIR" TTW_VIEW_RANGE="$VIEW_RANGE"
SUB=""
[[ -n "$VIEW_RANGE" ]] && SUB="/chunks/${VIEW_RANGE/:/-}"

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
# The device profiler grows kernel binaries. The host sizes the Tensix kernel config buffer for
# the pfwc config plus 8 KB when TT_METAL_DEVICE_PROFILER is on (render/host/kcfg_size.h), so
# leave GSPLAT_TT_KCFG_EXTRA_KB unset unless overriding: task #258 found the old default of 4
# here replaced the +24 KB the default pfwc writer split needs and failed with
# TT_FATAL state.offset <= max_size.

REPO="${GSTT2_REPO:-/localdev/smarton/gstt2}"  # override to capture from another tree
cd "$REPO" || { echo "[capture_tracy] FATAL: cannot cd $REPO" >&2; exit 1; }
OUTDIR="$REPO/opt/profiler/wrap_out_${ITER_DIR}${SUB//\//_}"
TRACY="$OUTDIR/.logs/tracy_profile_log_host.tracy"
DLOG="$OUTDIR/.logs/profile_log_device.csv"
DST="$REPO/opt/profiler/${ITER_DIR}${SUB}/render.tracy"
DST_CSV="$REPO/opt/profiler/${ITER_DIR}${SUB}/profile_log_device.csv"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$(dirname "$DST")"

if [[ ! -f "$REPO/.venv/bin/activate" ]]; then
  echo "[capture_tracy] FATAL: missing $REPO/.venv (loguru/tracy need venv)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"

echo "[capture_tracy] render_clean capture (iter-dir=$ITER_DIR views=${VIEW_RANGE:-all}) -> $DST"
PY="$REPO/.venv/bin/python3"
INNER="$REPO/opt/profiler/_capture_inner.sh"
echo "[capture_tracy] CMD: $PY -m tracy -r -p -v --dump-device-data-mid-run -o $OUTDIR $INNER"
"$PY" -m tracy -r -p -v --dump-device-data-mid-run -o "$OUTDIR" "$INNER"
RC=$?
echo "[capture_tracy] wrapper exited rc=$RC"

# Make sure no capture-release lingers and holds the devrun ssh pipe open.
pkill -f 'profiler/bin/capture[-]release' 2>/dev/null || true

if [[ -s "$TRACY" ]]; then
  cp -f "$TRACY" "$DST"
  echo "[capture_tracy] OK tracy: $(ls -la "$DST")"
  if [[ -f "$DLOG" ]]; then
    cp -f "$DLOG" "$DST_CSV"
    rows=$(($(wc -l < "$DLOG") - 1))
    echo "[capture_tracy] device profiler CSV data rows: $rows (~19.9k rows/view at ttw-142; 30 views + warmup ~616k)"
    echo "[capture_tracy] per-zone-hash device marker counts (each zone repeats ~30x across views):"
    awk -F, 'NR>1 {print $5}' "$DLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -25
  else
    echo "[capture_tracy] WARN: no $DLOG (cannot prove device-zone coverage)"
  fi
else
  echo "[capture_tracy] FAIL: $TRACY missing/empty; listing $OUTDIR:"
  ls -laR "$OUTDIR" 2>&1
fi
echo "[capture_tracy] DONE rc=$RC tracy=$DST"
exit "$RC"
