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

ITER_DIR="${1:?usage: capture_tracy.sh <iter-dir>}"
export TTW_ITER_DIR="$ITER_DIR"

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
cd "$REPO" || { echo "[capture_tracy] FATAL: cannot cd $REPO" >&2; exit 1; }
OUTDIR="$REPO/opt/profiler/wrap_out_${ITER_DIR}"
TRACY="$OUTDIR/.logs/tracy_profile_log_host.tracy"
DLOG="$OUTDIR/.logs/profile_log_device.csv"
DST="$REPO/opt/profiler/${ITER_DIR}/render.tracy"
DST_CSV="$REPO/opt/profiler/${ITER_DIR}/profile_log_device.csv"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR" "$(dirname "$DST")"

if [[ ! -f "$REPO/.venv/bin/activate" ]]; then
  echo "[capture_tracy] FATAL: missing $REPO/.venv (loguru/tracy need venv)" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"

echo "[capture_tracy] render_clean FULL 30-view capture (iter-dir=$ITER_DIR) -> $DST"
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
    echo "[capture_tracy] device profiler CSV data rows: $rows (1-view baseline ~65600; ~30x => full 30-view)"
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
