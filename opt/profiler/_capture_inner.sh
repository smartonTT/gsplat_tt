#!/usr/bin/env bash
# Inner render for the canonical Tracy capture (opt/profiler/capture_tracy.sh),
# invoked by `python -m tracy` as a SUBPROCESS: passing a shell script (not
# python) makes the wrapper subprocess.run() it instead of exec()ing it
# in-process, which avoids the wrapper's `import ttnn` (render_clean uses its own
# pybind module). All TT_METAL_* / GSPLAT_TT_PROFILE / TRACY_PORT /
# TT_METAL_PROFILER_MID_RUN_DUMP / TTW_ITER_DIR env is inherited from
# capture_tracy.sh -> tracy wrapper -> here.
#
# Renders the FULL 30-view bicycle bench through render_clean (run.py) with the
# SAME gsplat flags as ttw.toml [run] verify_cmd (--iter-dir), plus --no-ref so
# the trace contains ONLY render_clean's per-view device zones (the cpu_cpp_mb
# reference is CPU-only and is skipped). render_clean's per-frame
# maybe_dump_device_profiler() (GSPLAT_TT_PROFILE=1) pushes each view's device
# zones into the live stream under --dump-device-data-mid-run.
set -uo pipefail
REPO=/localdev/smarton/gstt2
cd "$REPO" || exit 1
ITER_DIR="${TTW_ITER_DIR:-tracy}"
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"
echo "[capture-inner] python3 render/run.py --no-ref --iter-dir $ITER_DIR" \
     "(MID_RUN_DUMP=${TT_METAL_PROFILER_MID_RUN_DUMP:-unset}" \
     "DEVICE_PROFILER=${TT_METAL_DEVICE_PROFILER:-unset}" \
     "GSPLAT_TT_PROFILE=${GSPLAT_TT_PROFILE:-unset} TRACY_PORT=${TRACY_PORT:-unset})"
python3 render/run.py --no-ref --iter-dir "$ITER_DIR"
echo "[capture-inner] run.py rc=$?"
# Exit 0 so the wrapper's subprocess.run(check=True) finalizes the capture; the
# real rc is echoed above and device-zone coverage is verified after.
exit 0
