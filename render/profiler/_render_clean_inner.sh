#!/usr/bin/env bash
# Inner render for the render_clean Tracy capture, run by `python -m tracy` as a
# subprocess (passing a shell script — not python — makes the wrapper use
# subprocess.run() instead of exec()ing it in-process, avoiding the wrapper's
# `import ttnn`; render_clean uses its own pybind module). All TT_METAL_* /
# GSPLAT_TT_PROFILE / TRACY_PORT / TT_METAL_PROFILER_MID_RUN_DUMP env is inherited
# from capture_tracy_clean.sh -> tracy wrapper -> here.
#
# Renders the FULL 30-view bicycle bench set through render_clean (run.py), with
# --no-ref so the trace contains ONLY render_clean's per-view device zones (the
# cpu_cpp_mb reference is CPU-only and is skipped). render_clean's per-frame
# maybe_dump_device_profiler() (GSPLAT_TT_PROFILE=1) pushes each view's device
# zones into the live stream under --dump-device-data-mid-run.
set -uo pipefail
source .venv/bin/activate 2>/dev/null || true
echo "[render-clean-inner] python3 render/run.py --no-ref --iter-dir rc-tracy" \
     "(MID_RUN_DUMP=${TT_METAL_PROFILER_MID_RUN_DUMP:-unset}" \
     "DEVICE_PROFILER=${TT_METAL_DEVICE_PROFILER:-unset}" \
     "GSPLAT_TT_PROFILE=${GSPLAT_TT_PROFILE:-unset} TRACY_PORT=${TRACY_PORT:-unset})"
python3 render/run.py --no-ref --iter-dir rc-tracy
echo "[render-clean-inner] run.py rc=$?"
# Exit 0 so the wrapper's subprocess.run(check=True) finalizes the capture; the
# real rc is echoed above and device-zone coverage is verified after.
exit 0
