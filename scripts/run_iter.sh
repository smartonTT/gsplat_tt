#!/usr/bin/env bash
# Single-command worker entry point for one iter.
#
# Usage: scripts/run_iter.sh <iter_num> <slug> <class> [--profile]
#   <iter_num>  e.g. 1, 2, 47
#   <slug>      kebab-case label, e.g. "dst-resident-state"
#   <class>     one of: kernel-algebra | precompute | dispatch | binning | sort | host-prep
#   --profile   (optional) build with Tracy enabled and capture per-zone timings.
#               Adds ~5-10% kernel overhead, so OFF by default. Use when an iter
#               is surprising or we plateau and need per-zone attribution.
#
# Steps:
#   1. clean tree check
#   2. devsync gate
#   3. JIT cache wipe if needed
#   4. remote build (Tracy-enabled if --profile)
#   5. remote render (training-pattern cycles; Tracy capture if --profile)
#   6. scp results back (+ .tracy + zones.csv if --profile)
#   7. compute_metrics locally (merges zones.csv if present)
#   8. (supervisor invokes dispatch_validator separately)
#   9. (supervisor invokes decide_and_log separately)
#  10. build_report
#
# Returns exit 0 + path to metrics.json on stdout. Non-zero exit = the experiment
# failed before producing artifacts; supervisor handles per §4 decision matrix.
set -euo pipefail

ITER_NUM="${1:?usage: $0 <iter_num> <slug> <class> [--profile]}"
SLUG="${2:?slug required}"
CLASS="${3:?class required}"
PROFILE=0
if [[ "${4:-}" == "--profile" ]]; then PROFILE=1; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ITER_NAME="$(printf "iter-%03d-%s" "$ITER_NUM" "$SLUG")"
ITER_DIR="$REPO/docs/optimization-log/screenshots/$ITER_NAME"
SENTINEL="$REPO/.opt-v2-last-build-commit"
BOX_USER="smarton"
BOX_HOST="yyzo-bh-14"
REMOTE_REPO="/proj_sw/user_dev/smarton/gsplat_tt"
# Local python on Mac: project venv has broken symlinks (it points to /opt/venv which is bh-14-only),
# so use /usr/bin/python3 which has numpy + PIL + matplotlib installed system-wide. Remote python on
# bh-14 is unaffected (the box has /opt/venv).
LOCAL_PY="${LOCAL_PY:-/usr/bin/python3}"

mkdir -p "$ITER_DIR"

# Step 1: clean tree check
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "ERROR: working tree not clean; supervisor must resolve before next iter" >&2
  git -C "$REPO" status --short >&2
  exit 10
fi

# Step 2: devsync gate
if ! devsync is-finished "$BOX_HOST"; then
  echo "ERROR: devsync to $BOX_HOST not finished" >&2
  exit 11
fi

# Step 3: JIT cache wipe if kernel cpp/hpp or CT-args header changed since last build
LAST_BUILD="$(cat "$SENTINEL" 2>/dev/null || echo "")"
CURR="$(git -C "$REPO" rev-parse HEAD)"
NEEDS_WIPE=0
if [[ -z "$LAST_BUILD" ]]; then
  NEEDS_WIPE=1  # first build on opt-v2
else
  # NB: real source root is .../tt-metal/tt_metal/programming_examples/...
  # (the doubled `tt-metal/tt_metal/` is intentional — outer is the vendored
  # repo dir, inner is its source subdir). The old filter was missing the
  # inner segment and silently never wiped the JIT cache for kernel edits,
  # so iter-N ran with iter-(N-1) cached kernels (e.g. M2 ran cached M1 → hang).
  CHANGED="$(git -C "$REPO" diff --name-only "$LAST_BUILD" "$CURR" -- \
    'backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/**/*.cpp' \
    'backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/**/*.hpp' \
    'backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/*.cpp' \
    'backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/*.hpp' || true)"
  if [[ -n "$CHANGED" ]]; then
    NEEDS_WIPE=1
  fi
fi
if [[ "$NEEDS_WIPE" == "1" ]]; then
  echo "[run_iter] wiping JIT cache on $BOX_HOST"
  ssh "$BOX_USER@$BOX_HOST" 'rm -rf /localdev/smarton/.cache/tt-metal-cache/' >>"$ITER_DIR/build.log" 2>&1
fi

# Step 4: remote build
echo "[run_iter] building on $BOX_HOST (profile=$PROFILE)"
if [[ "$PROFILE" == "1" ]]; then
  # Tracy-enabled build in build-tracy/. Configure only if build.ninja missing
  # (re-configuring an existing cache can clobber generator state). Pin clang-20
  # explicitly — gcc default fails the "GCC-12+ required" check on this box.
  BUILD_CMD="cd $REMOTE_REPO && \
    if [[ ! -f backends/tt/tt-metal/build-tracy/build.ninja ]]; then \
      sudo rm -rf backends/tt/tt-metal/build-tracy && \
      sudo cmake -G Ninja -S backends/tt/tt-metal -B backends/tt/tt-metal/build-tracy \
        -DENABLE_TRACY=ON -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PROGRAMMING_EXAMPLES=ON \
        -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20; \
    fi && \
    sudo ninja -C backends/tt/tt-metal/build-tracy metal_example_gaussian_splatting"
else
  BUILD_CMD="cd $REMOTE_REPO && sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting"
fi
TIMEOUT_BIN="$(command -v gtimeout || command -v timeout || true)"
if [[ -z "$TIMEOUT_BIN" ]]; then
  echo "[run_iter] ERROR: need 'timeout' (Linux) or 'gtimeout' (macOS: brew install coreutils)" >&2
  exit 2
fi
"$TIMEOUT_BIN" 240 ssh "$BOX_USER@$BOX_HOST" "$BUILD_CMD" \
  >>"$ITER_DIR/build.log" 2>&1 || { touch "$ITER_DIR/BUILD_FAIL"; exit 2; }
echo "$CURR" > "$SENTINEL"

# Step 5: remote render (training-pattern cycles)
echo "[run_iter] rendering on $BOX_HOST"
REMOTE_OUT="/tmp/$ITER_NAME"
ssh "$BOX_USER@$BOX_HOST" "mkdir -p $REMOTE_OUT"
if [[ "$PROFILE" == "1" ]]; then
  # Start tracy-capture in the background, then render, then stop capture.
  # tracy-capture listens on TCP 8086 by default; -o writes the .tracy file.
  # Also wipe & collect the device-profiler output dir (per-RISC durations CSV
  # — that's the headline bound-class signal per tt-buddy interpretation.md).
  ssh "$BOX_USER@$BOX_HOST" "
    set -e
    cd $REMOTE_REPO
    # Kill any stale daemons that might still be holding the Wormhole device.
    # iter-PROF first attempt hung 33+ minutes because a daemon from a prior
    # session was still alive. Belt-and-suspenders since render_fixed.py spawns
    # its own daemon: anything holding the device WILL hang the new spawn.
    pkill -f 'metal_example_gaussian_splatting' 2>/dev/null || true
    sleep 1
    export TT_METAL_DEVICE_PROFILER=1
    export TT_METAL_HOME=$REMOTE_REPO/backends/tt/tt-metal
    # Daemon phase-split inside kernel_ms. Adds Finish() barriers between
    # upload / dispatch / readback so the report can attribute the host-side
    # bound. Gated on this env var so the prod daemon stays overhead-free.
    export GSPLAT_PROFILE_PHASES=1
    rm -rf \$TT_METAL_HOME/generated/profiler/.logs 2>/dev/null || true
    # tracy-capture is the network-mode capture tool. We DON'T require it —
    # the per-RISC device_profile CSV (written by TT_METAL_DEVICE_PROFILER=1)
    # is what the bound-class classifier and the new kernel-side DeviceZone
    # zones depend on. Tracy is a nice-to-have for the full timeline view.
    CAP=
    if command -v tracy-capture >/dev/null 2>&1; then
      tracy-capture -o $REMOTE_OUT/iter.tracy -a 127.0.0.1 >/tmp/tracy-cap.log 2>&1 &
      CAP=\$!
      sleep 1
    else
      echo \"tracy-capture not found on PATH; skipping Tracy timeline capture (device_profile CSV is still collected)\" >&2
    fi
    MESH_DEVICE=P100 TT_BINARY_PATH=backends/tt/tt-metal/build-tracy/programming_examples/metal_example_gaussian_splatting GSPLAT_PROFILE_PHASES=1 ./venv/bin/python scripts/render_fixed.py --cycles --backend tt --scene stitch --size 1024 --warmup-cycles 1 --measure-cycles 10 --out-dir $REMOTE_OUT
    if [[ -n \"\$CAP\" ]]; then
      sleep 2
      kill -TERM \$CAP 2>/dev/null || true
      wait \$CAP 2>/dev/null || true
    fi
    # tracy-csvexport may not be on PATH on every box — don't let a missing
    # tool abort the rest of the script (especially the profiler CSV copy).
    if command -v tracy-csvexport >/dev/null 2>&1 && [[ -f $REMOTE_OUT/iter.tracy ]]; then
      tracy-csvexport $REMOTE_OUT/iter.tracy > $REMOTE_OUT/zones.csv || true
    else
      echo \"tracy-csvexport not found on PATH or no .tracy capture; skipping zones.csv\" >&2
      : > $REMOTE_OUT/zones.csv
    fi
    # Copy device-profiler CSVs. tt-metal writes them to .logs/, not reports/.
    # The reports/ dir is only populated by tools/tracy/process_device_log.py
    # which we don't run on the box.
    PROF_LOGS=\$TT_METAL_HOME/generated/profiler/.logs
    if [[ -d \$PROF_LOGS ]]; then
      mkdir -p $REMOTE_OUT/device_profile
      cp \$PROF_LOGS/*.csv $REMOTE_OUT/device_profile/ 2>/dev/null || true
      cp \$PROF_LOGS/*.log $REMOTE_OUT/device_profile/ 2>/dev/null || true
      ls $REMOTE_OUT/device_profile/ || true
    fi
  " >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/DEVICE_FAIL"; exit 3; }
else
  ssh "$BOX_USER@$BOX_HOST" \
    "cd $REMOTE_REPO && MESH_DEVICE=P100 ./venv/bin/python scripts/render_fixed.py --cycles --backend tt --scene stitch --size 1024 --warmup-cycles 1 --measure-cycles 10 --out-dir $REMOTE_OUT" \
    >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/DEVICE_FAIL"; exit 3; }
fi

# Step 6: scp results back
# macOS scp (sftp-based) doesn't expand brace patterns server-side, so do per-file copies.
for f in hero.png side.png top.png timing.jsonl; do
  scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/$f" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1
done
if [[ "$PROFILE" == "1" ]]; then
  scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/iter.tracy" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1 || true
  scp "$BOX_USER@$BOX_HOST:$REMOTE_OUT/zones.csv" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1 || true
  # Per-RISC device profile CSVs (ops_perf_results_*.csv, profile_log_device.csv)
  mkdir -p "$ITER_DIR/device_profile"
  scp -r "$BOX_USER@$BOX_HOST:$REMOTE_OUT/device_profile/" "$ITER_DIR/" >>"$ITER_DIR/run.log" 2>&1 || true
fi

# Step 7: compute metrics locally
PREV_BEST="$(jq -r '[.[]?] | map(select(.action=="commit")) | min_by(.kernel_ms_median).kernel_ms_median // "Infinity"' \
              < <(cat "$REPO/docs/optimization-log/iters.jsonl" | jq -s '.') 2>/dev/null || echo "Infinity")"
"$LOCAL_PY" "$REPO/scripts/compute_metrics.py" \
  --iter-dir "$ITER_DIR" \
  --reference-dir "$REPO/benchmarks/reference" \
  --class "$CLASS" \
  --prev-best-ms "$PREV_BEST" >>"$ITER_DIR/run.log" 2>&1

# Step 7b: device-side bound-class classifier (only when --profile is on
# and the per-RISC CSV is present). Writes device_profile/classification.json
# which the report generator consumes for the overhead_ratio + per-RISC table.
if [[ "$PROFILE" == "1" && -f "$ITER_DIR/device_profile/profile_log_device.csv" ]]; then
  HOST_KERNEL_MS="$(jq -r '.kernel_ms_median // empty' "$ITER_DIR/metrics.json" 2>/dev/null || echo "")"
  HK_FLAG=()
  if [[ -n "$HOST_KERNEL_MS" ]]; then HK_FLAG=(--host-kernel-ms "$HOST_KERNEL_MS"); fi
  "$LOCAL_PY" "$REPO/scripts/analyze_device_profile.py" \
    --csv "$ITER_DIR/device_profile/profile_log_device.csv" \
    "${HK_FLAG[@]}" \
    --out-json "$ITER_DIR/device_profile/classification.json" \
    >>"$ITER_DIR/run.log" 2>&1 || true
fi

# Step 7c: unified pipeline profile report. Persistent — runs every iter
# regardless of --profile so we always have a sub-stage breakdown to compare
# against. Device-side bound-class numbers are richer when --profile is on
# (device_profile/classification.json present); without it the report still
# emits the full host-side breakdown.
"$LOCAL_PY" "$REPO/scripts/pipeline_profile_report.py" \
  --iter-dir "$ITER_DIR" >>"$ITER_DIR/run.log" 2>&1 || true

# Step 10: rebuild report (validator + decide steps happen via separate supervisor calls)
"$LOCAL_PY" "$REPO/docs/optimization-log/build_report.py" >>"$ITER_DIR/run.log" 2>&1

echo "$ITER_DIR/metrics.json"
