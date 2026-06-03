#!/usr/bin/env bash
# Local-Mac iter driver. One command per iter end-to-end.
#
# Usage: scripts/run_iter.sh <iter_num> <slug> <class> [--backend cpu|cpu_cpp|cpu_ref]
#   <iter_num>  e.g. 0, 6, 47
#   <slug>      kebab-case label, e.g. "microblock-cull-numpy"
#   <class>     one of: baseline | scaffolding | port | algorithm | viewer
#   --backend   default cpu_cpp (Phase 2+); set to cpu_ref for iter-000
#
# Steps:
#   1. clean tree gate
#   2. build C++ (skipped if --backend=cpu_ref OR no src/ changes since last build)
#   3. run unit tests (Catch2 via ctest, skipped if cpu_ref)
#   4. render 30-frame benchmark
#   5. compute_metrics → metrics.json + diff10
#   6. (supervisor invokes dispatch_validator separately)
#   7. (supervisor invokes decide_and_log separately)
#   8. build_report.py refresh
#
# Returns 0 + path to metrics.json on stdout. Non-zero = experiment failed
# before producing artifacts.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

ITER_NUM="${1:?usage: $0 <iter_num> <slug> <class> [--backend ...]}"
SLUG="${2:?slug required}"
CLASS="${3:?class required}"
BACKEND="cpu_cpp"
if [[ "${4:-}" == "--backend" ]]; then BACKEND="${5:-cpu_cpp}"; fi

ITER_NAME="$(printf "iter-%03d-%s" "$ITER_NUM" "$SLUG")"
ITER_DIR="$ITER_DIR_PARENT/$ITER_NAME"
SENTINEL="$REPO_ROOT/.last-build-commit"
BUILD_DIR="$REPO_ROOT/build"

SCENE="${SCENE:-bicycle}"

mkdir -p "$ITER_DIR"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain | grep -v '^?? opt/screenshots/' | grep -v '^?? build/' || true)" ]]; then
  echo "ERROR: working tree not clean; supervisor must resolve before next iter" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 10
fi

# Step 2: build (only if C++ backend selected and src/ changed)
if [[ "$BACKEND" == "cpu_cpp" ]]; then
  CURR_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  LAST_SHA="$(cat "$SENTINEL" 2>/dev/null || echo "")"
  NEEDS_BUILD=1
  if [[ -d "$BUILD_DIR" && "$LAST_SHA" == "$CURR_SHA" ]]; then NEEDS_BUILD=0; fi

  # Force rebuild if anything in src/ or tests/ or CMakeLists changed since last build.
  if [[ "$NEEDS_BUILD" == "0" ]]; then
    if [[ -n "$LAST_SHA" ]]; then
      CHANGED="$(git -C "$REPO_ROOT" diff --name-only "$LAST_SHA" "$CURR_SHA" -- 'src/**' 'tests/**' 'CMakeLists.txt' || true)"
      [[ -n "$CHANGED" ]] && NEEDS_BUILD=1
    fi
  fi

  if [[ "$NEEDS_BUILD" == "1" ]]; then
    echo "[run_iter] cmake configure + build"
    if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
      cmake -G Ninja -S "$REPO_ROOT/src" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        >>"$ITER_DIR/build.log" 2>&1
    fi
    cmake --build "$BUILD_DIR" -j >>"$ITER_DIR/build.log" 2>&1 || { touch "$ITER_DIR/BUILD_FAIL"; exit 2; }
    echo "$CURR_SHA" > "$SENTINEL"
  fi

  # Step 3: unit tests
  echo "[run_iter] ctest"
  if ! ctest --test-dir "$BUILD_DIR" --output-on-failure -j >>"$ITER_DIR/test.log" 2>&1; then
    touch "$ITER_DIR/TEST_FAIL"
    exit 4
  fi
fi

# Step 4: render the 30-frame benchmark
echo "[run_iter] rendering 30 frames (backend=$BACKEND)"
"$LOCAL_PY" "$REPO_ROOT/scripts/render_30frame.py" \
  --backend "$BACKEND" \
  --cameras "$REPO_ROOT/benchmarks/cameras_v2.json" \
  --scene "$SCENE" \
  --out-dir "$ITER_DIR" \
  >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/RENDER_FAIL"; exit 3; }

# Step 5: compute metrics (auto-discovers views by filename match)
PREV_BEST="$(jq -r '[.[]?] | map(select(.action=="commit")) | min_by(.sum_total_ms).sum_total_ms // "Infinity"' \
              < <(cat "$OPT_DIR/iters.jsonl" 2>/dev/null | jq -s '.') 2>/dev/null || echo "Infinity")"
"$LOCAL_PY" "$REPO_ROOT/scripts/compute_metrics.py" \
  --iter-dir "$ITER_DIR" \
  --reference-dir "$REF_DIR" \
  --class "$CLASS" \
  --prev-best-ms "$PREV_BEST" \
  >>"$ITER_DIR/run.log" 2>&1

# Step 8: rebuild report (validator + decide steps happen via separate supervisor calls)
[[ -f "$OPT_DIR/build_report.py" ]] && \
  "$LOCAL_PY" "$OPT_DIR/build_report.py" >>"$ITER_DIR/run.log" 2>&1 || true

echo "$ITER_DIR/metrics.json"
