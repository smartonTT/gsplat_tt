#!/usr/bin/env bash
# Metal-port iter driver (run on bh-30 or any TT box with device).
#
# Usage:
#   scripts/run_iter_metal.sh <iter_num> <slug> <class> [--backend tt]
#   REMOTE_HOST=bh-30 scripts/run_iter_metal.sh ...   # SSH wrapper (Mac supervisor)
#
# Steps:
#   1. verify tt-metal binary exists (build if missing)
#   2. Layer 2: verify_blend_metal.py on hero fixture
#   3. render 30-frame benchmark (--backend tt or tt_blend_only when wired)
#   4. compute_metrics vs benchmarks/reference_v2
#   5. append opt/metal-iters.jsonl (via decide_and_log_metal.py when present)
#
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-}"
if [[ -n "$REMOTE_HOST" && -z "${RUN_ITER_METAL_REMOTE:-}" ]]; then
  REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
  exec ssh "$REMOTE_HOST" \
    "cd /proj_sw/user_dev/smarton/gstt2 && RUN_ITER_METAL_REMOTE=1 REMOTE_HOST= bash scripts/run_iter_metal.sh $(printf '%q ' "$@")"
fi

source "$(dirname "$0")/_env.sh"

ITER_NUM="${1:?usage: $0 <iter_num> <slug> <class> [--backend ...]}"
SLUG="${2:?slug required}"
CLASS="${3:?class required}"
BACKEND="tt"
if [[ "${4:-}" == "--backend" ]]; then BACKEND="${5:-tt}"; fi

SCENE="${SCENE:-bicycle}"
ITER_NAME="$(printf "metal-iter-%03d-%s" "$ITER_NUM" "$SLUG")"
ITER_DIR="$REPO_ROOT/opt/metal-screenshots/$ITER_NAME"
mkdir -p "$ITER_DIR"

# Repair tt-metal symlink if devsync dropped a stub tree on the remote box.
if [[ -f "$REPO_ROOT/scripts/setup_bh30_metal.sh" ]]; then
  bash "$REPO_ROOT/scripts/setup_bh30_metal.sh" >>"$ITER_DIR/setup.log" 2>&1 || true
fi

TT_BIN="$REPO_ROOT/backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"

export TT_METAL_HOME="${TT_METAL_HOME:-$REPO_ROOT/backends/tt/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export MESH_DEVICE="${MESH_DEVICE:-P100}"
export TT_METAL_ARCH_NAME="${TT_METAL_ARCH_NAME:-blackhole}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/localdev/smarton/.cache/tt-metal-cache}"

if [[ ! -x "$TT_BIN" ]]; then
  echo "[run_iter_metal] building metal_example_gaussian_splatting" | tee -a "$ITER_DIR/build.log"
  cmake -G Ninja -S "$TT_METAL_HOME" -B "$TT_METAL_HOME/build" \
    -DCMAKE_BUILD_TYPE=Release >>"$ITER_DIR/build.log" 2>&1 || true
  ninja -C "$TT_METAL_HOME/build" metal_example_gaussian_splatting \
    >>"$ITER_DIR/build.log" 2>&1 || { touch "$ITER_DIR/BUILD_FAIL"; exit 2; }
fi

echo "[run_iter_metal] Layer 2: hero fixture blend verify" | tee -a "$ITER_DIR/run.log"
if "$LOCAL_PY" "$REPO_ROOT/scripts/verify_blend_metal.py" --backend "$BACKEND" \
    >>"$ITER_DIR/run.log" 2>&1; then
  echo "LAYER2_PASS" > "$ITER_DIR/layer2.ok"
else
  touch "$ITER_DIR/LAYER2_FAIL"
  exit 4
fi

echo "[run_iter_metal] Layer 3: 30-frame render (backend=$BACKEND scene=$SCENE)" \
  | tee -a "$ITER_DIR/run.log"
"$LOCAL_PY" "$REPO_ROOT/scripts/render_30frame.py" \
  --backend "$BACKEND" \
  --cameras "$REPO_ROOT/benchmarks/cameras_v2.json" \
  --scene "$SCENE" \
  --out-dir "$ITER_DIR" \
  >>"$ITER_DIR/run.log" 2>&1 || { touch "$ITER_DIR/RENDER_FAIL"; exit 3; }

PREV_BEST="$(jq -r '[.[]?] | map(select(.action=="commit")) | min_by(.sum_total_ms).sum_total_ms // "Infinity"' \
              < <(cat "$REPO_ROOT/opt/metal-iters.jsonl" 2>/dev/null | jq -s '.') 2>/dev/null || echo "Infinity")"
"$LOCAL_PY" "$REPO_ROOT/scripts/compute_metrics.py" \
  --iter-dir "$ITER_DIR" \
  --reference-dir "$REF_DIR" \
  --class "$CLASS" \
  --prev-best-ms "$PREV_BEST" \
  >>"$ITER_DIR/run.log" 2>&1

echo "$ITER_DIR/metrics.json"
