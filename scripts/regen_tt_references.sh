#!/usr/bin/env bash
# Regenerate ALL 9 reference renders using TT backend on iter 0 (dbb856a).
#
# "Iter 0" is the first KEEP commit in the optimization log — it's the cleanest
# pre-optimization TT-correct rendering. By using dbb856a's OWN render_fixed.py
# we avoid forward/backward API mismatches (e.g. cull_pairs added later).
#
# Inputs (must already exist; populated in earlier supervisor steps):
#   /tmp/rerun_assets/cameras.json    — all scenes at image_size=[1024,1024],
#                                        viewer-default hero, full-fit side/top
#
# Outputs (on bh-14):
#   /tmp/tt_refs/<scene>_<view>.png   — staging (all 1024×1024, all 9 combos)
#   benchmarks/reference/<scene>_<view>.png        — final references
#
# Side effects:
#   - checks out dbb856a (resets tracked files to that commit)
#   - rebuilds metal_example_gaussian_splatting at dbb856a
#   - wipes /localdev/smarton/.cache/tt-metal-cache-regen
#   - at the end: checks out smarton/opt-stable, rebuilds, restarts dev viewer
#
# Caller MUST `devsync pause yyzo-bh-14` before running and
# `devsync resume yyzo-bh-14` after, so the dbb856a checkout doesn't propagate
# back to the Mac working tree.
set -uo pipefail

cd /proj_sw/user_dev/smarton/gsplat_tt
source venv/bin/activate
export TT_METAL_HOME="$PWD/backends/tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
export TT_METAL_LOGGER_LEVEL=warning
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache-regen
mkdir -p "$TT_METAL_CACHE"
unset GSPLAT_TT_MAX_G_PER_TILE GSPLAT_TT_CULL_EPS GSPLAT_TT_BINARY GSPLAT_TT_KERNEL_PREFIX

PROGRESS=/tmp/regen_refs_progress.log
OUT_DIR=/tmp/tt_refs
: > "$PROGRESS"
mkdir -p "$OUT_DIR"

if [ ! -f /tmp/rerun_assets/cameras.json ]; then
  echo "FATAL: /tmp/rerun_assets/cameras.json missing — populate first" | tee -a "$PROGRESS"
  exit 1
fi

stop_dev_viewer() {
  pkill -TERM -f "scenes/stitch_doll" 2>/dev/null || true
  pkill -TERM -f "metal_example_gaussian" 2>/dev/null || true
  sleep 8
}

echo "[regen] stopping dev viewer ..." | tee -a "$PROGRESS"
stop_dev_viewer

echo "[regen] checking out iter 0 (dbb856a) ..." | tee -a "$PROGRESS"
git checkout -f dbb856a 2>&1 | tail -3 | tee -a "$PROGRESS"

echo "[regen] copying new 1024 cameras.json into checkout ..." | tee -a "$PROGRESS"
cp /tmp/rerun_assets/cameras.json benchmarks/cameras.json

echo "[regen] building iter 0 ..." | tee -a "$PROGRESS"
if ! sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting 2>&1 | tail -5 | tee -a "$PROGRESS"; then
  echo "FATAL: build failed at iter 0" | tee -a "$PROGRESS"
  exit 1
fi
rm -rf "$TT_METAL_CACHE"/*

START=$(date +%s)
SCENES_VIEWS=(
  "stitch hero" "stitch side" "stitch top"
  "luigi hero"  "luigi side"  "luigi top"
  "strawberry hero" "strawberry side" "strawberry top"
)

FAILED=()
for sv in "${SCENES_VIEWS[@]}"; do
  scene="${sv%% *}"
  view="${sv##* }"
  out="$OUT_DIR/${scene}_${view}.png"
  echo "[regen] rendering ${scene}/${view} -> $out" | tee -a "$PROGRESS"
  if ! timeout 240 python scripts/render_fixed.py "$scene" "$view" \
        --backend tt --warmup 2 --frames 4 \
        --out "$out" > "/tmp/regen_${scene}_${view}.log" 2>&1; then
    echo "  FAILED — see /tmp/regen_${scene}_${view}.log" | tee -a "$PROGRESS"
    tail -15 "/tmp/regen_${scene}_${view}.log" | tee -a "$PROGRESS"
    FAILED+=("${scene}/${view}")
    continue
  fi
  if [ ! -s "$out" ]; then
    echo "  FAILED — empty output" | tee -a "$PROGRESS"
    FAILED+=("${scene}/${view}")
    continue
  fi
  echo "  ok ($(stat -c%s "$out") bytes)" | tee -a "$PROGRESS"
done
END=$(date +%s)
echo "[regen] all renders done in $((END - START))s" | tee -a "$PROGRESS"

# Switch back to opt-stable BEFORE copying renders into benchmarks/reference/.
# The references are tracked files; copying into the dbb856a worktree then
# stashing across a branch switch is conflict-prone if opt-stable and dbb856a
# have different reference content. Staging in /tmp/tt_refs/ avoids that.
echo "[regen] checking out smarton/opt-stable ..." | tee -a "$PROGRESS"
git checkout -f smarton/opt-stable 2>&1 | tail -3 | tee -a "$PROGRESS"

echo "[regen] restoring 1024 cameras.json ..." | tee -a "$PROGRESS"
cp /tmp/rerun_assets/cameras.json benchmarks/cameras.json

echo "[regen] copying renders from $OUT_DIR/ into benchmarks/reference/ ..." | tee -a "$PROGRESS"
for sv in "${SCENES_VIEWS[@]}"; do
  scene="${sv%% *}"
  view="${sv##* }"
  src="$OUT_DIR/${scene}_${view}.png"
  if [ ! -s "$src" ]; then
    echo "  [skip] $src missing/empty" | tee -a "$PROGRESS"
    continue
  fi
  cp "$src" "benchmarks/reference/${scene}_${view}.png"
done

echo "[regen] rebuilding opt-stable ..." | tee -a "$PROGRESS"
sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting 2>&1 | tail -3 | tee -a "$PROGRESS"
rm -rf "$TT_METAL_CACHE"/*

echo "[regen] restarting dev viewer ..." | tee -a "$PROGRESS"
bash /tmp/start_viewer.sh 2>&1 | tail -5 | tee -a "$PROGRESS"

echo "[regen] done." | tee -a "$PROGRESS"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[regen] FAILED: ${FAILED[*]}" | tee -a "$PROGRESS"
  exit 2
fi
