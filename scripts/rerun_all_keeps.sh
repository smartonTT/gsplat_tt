#!/usr/bin/env bash
# Re-render every KEEP commit at the new viewer-default hero pose.
#
# Each commit uses its OWN scripts/render_fixed.py + backends/tt/backend.py to
# avoid forward/backward API mismatches (e.g. cull_pairs added later). The new
# benchmarks/cameras.json is the only "future" asset copied into each checkout
# (since older cameras.json has the back-of-head hero). image_size is forced
# to [1024,1024] in cameras.json so older render_fixed.py — which lacks an
# --image-size flag and reads dims from cameras.json — renders at 1024.
#
# Iter 0 (dbb856a) IS the first commit; its TT render becomes the new
# reference (stitch_hero.png). PSNR + diff×10 are computed on Mac
# post-hoc, not here.
#
# Runs on bh-14 only. Stops the dev viewer for the duration (~50 min wall),
# then iterates: checkout <commit> → swap-in new cameras.json (1024-sized) →
# rebuild → wipe JIT cache → run THAT commit's render_fixed.py → save shot.
#
# After all iters: checkout smarton/opt-stable, restore HEAD assets, rebuild,
# restart dev viewer.
#
# Outputs on bh-14:
#   /tmp/rerun_shots/iter-NNN-hero.png   (one per iter, including iter 0)
#   /tmp/rerun_progress.log                    (per-iter status + timing)
#   /tmp/rerun_failures.log                    (iters that failed)
set -uo pipefail

cd /proj_sw/user_dev/smarton/gsplat_tt
source venv/bin/activate
export TT_METAL_HOME="$PWD/backends/tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
export TT_METAL_LOGGER_LEVEL=warning
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache-rerun
mkdir -p "$TT_METAL_CACHE"
unset GSPLAT_TT_MAX_G_PER_TILE GSPLAT_TT_CULL_EPS GSPLAT_TT_BINARY GSPLAT_TT_KERNEL_PREFIX

PROGRESS=/tmp/rerun_progress.log
FAIL=/tmp/rerun_failures.log
SHOTS_TMP=/tmp/rerun_shots
: > "$PROGRESS"
: > "$FAIL"
mkdir -p "$SHOTS_TMP"

# Stash the new viewer-default cameras.json + reference + a known-good
# render_fixed.py at /tmp/rerun_assets (the supervisor populated this in a
# prior step). cameras.json already has image_size=[1024,1024] for every
# scene (set by scripts/derive_all_views.py), so older render_fixed.py
# versions — which lack --image-size and read dims from cameras.json —
# render at 1024×1024 without any extra patching.
mkdir -p /tmp/rerun_assets
if [ ! -f /tmp/rerun_assets/cameras.json ]; then
  cp benchmarks/cameras.json                          /tmp/rerun_assets/cameras.json
  cp benchmarks/reference/stitch_hero.png   /tmp/rerun_assets/stitch_hero.png
  cp scripts/render_fixed.py                          /tmp/rerun_assets/render_fixed.py
fi

restore_assets() {
  # Copy the all-1024 cameras.json into the checkout so render_fixed.py
  # (every commit's own version) reads the new viewer-default hero AND
  # renders at 1024×1024 (no --image-size flag needed).
  cp /tmp/rerun_assets/cameras.json benchmarks/cameras.json
}

restore_head_assets() {
  cp /tmp/rerun_assets/cameras.json                       benchmarks/cameras.json
  cp /tmp/rerun_assets/stitch_hero.png          benchmarks/reference/stitch_hero.png
  cp /tmp/rerun_assets/render_fixed.py                    scripts/render_fixed.py
}

stop_dev_viewer() {
  pkill -TERM -f "scenes/stitch_doll" 2>/dev/null || true
  pkill -TERM -f "metal_example_gaussian" 2>/dev/null || true
  sleep 8
}

# Iter → commit hash map (must match build_report.py EXPERIMENTS KEEP rows).
# Iter 0 is first; its rendered output also becomes the new TT reference.
ITERS=(
  "0:dbb856a"
  "17:01b321c"
  "19:144ca57"
  "24:7a6ca88"
  "26:38e7510"
  "29:f60ca8f"
  "30:167b7a3"
  "32:22599a0"
  "34:69d1857"
  "35:955df1a"
  "37:8a3296d"
  "38:d006fab"
  "39:d006fab"
  "40:d006fab"
  "41:d006fab"
  "44:9af53a3"
  "58:e9ec4df"
  "59:61b3fa2"
  "60:61b3fa2"
  "61:68a55ba"
  "64:3d2f6d0"
  "66:6d2c847"
  "68:83ba871"
  "69:5e691a9"
  "71:e0f1678"
)

echo "[rerun] stopping dev viewer ..." | tee -a "$PROGRESS"
stop_dev_viewer

START_ALL=$(date +%s)
LAST_HASH=""
SUCCEEDED=()

for entry in "${ITERS[@]}"; do
  iter="${entry%%:*}"
  hash="${entry##*:}"
  printf -v ZITER "%03d" "$iter"
  OUT="$SHOTS_TMP/iter-${ZITER}-hero.png"
  printf '\n=== iter %s (%s) ===\n' "$iter" "$hash" | tee -a "$PROGRESS"
  start=$(date +%s)

  if [ "$hash" != "$LAST_HASH" ]; then
    git checkout -f "$hash" 2>&1 | tail -3 | tee -a "$PROGRESS"
    restore_assets
    echo "[rerun] building $hash ..." | tee -a "$PROGRESS"
    if ! sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting 2>&1 | tail -3 | tee -a "$PROGRESS"; then
      echo "iter $iter ($hash) BUILD FAILED" | tee -a "$FAIL"
      LAST_HASH="$hash"
      continue
    fi
    rm -rf "$TT_METAL_CACHE"/*
    LAST_HASH="$hash"
  else
    restore_assets
    echo "[rerun] reusing build from prior bundled commit" | tee -a "$PROGRESS"
  fi

  # Run THIS commit's render_fixed.py — older versions don't have
  # --image-size; cameras.json has image_size=[1024,1024] baked in.
  if ! timeout 180 python scripts/render_fixed.py stitch hero \
        --backend tt \
        --warmup 3 --frames 8 \
        --out "$OUT" --json > /tmp/iter_${iter}.json 2>&1; then
    echo "iter $iter ($hash) RENDER FAILED" | tee -a "$FAIL"
    tail -20 /tmp/iter_${iter}.json | tee -a "$FAIL"
    continue
  fi

  if [ ! -s "$OUT" ]; then
    echo "iter $iter ($hash) PRODUCED NO OUTPUT" | tee -a "$FAIL"
    continue
  fi

  SUCCEEDED+=("$iter")
  end=$(date +%s)
  echo "[rerun] iter $iter saved to $OUT in $((end - start))s" | tee -a "$PROGRESS"
done

# Final restore.
echo "[rerun] restoring HEAD (smarton/opt-stable) ..." | tee -a "$PROGRESS"
git checkout -f smarton/opt-stable 2>&1 | tail -3 | tee -a "$PROGRESS"
restore_head_assets
sudo ninja -C backends/tt/tt-metal/build metal_example_gaussian_splatting 2>&1 | tail -3 | tee -a "$PROGRESS"
rm -rf "$TT_METAL_CACHE"/*

# Use iter 0's render as the new TT reference (only if it succeeded).
# Note: this script only re-renders stitch HERO; sub-task A.3 handles the
# other 8 references (luigi/strawberry × hero/side/top + stitch side/top)
# in a separate step that also runs against iter 0 (dbb856a).
if [ -s "$SHOTS_TMP/iter-000-hero.png" ]; then
  cp "$SHOTS_TMP/iter-000-hero.png" benchmarks/reference/stitch_hero.png
  echo "[rerun] new TT stitch_hero reference (iter 0) copied to benchmarks/reference/" | tee -a "$PROGRESS"
fi

echo "[rerun] restarting dev viewer ..." | tee -a "$PROGRESS"
bash /tmp/start_viewer.sh 2>&1 | tail -3 | tee -a "$PROGRESS"

END_ALL=$(date +%s)
echo "[rerun] total wall: $((END_ALL - START_ALL))s" | tee -a "$PROGRESS"
echo "[rerun] succeeded iters: ${SUCCEEDED[*]}" | tee -a "$PROGRESS"
echo "[rerun] failures: $(wc -l < "$FAIL") iters" | tee -a "$PROGRESS"
echo "[rerun] shots in $SHOTS_TMP/"
ls "$SHOTS_TMP"/iter-*-hero.png 2>/dev/null | wc -l | xargs -I{} echo "[rerun]   hero shots: {}" | tee -a "$PROGRESS"
