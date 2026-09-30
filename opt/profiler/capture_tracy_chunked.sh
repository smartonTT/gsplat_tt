#!/usr/bin/env bash
# Chunked 30-view Tracy device capture: one devrun job per view range, so each job
# stays under devrun's 600 s ceiling, then the chunk CSVs are stitched into one.
# Run from the Mac repo root (devrun.sh resolves the device host from ttw.toml):
#
#   bash opt/profiler/capture_tracy_chunked.sh <iter-dir> [chunk_views=10] [timeout_s=480]
#
# Box tree: $GSTT2_REPO (default /localdev/smarton/gstt2). Output on the box:
#   opt/profiler/<iter-dir>/chunks/<a>-<b>/{render.tracy,profile_log_device.csv}
#   opt/profiler/<iter-dir>/profile_log_device.csv   (stitched, warmups dropped)
# Each chunk re-runs one warmup hero render (JIT cache warm); the stitcher drops it.
set -euo pipefail
ITER_DIR="${1:?usage: capture_tracy_chunked.sh <iter-dir> [chunk_views] [timeout_s]}"
CHUNK="${2:-10}"
TIMEOUT="${3:-480}"
NVIEWS=30
REPO="${GSTT2_REPO:-/localdev/smarton/gstt2}"
DEVRUN="${DEVRUN:-$HOME/dev/tt-workflows/scripts/devrun.sh}"
HOST="$(awk -F'"' '/^device_host/{print $2}' ttw.toml)"

csvs=()
for ((a = 0; a < NVIEWS; a += CHUNK)); do
  b=$((a + CHUNK < NVIEWS ? a + CHUNK : NVIEWS))
  "$DEVRUN" --timeout "$TIMEOUT" --tag "tracy-${ITER_DIR}-${a}-${b}" -- \
    "export GSTT2_REPO=$REPO; bash $REPO/opt/profiler/capture_tracy.sh $ITER_DIR $a:$b 2>&1 \
     | grep -v '^\[run\]\|OVERFLOW-DIST\|SUBCHUNK\|^\[SORT\]' | tail -30"
  csvs+=("$REPO/opt/profiler/$ITER_DIR/chunks/$a-$b/profile_log_device.csv")
done

# Stitching is off-device (no devrun / device lock needed).
ssh "$HOST" "cd $REPO && source .venv/bin/activate && \
  python3 opt/profiler/stitch_device_csv.py -o opt/profiler/$ITER_DIR/profile_log_device.csv ${csvs[*]} && \
  python3 opt/profiler/zone_occupancy.py opt/profiler/$ITER_DIR/profile_log_device.csv"
