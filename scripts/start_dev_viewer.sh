#!/usr/bin/env bash
# Dev viewer — runs the current opt-v2 build on bh-14, port 8080.
# Pair with start_stable_viewer.sh on bh-30 (port 8081) for side-by-side
# comparison of dev vs. last-promoted stable iter.
#
# Source: run this from the gsplat_tt repo root.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_BINARY="${REPO_ROOT}/backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"
PORT="${GSPLAT_DEV_PORT:-8080}"
SCENE="${1:-scenes/stitch_doll.ply}"

if [[ ! -x "$DEV_BINARY" ]]; then
  echo "ERROR: dev binary not found at $DEV_BINARY — run a build first" >&2
  exit 1
fi

cd "$REPO_ROOT"
source venv/bin/activate

export TT_METAL_HOME="$REPO_ROOT/backends/tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
export TT_METAL_LOGGER_LEVEL=warning
export GSPLAT_TT_BINARY="$DEV_BINARY"

mkdir -p /localdev/smarton/viewer_logs

# Kill any old dev viewer instance on this port (avoid self-match: do not
# pkill on a string that matches this script's own path).
pkill -TERM -f "gsplat.*--port ${PORT}" 2>/dev/null || true
sleep 5

LOG_FILE="/localdev/smarton/viewer_logs/dev_viewer_$(date +%Y%m%d-%H%M%S).log"
nohup python -m gsplat "$SCENE" --backend tt --port "$PORT" --force-square 1024 --verbose \
  > "$LOG_FILE" 2>&1 &
echo "Dev viewer PID: $! — log: $LOG_FILE"
