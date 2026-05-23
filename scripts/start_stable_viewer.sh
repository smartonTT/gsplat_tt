#!/usr/bin/env bash
# Stable viewer — always runs last KEEP binary.
# Intended for bh-30 (Austin) so bh-14 (Toronto) dev cycle never disrupts it.
# Source: run this from the gsplat_tt repo root.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STABLE_BIN_DIR="/proj_sw/user_dev/smarton/stable_viewer"
STABLE_BINARY="${STABLE_BIN_DIR}/metal_example_gaussian_splatting_iter068"
STABLE_KERNELS_DIR="${STABLE_BIN_DIR}/kernels"
PORT="${GSPLAT_STABLE_PORT:-8080}"
SCENE="${1:-scenes/stitch_doll.ply}"

if [[ ! -x "$STABLE_BINARY" ]]; then
  echo "ERROR: stable binary not found at $STABLE_BINARY" >&2
  exit 1
fi

cd "$REPO_ROOT"
source venv/bin/activate

export TT_METAL_HOME="$REPO_ROOT/backends/tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export TT_MESH_GRAPH_DESC_PATH="$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto"
export TT_METAL_LOGGER_LEVEL=warning
export TT_METAL_CACHE="/localdev/smarton/.cache/tt-metal-cache-stable"
export GSPLAT_TT_BINARY="$STABLE_BINARY"
# Point JIT kernel sources at stable snapshot (outside gsplat_tt/ tree, so
# devsync/iter updates on bh-14 never reach here and break the stable viewer).
export GSPLAT_TT_KERNEL_PREFIX="${STABLE_BIN_DIR}/k/"

mkdir -p /localdev/smarton/.cache/tt-metal-cache-stable
mkdir -p /localdev/smarton/viewer_logs

# Kill any old stable viewer instance
pkill -TERM -f "stable_viewer" 2>/dev/null || true
pkill -TERM -f "gsplat.*stitch_doll.*port.*${PORT}" 2>/dev/null || true
sleep 5

LOG_FILE="/localdev/smarton/viewer_logs/stable_viewer_$(date +%Y%m%d-%H%M%S).log"
nohup python -m gsplat "$SCENE" --backend tt --port "$PORT" --force-square 1024 --verbose \
  > "$LOG_FILE" 2>&1 &
echo "Stable viewer PID: $! — log: $LOG_FILE"
