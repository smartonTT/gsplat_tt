#!/usr/bin/env bash
# One-time bh-30 setup for gsplat_tt_2 metal port.
# Run ON bh-30 after devsync: bash scripts/setup_bh30_metal.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TT_SRC="/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal"
TT_DST="$REPO/backends/tt/tt-metal"

if [[ ! -d "$TT_SRC" ]]; then
  echo "ERROR: tt-metal not found at $TT_SRC" >&2
  exit 1
fi

mkdir -p "$REPO/backends/tt"
if [[ -d "$TT_DST" && ! -L "$TT_DST" ]]; then
  rm -rf "$TT_DST"
  echo "removed partial tt-metal tree (git-tracked stub)"
fi
if [[ ! -L "$TT_DST" ]]; then
  ln -sf "$TT_SRC" "$TT_DST"
  echo "linked tt-metal -> $TT_SRC"
fi

# Mac-absolute symlink from devsync; use stitch on box until bicycle PLY is copied.
if [[ -L scenes/point_cloud.ply ]] && ! test -f scenes/point_cloud.ply; then
  rm -f scenes/point_cloud.ply
  ln -sf stitch_doll.ply scenes/point_cloud.ply
  echo "WARNING: bicycle PLY not on box; scenes/point_cloud.ply -> stitch_doll.ply placeholder"
fi

export TT_METAL_HOME="$TT_DST"
export TT_METAL_RUNTIME_ROOT="$TT_DST"
export MESH_DEVICE="${MESH_DEVICE:-P100}"
export TT_METAL_ARCH_NAME="${TT_METAL_ARCH_NAME:-blackhole}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-/localdev/smarton/.cache/tt-metal-cache}"
mkdir -p "$TT_METAL_CACHE"

VENV_PY="$REPO/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "recreating broken .venv with system python3..."
  rm -rf "$REPO/.venv"
  python3 -m venv "$REPO/.venv"
  "$REPO/.venv/bin/pip" install -q --upgrade pip
  "$REPO/.venv/bin/pip" install -q numpy torch pillow
else
  "$REPO/.venv/bin/pip" install -q numpy torch pillow 2>/dev/null || true
fi

BIN="$TT_DST/build/programming_examples/metal_example_gaussian_splatting"
if [[ ! -x "$BIN" ]]; then
  echo "building metal_example_gaussian_splatting..."
  cmake -G Ninja -S "$TT_DST" -B "$TT_DST/build" -DCMAKE_BUILD_TYPE=Release
  ninja -C "$TT_DST/build" metal_example_gaussian_splatting
fi

echo "setup ok: $BIN"
