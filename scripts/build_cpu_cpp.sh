#!/usr/bin/env bash
# Build cpu_cpp pybind extension with common gstt2 options.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SCALAR="${GSPLAT_SCALAR_ONLY:-OFF}"
AVX2="${GSPLAT_SIMD_AVX2:-OFF}"
WITH_TT="${GSPLAT_WITH_TT:-OFF}"
BUILD_DIR="${BUILD_DIR:-build}"

OPTS=(-G Ninja -S src -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release)
if [[ "$SCALAR" == "ON" ]]; then
  OPTS+=(-DGSPLAT_SCALAR_ONLY=ON)
fi
if [[ "$AVX2" == "ON" ]]; then
  OPTS+=(-DGSPLAT_SIMD_AVX2=ON)
fi
if [[ "$WITH_TT" == "ON" ]]; then
  : "${TT_METAL_HOME:?set TT_METAL_HOME for GSPLAT_WITH_TT}"
  export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
  bash "$REPO/scripts/fix_tt_metal_cmake_exports.sh"
  OPTS+=(-DGSPLAT_WITH_TT=ON)
fi

cmake "${OPTS[@]}"
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
ctest --test-dir "$BUILD_DIR" --output-on-failure -j 8

echo "built: $REPO/backends/cpu_cpp/_gsplat_cpu*.so"
