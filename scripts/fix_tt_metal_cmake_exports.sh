#!/usr/bin/env bash
# Repair broken relative symlinks in tt-metal's cmake export tree (common after rsync).
set -euo pipefail
: "${TT_METAL_HOME:?set TT_METAL_HOME}"
BUILD="$TT_METAL_HOME/build"

fix_link() {
  local link="$1" target="$2"
  if [[ -L "$link" ]]; then
    rm -f "$link"
  fi
  ln -sf "$target" "$link"
  if [[ ! -e "$link" ]]; then
    echo "ERROR: still broken: $link -> $target" >&2
    return 1
  fi
  echo "fixed: $link -> $target"
}

UMD_EXPORT="device/CMakeFiles/Export/a66ff87e1dd248bcbf3c902600a65777/umdTargets.cmake"
fix_link "$BUILD/tt_metal/third_party/umd/umdTargets.cmake" "$UMD_EXPORT"

LOGGER_EXPORT="CMakeFiles/Export/deaccb5626f933bc02728d438d6dcc90/tt-logger-targets.cmake"
fix_link "$BUILD/_deps/tt-logger-build/cmake/tt-logger-targets.cmake" "../$LOGGER_EXPORT"

METALIUM_EXPORT="tt_metal/CMakeFiles/Export/253da97e5fbe2e6037715c1aa0ef0d9f/Metalium.cmake"
METALIUM_CONFIG="$BUILD/tt-metalium-config.cmake"
# Symlink at build/Metalium.cmake breaks _IMPORT_PREFIX (resolves to /localdev). Point config at real export.
if [[ -L "$BUILD/Metalium.cmake" ]]; then
  rm -f "$BUILD/Metalium.cmake"
  echo "removed broken Metalium.cmake symlink"
fi
if [[ -f "$METALIUM_CONFIG" ]]; then
  sed -i 's|include("${CMAKE_CURRENT_LIST_DIR}/Metalium.cmake")|include("${CMAKE_CURRENT_LIST_DIR}/tt_metal/CMakeFiles/Export/253da97e5fbe2e6037715c1aa0ef0d9f/Metalium.cmake")|' "$METALIUM_CONFIG"
fi

# Staging layout expected by Metalium.cmake (_IMPORT_PREFIX = build/tt_metal/)
STAGE="$BUILD/tt_metal"
STAGE_LIB="$STAGE/lib"
mkdir -p "$STAGE_LIB"
fix_link "$STAGE_LIB/libtt_metal.so" "../libtt_metal.so"
fix_link "$STAGE_LIB/libtt_stl.so" "../../tt_stl/libtt_stl.so"
fix_link "$STAGE_LIB/libtracy.so.0.10.0" "../../lib/libtracy.so.0.10.0"

STAGE_INC="$STAGE/include"
mkdir -p "$STAGE_INC"
fix_link "$STAGE_INC/tt-metalium" "../../../tt_metal/api/tt-metalium"
fix_link "$STAGE_INC/tt_stl" "../../../tt_stl/tt_stl"
fix_link "$STAGE_INC/hostdevcommon" "../../../tt_metal/hostdevcommon/api/hostdevcommon"
if [[ -d "$TT_METAL_HOME/tt_metal/third_party/metalium-thirdparty" ]]; then
  fix_link "$STAGE_INC/metalium-thirdparty" "../../../tt_metal/third_party/metalium-thirdparty"
fi

echo "tt-metal cmake exports ok under $BUILD"
