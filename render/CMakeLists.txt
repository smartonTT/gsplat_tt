# Clean extract of the production Tenstorrent gsplat render pipeline.
#
# Standalone CMake project (does NOT touch the existing src/ build). Builds a
# single pybind module `render_clean` from render/host/*.cpp, links the in-tree
# tt-metal, and points OVERRIDE_KERNEL_PREFIX at render/kernels/ so the device
# JIT compiles the clean kernels.
#
# Configure + build (on the device, after scripts/sync_to_bh30.sh):
#   export TT_METAL_HOME=/localdev/smarton/tt-metal
#   cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release
#   cmake --build render/build-tt -j 16
#
# Output: render/render_clean.cpython-3xx-*.so (imported by render/run.py).

cmake_minimum_required(VERSION 3.20)
project(render_clean LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_compile_options(
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-DNDEBUG>
  -fno-omit-frame-pointer
)

# ── pybind11 (resolved from the project venv, same as src/CMakeLists.txt) ─────
set(PYBIND11_FINDPYTHON ON)
set(Python_EXECUTABLE "${REPO_ROOT}/.venv/bin/python")
execute_process(
  COMMAND "${REPO_ROOT}/.venv/bin/python" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# ── gsplat_cpu host helpers (ThreadPool + the shared result structs) ──────────
# The TT drivers depend on gsplat_cpu only for the ThreadPool used by the host
# bridges and a handful of POD result structs. We compile the same sources the
# original build uses so the (dead, never-executed on the resident path) CPU
# fallback references still link. Built with -ffp-contract=off + AVX2 to mirror
# the device build exactly.
add_library(render_gsplat_cpu STATIC
  ${REPO_ROOT}/src/gsplat_cpu/thread_pool.cpp
  ${REPO_ROOT}/src/gsplat_cpu/project.cpp
  ${REPO_ROOT}/src/gsplat_cpu/tile_assign.cpp
  ${REPO_ROOT}/src/gsplat_cpu/sort.cpp
  ${REPO_ROOT}/src/gsplat_cpu/blend.cpp
  ${REPO_ROOT}/src/gsplat_cpu/microblock_cull.cpp
  ${REPO_ROOT}/src/gsplat_cpu/blend_microblock.cpp
  ${REPO_ROOT}/src/gsplat_cpu/cull_and_blend.cpp
)
target_include_directories(render_gsplat_cpu PUBLIC ${REPO_ROOT}/src)
target_compile_definitions(render_gsplat_cpu PRIVATE GSPLAT_SIMD_AVX2)
target_compile_options(render_gsplat_cpu PRIVATE
  -ffp-contract=off -O3 -march=native -mavx2 -mfma)
set_target_properties(render_gsplat_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ── TT device drivers + kernels ───────────────────────────────────────────────
include(${REPO_ROOT}/cmake/TtMetalInTree.cmake)

add_library(render_tt STATIC
  host/project_device.cpp
  host/pfwc_device.cpp
  host/gather_visible_device.cpp
  host/tile_assign_device.cpp
  host/sort_device.cpp
  host/blend_device.cpp
  host/device_state.cpp
  host/jit_warmup.cpp
)
set_target_properties(render_tt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(render_tt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/host
  ${REPO_ROOT}/src)
target_compile_definitions(render_tt PRIVATE
  GSPLAT_WITH_TT
  "OVERRIDE_KERNEL_PREFIX=\"${CMAKE_CURRENT_SOURCE_DIR}/\"")
gsplat_link_tt_metal(render_tt)
target_link_libraries(render_tt PUBLIC render_gsplat_cpu)

# ── pybind module ─────────────────────────────────────────────────────────────
pybind11_add_module(render_clean MODULE host/render.cpp)
target_include_directories(render_clean PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/host
  ${REPO_ROOT}/src)
target_compile_definitions(render_clean PRIVATE GSPLAT_WITH_TT)
target_link_libraries(render_clean PRIVATE render_tt render_gsplat_cpu)
set_target_properties(render_clean PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
