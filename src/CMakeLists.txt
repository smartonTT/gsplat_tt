# gsplat_cpu — C++ implementation of the gsplat rendering pipeline.
#
# Phase 0: skeleton only. iter-001 populates libgsplat_cpu (thread pool +
# scaffolding); iter-002 adds project; iter-003 adds tile_assign + sort;
# iter-004 adds alpha_blend.
#
# Build:  cmake -G Ninja -S src -B build -DCMAKE_BUILD_TYPE=Release
#         cmake --build build -j
#         ctest --test-dir build --output-on-failure -j

cmake_minimum_required(VERSION 3.20)
project(gsplat_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(
  -Wall -Wextra -Wpedantic
  -fno-omit-frame-pointer
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Release>:-march=native>
  $<$<CONFIG:Release>:-DNDEBUG>
  $<$<CONFIG:Debug>:-O0>
  $<$<CONFIG:Debug>:-g>
)

include(FetchContent)

# Catch2 v3 for unit tests (header-only-ish; build the static lib).
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.10.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

# pybind11 — resolved via the project venv's pybind11 module.
# iter-001 will activate this once the C++ library has anything to bind.
# find_package(pybind11 CONFIG REQUIRED)

# Library target — placeholder. iter-001 fleshes this out.
add_library(gsplat_cpu STATIC
  gsplat_cpu/dummy.cpp
)
target_include_directories(gsplat_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Tests
enable_testing()
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests ${CMAKE_BINARY_DIR}/tests)
