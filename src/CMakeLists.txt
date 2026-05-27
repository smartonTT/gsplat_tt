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
set(PYBIND11_FINDPYTHON ON)
set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/../.venv/bin/python")
execute_process(
  COMMAND "${CMAKE_SOURCE_DIR}/../.venv/bin/python" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT PYBIND11_CMAKE_DIR)
  # Fallback: brew pybind11
  execute_process(
    COMMAND pybind11-config --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# Library target — thread pool (iter-001).
add_library(gsplat_cpu STATIC
  gsplat_cpu/thread_pool.cpp
  gsplat_cpu/project.cpp
  gsplat_cpu/tile_assign.cpp
  gsplat_cpu/sort.cpp
  gsplat_cpu/blend.cpp
)
target_include_directories(gsplat_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gsplat_cpu PRIVATE -ffp-contract=off)
if(APPLE)
  find_library(GSPLAT_ACCELERATE Accelerate REQUIRED)
  target_link_libraries(gsplat_cpu PUBLIC ${GSPLAT_ACCELERATE})
endif()
set_target_properties(gsplat_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# pybind11 extension — lives next to backends/cpu_cpp/backend.py
pybind11_add_module(_gsplat_cpu MODULE
  ${CMAKE_SOURCE_DIR}/../backends/cpu_cpp/pybind_module.cpp
)
target_link_libraries(_gsplat_cpu PRIVATE gsplat_cpu)
set_target_properties(_gsplat_cpu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../backends/cpu_cpp
)

# Tests. include(CTest) (rather than bare enable_testing()) writes
# CTestTestfile.cmake at the cmake binary root, so `ctest --test-dir build`
# at the top-level build dir finds tests even though tests/ lives outside
# the src/ source tree.
include(CTest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests ${CMAKE_BINARY_DIR}/tests)
