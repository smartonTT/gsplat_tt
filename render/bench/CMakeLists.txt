# T-A hardware-ceiling microbenchmark suite (docs/OPTIMIZATION-PLAN.md, Task 1).
#
# Standalone target; does NOT build or link render_clean.
#   export TT_METAL_HOME=/localdev/smarton/tt-metal
#   cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build render/bench/build
#   TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME render/bench/build/risc_microbench
cmake_minimum_required(VERSION 3.20)
project(risc_microbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/TtMetalInTree.cmake)

add_executable(risc_microbench risc_microbench.cpp)
target_compile_definitions(risc_microbench PRIVATE
  "MB_KERNEL_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/\"")
gsplat_link_tt_metal(risc_microbench)
