# T1 blend backward device bench (#693, docs/t685/DESIGN.md).
# Standalone target; does NOT build or link render_clean.
#   export TT_METAL_HOME=/localdev/smarton/tt-metal
#   cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build render/train/build
#   TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME render/train/build/blend_bwd_bench <prep_dir>
cmake_minimum_required(VERSION 3.20)
project(blend_bwd_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/TtMetalInTree.cmake)

add_executable(blend_bwd_bench blend_bwd_bench.cpp)
target_compile_definitions(blend_bwd_bench PRIVATE
  "BWD_KERNEL_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/\"")
gsplat_link_tt_metal(blend_bwd_bench)

# T2 param pass bench (#697): t2_bench <in_dir> <out_dir> [n_cores] [reps] [grads]
add_executable(t2_bench t2_bench.cpp)
target_compile_definitions(t2_bench PRIVATE
  "T2_KERNEL_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/\"")
gsplat_link_tt_metal(t2_bench)
