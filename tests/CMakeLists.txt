include(Catch)

include(FetchContent)
FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
  GIT_TAG        4e8810b
)
FetchContent_GetProperties(cnpy)
if(NOT cnpy_POPULATED)
  FetchContent_Populate(cnpy)
endif()

find_package(ZLIB REQUIRED)

add_library(cnpy STATIC ${cnpy_SOURCE_DIR}/cnpy.cpp)
target_include_directories(cnpy PUBLIC ${cnpy_SOURCE_DIR})
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

add_executable(gsplat_cpu_tests
  unit/test_thread_pool.cpp
  unit/test_project.cpp
  unit/test_tile_assign.cpp
  unit/test_sort.cpp
  unit/test_blend.cpp
  unit/test_microblock_cull.cpp
)
target_link_libraries(gsplat_cpu_tests PRIVATE gsplat_cpu Catch2::Catch2WithMain cnpy)
target_include_directories(gsplat_cpu_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(gsplat_cpu_tests PRIVATE
  GSPLAT_HERO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/../tests/fixtures/hero")

catch_discover_tests(gsplat_cpu_tests)
