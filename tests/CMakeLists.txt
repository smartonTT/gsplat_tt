include(Catch)

add_executable(gsplat_cpu_tests
  unit/test_thread_pool.cpp
  unit/test_project.cpp
  unit/test_tile_assign.cpp
  unit/test_sort.cpp
  unit/test_blend.cpp
)
target_link_libraries(gsplat_cpu_tests PRIVATE gsplat_cpu Catch2::Catch2WithMain)
target_include_directories(gsplat_cpu_tests PRIVATE ${CMAKE_SOURCE_DIR})

catch_discover_tests(gsplat_cpu_tests)
