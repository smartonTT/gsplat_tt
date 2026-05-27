// Phase 0 sanity test: catch2 framework wires up, CTest discovers it.
// iter-001 deletes this and adds real tests for the thread pool.
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Phase 0 sanity: arithmetic", "[smoke]") {
  REQUIRE(1 + 1 == 2);
}
