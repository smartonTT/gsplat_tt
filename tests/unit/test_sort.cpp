#include <catch2/catch_test_macros.hpp>

#include "gsplat_cpu/sort.h"

TEST_CASE("sort: same tile sorted by depth front-to-back", "[sort]") {
    const int64_t gaussian_ids[] = {0, 1, 2};
    const int64_t tile_ids[] = {5, 5, 5};
    const float depths[] = {0.5f, 0.3f, 0.7f};

    const gsplat_cpu::SortResult result =
        gsplat_cpu::sort_and_bin(gaussian_ids, tile_ids, depths, 3, 3, 8, 8);

    REQUIRE(result.sorted_gaussian_ids.size() == 3);
    REQUIRE(result.sorted_gaussian_ids[0] == 1);
    REQUIRE(result.sorted_gaussian_ids[1] == 0);
    REQUIRE(result.sorted_gaussian_ids[2] == 2);
}

TEST_CASE("sort: tile_ranges for three populated tiles", "[sort]") {
    const int64_t gaussian_ids[] = {0, 1, 2, 3, 4};
    const int64_t tile_ids[] = {2, 2, 7, 7, 10};
    const float depths[] = {0.4f, 0.6f, 0.5f, 0.8f, 0.3f, 0.9f};

    const gsplat_cpu::SortResult result =
        gsplat_cpu::sort_and_bin(gaussian_ids, tile_ids, depths, 5, 3, 4, 3);

    REQUIRE(result.tile_ranges[2 * 2 + 0] == 0);
    REQUIRE(result.tile_ranges[2 * 2 + 1] == 2);
    REQUIRE(result.tile_ranges[7 * 2 + 0] == 2);
    REQUIRE(result.tile_ranges[7 * 2 + 1] == 4);
    REQUIRE(result.tile_ranges[10 * 2 + 0] == 4);
    REQUIRE(result.tile_ranges[10 * 2 + 1] == 5);
}

TEST_CASE("sort: empty tiles keep zero ranges", "[sort]") {
    const int64_t gaussian_ids[] = {0};
    const int64_t tile_ids[] = {3};
    const float depths[] = {1.0f};

    const gsplat_cpu::SortResult result =
        gsplat_cpu::sort_and_bin(gaussian_ids, tile_ids, depths, 1, 1, 4, 4);

    REQUIRE(result.tile_ranges[0] == 0);
    REQUIRE(result.tile_ranges[1] == 0);
    REQUIRE(result.tile_ranges[3 * 2 + 0] == 0);
    REQUIRE(result.tile_ranges[3 * 2 + 1] == 1);
    REQUIRE(result.tile_ranges[5 * 2 + 0] == 0);
    REQUIRE(result.tile_ranges[5 * 2 + 1] == 0);
}

TEST_CASE("sort: same tile_id orders by depth not gaussian_id", "[sort]") {
    const int64_t gaussian_ids[] = {5, 1};
    const int64_t tile_ids[] = {4, 4};
    const float depths[] = {0.2f, 0.9f, 0.4f, 0.6f, 0.1f, 0.8f};

    const gsplat_cpu::SortResult result =
        gsplat_cpu::sort_and_bin(gaussian_ids, tile_ids, depths, 2, 2, 4, 4);

    REQUIRE(result.sorted_gaussian_ids.size() == 2);
    REQUIRE(result.sorted_gaussian_ids[0] == 5);
    REQUIRE(result.sorted_gaussian_ids[1] == 1);
}

TEST_CASE("sort: empty input returns empty sorted ids and zero ranges", "[sort]") {
    const float depths[] = {1.0f};

    const gsplat_cpu::SortResult result =
        gsplat_cpu::sort_and_bin(nullptr, nullptr, depths, 0, 1, 2, 2);

    REQUIRE(result.sorted_gaussian_ids.empty());
    REQUIRE(result.tile_ranges.size() == 8);
    for (const int64_t v : result.tile_ranges) {
        REQUIRE(v == 0);
    }
}
