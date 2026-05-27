#include <catch2/catch_test_macros.hpp>

#include "gsplat_cpu/tile_assign.h"

namespace {

int64_t sum_tpg(const gsplat_cpu::TileAssignResult& result) {
    int64_t s = 0;
    for (const int64_t v : result.tiles_per_gaussian) {
        s += v;
    }
    return s;
}

}  // namespace

TEST_CASE("tile_assign: single Gaussian at tile center gets one tile", "[tile_assign]") {
    const float means_2d[] = {48.0f, 48.0f};
    const float radii[] = {8.0f, 8.0f};

    const gsplat_cpu::TileAssignResult result =
        gsplat_cpu::tile_assign(means_2d, radii, 1, 256, 256, 32, nullptr, nullptr, 15.0f / 255.0f);

    REQUIRE(result.gaussian_ids.size() == 1);
    REQUIRE(result.tile_ids.size() == 1);
    REQUIRE(result.tiles_per_gaussian.size() == 1);
    REQUIRE(result.gaussian_ids[0] == 0);
    REQUIRE(result.tile_ids[0] == 1 * 8 + 1);
    REQUIRE(result.tiles_per_gaussian[0] == 1);
}

TEST_CASE("tile_assign: Gaussian straddling four tiles gets four pairs", "[tile_assign]") {
    const float means_2d[] = {32.0f, 32.0f};
    const float radii[] = {16.0f, 16.0f};

    const gsplat_cpu::TileAssignResult result =
        gsplat_cpu::tile_assign(means_2d, radii, 1, 256, 256, 32, nullptr, nullptr, 15.0f / 255.0f);

    REQUIRE(result.gaussian_ids.size() == 4);
    REQUIRE(result.tile_ids.size() == 4);
    REQUIRE(result.tiles_per_gaussian[0] == 4);
    REQUIRE(sum_tpg(result) == static_cast<int64_t>(result.gaussian_ids.size()));
}

TEST_CASE("tile_assign: Mahalanobis cull drops low-contribution pair", "[tile_assign]") {
    const float means_2d[] = {16.0f, 16.0f, 240.0f, 240.0f};
    const float radii[] = {20.0f, 20.0f, 20.0f, 20.0f};
    const float covs_2d[] = {
        0.01f, 0.0f, 0.0f, 0.01f,
        0.01f, 0.0f, 0.0f, 0.01f,
    };
    const float opacities[] = {0.5f, 0.5f};

    const gsplat_cpu::TileAssignResult no_cull =
        gsplat_cpu::tile_assign(means_2d, radii, 2, 256, 256, 32, nullptr, nullptr, 15.0f / 255.0f);
    const gsplat_cpu::TileAssignResult culled = gsplat_cpu::tile_assign(
        means_2d, radii, 2, 256, 256, 32, covs_2d, opacities, 15.0f / 255.0f);

    REQUIRE(culled.gaussian_ids.size() < no_cull.gaussian_ids.size());
    REQUIRE(sum_tpg(culled) == static_cast<int64_t>(culled.gaussian_ids.size()));
}

TEST_CASE("tile_assign: tiles_per_gaussian sum equals pair count", "[tile_assign]") {
    const float means_2d[] = {100.0f, 50.0f, 200.0f, 150.0f, 80.0f, 120.0f};
    const float radii[] = {24.0f, 18.0f, 30.0f, 22.0f, 12.0f, 16.0f, 28.0f, 20.0f, 10.0f, 14.0f};
    const float covs_2d[] = {
        1.0f, 0.0f, 0.0f, 1.0f,
        2.0f, 0.0f, 0.0f, 2.0f,
        0.5f, 0.0f, 0.0f, 0.5f,
    };
    const float opacities[] = {0.9f, 0.8f, 0.7f};

    const gsplat_cpu::TileAssignResult result = gsplat_cpu::tile_assign(
        means_2d, radii, 3, 256, 256, 32, covs_2d, opacities, 15.0f / 255.0f);

    REQUIRE(result.tiles_per_gaussian.size() == 3);
    REQUIRE(sum_tpg(result) == static_cast<int64_t>(result.gaussian_ids.size()));
}

TEST_CASE("tile_assign: empty input returns empty arrays", "[tile_assign]") {
    const gsplat_cpu::TileAssignResult result =
        gsplat_cpu::tile_assign(nullptr, nullptr, 0, 256, 256, 32, nullptr, nullptr, 15.0f / 255.0f);

    REQUIRE(result.gaussian_ids.empty());
    REQUIRE(result.tile_ids.empty());
    REQUIRE(result.tiles_per_gaussian.empty());
}
