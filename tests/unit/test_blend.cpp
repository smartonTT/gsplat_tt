#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "gsplat_cpu/blend.h"
#include "gsplat_cpu/thread_pool.h"

using Catch::Matchers::WithinAbs;

namespace {

gsplat_cpu::ThreadPool& test_pool() {
    static gsplat_cpu::ThreadPool pool(1);
    return pool;
}

float pixel_at(const std::vector<float>& image, int H, int W, int y, int x, int c) {
    return image[(static_cast<std::size_t>(y) * static_cast<std::size_t>(W) +
                  static_cast<std::size_t>(x)) *
                     3 +
                 static_cast<std::size_t>(c)];
}

}  // namespace

TEST_CASE("blend: single tile single Gaussian red at center", "[blend]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;

    const float means_2d[] = {16.5f, 16.5f};
    const float covs_2d[] = {50.0f, 0.0f, 0.0f, 50.0f};
    const float colors[] = {1.0f, 0.0f, 0.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 1};

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, H, W,
        tile_size, test_pool());

    REQUIRE(result.image.size() == static_cast<std::size_t>(H * W * 3));
    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    const float g_center = pixel_at(result.image, H, W, 16, 16, 1);
    const float b_center = pixel_at(result.image, H, W, 16, 16, 2);
    REQUIRE_THAT(r_center, WithinAbs(0.99f, 1e-3f));
    REQUIRE_THAT(g_center, WithinAbs(0.0f, 1e-3f));
    REQUIRE_THAT(b_center, WithinAbs(0.0f, 1e-3f));
}

TEST_CASE("blend: two Gaussians front-to-back second dominates center", "[blend]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;

    const float means_2d[] = {16.5f, 16.5f, 16.5f, 16.5f};
    const float covs_2d[] = {50.0f, 0.0f, 0.0f, 50.0f, 50.0f, 0.0f, 0.0f, 50.0f};
    const float colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const float opacities[] = {0.01f, 1.0f};
    const int64_t sorted_gaussian_ids[] = {0, 1};
    const int64_t tile_ranges[] = {0, 2};

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges, 2, 2, H, W,
        tile_size, test_pool());

    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    const float g_center = pixel_at(result.image, H, W, 16, 16, 1);
    REQUIRE(r_center < 0.05f);
    REQUIRE(g_center > 0.95f);
}

TEST_CASE("blend: transmittance early termination with stacked opaque Gaussians", "[blend]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;
    const int num_g = 100;

    std::vector<float> means_2d(static_cast<std::size_t>(num_g) * 2, 16.5f);
    std::vector<float> covs_2d(static_cast<std::size_t>(num_g) * 4);
    std::vector<float> colors(static_cast<std::size_t>(num_g) * 3, 0.0f);
    std::vector<float> opacities(static_cast<std::size_t>(num_g), 1.0f);
    std::vector<int64_t> sorted_gaussian_ids(static_cast<std::size_t>(num_g));
    for (int g = 0; g < num_g; ++g) {
        covs_2d[static_cast<std::size_t>(g) * 4 + 0] = 0.01f;
        covs_2d[static_cast<std::size_t>(g) * 4 + 1] = 0.0f;
        covs_2d[static_cast<std::size_t>(g) * 4 + 3] = 0.01f;
        colors[static_cast<std::size_t>(g) * 3 + 0] = 1.0f;
        sorted_gaussian_ids[static_cast<std::size_t>(g)] = g;
    }
    const int64_t tile_ranges[] = {0, num_g};

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend(
        means_2d.data(), covs_2d.data(), colors.data(), opacities.data(),
        sorted_gaussian_ids.data(), tile_ranges, static_cast<std::size_t>(num_g),
        static_cast<std::size_t>(num_g), H, W, tile_size, test_pool());

    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    REQUIRE(r_center > 0.99f);
    REQUIRE(r_center <= 1.0f);
}

TEST_CASE("blend: empty tile stays zero", "[blend]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;

    const float means_2d[] = {16.5f, 16.5f};
    const float covs_2d[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float colors[] = {1.0f, 1.0f, 1.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 0, 0, 1};

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, H, W,
        tile_size, test_pool());

    REQUIRE(std::all_of(result.image.begin(), result.image.end(),
                        [](const float v) { return v == 0.0f; }));
}

TEST_CASE("blend: edge tile writes only in-bounds region", "[blend]") {
    const int H = 40;
    const int W = 40;
    const int tile_size = 32;

    const float means_2d[] = {35.5f, 35.5f};
    const float covs_2d[] = {20.0f, 0.0f, 0.0f, 20.0f};
    const float colors[] = {0.0f, 0.0f, 1.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};

    const int tiles_x = (W + tile_size - 1) / tile_size;
    const int tiles_y = (H + tile_size - 1) / tile_size;
    const int num_tiles = tiles_x * tiles_y;
    std::vector<int64_t> tile_ranges(static_cast<std::size_t>(num_tiles) * 2, 0);
    const int corner_tile = (tiles_y - 1) * tiles_x + (tiles_x - 1);
    tile_ranges[static_cast<std::size_t>(corner_tile) * 2 + 0] = 0;
    tile_ranges[static_cast<std::size_t>(corner_tile) * 2 + 1] = 1;

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend(
        means_2d, covs_2d, colors, opacities, sorted_gaussian_ids, tile_ranges.data(), 1, 1,
        H, W, tile_size, test_pool());

    REQUIRE(pixel_at(result.image, H, W, 35, 35, 2) > 0.5f);
    REQUIRE(pixel_at(result.image, H, W, 0, 0, 0) == 0.0f);
    REQUIRE(pixel_at(result.image, H, W, 39, 39, 2) > 0.1f);
}
