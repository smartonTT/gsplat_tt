#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include "cnpy.h"
#include "gsplat_cpu/blend_microblock.h"
#include "gsplat_cpu/microblock_cull.h"
#include "gsplat_cpu/thread_pool.h"

using Catch::Matchers::WithinAbs;

namespace {

gsplat_cpu::ThreadPool& test_pool() {
    static gsplat_cpu::ThreadPool pool(1);
    return pool;
}

constexpr float kMbFloor = 1.0f / 16384.0f;

float pixel_at(const std::vector<float>& image, int H, int W, int y, int x, int c) {
    return image[(static_cast<std::size_t>(y) * static_cast<std::size_t>(W) +
                  static_cast<std::size_t>(x)) *
                     3 +
                 static_cast<std::size_t>(c)];
}

int header_offset(const std::vector<int64_t>& header, int tile_id, int m) {
    return static_cast<int>(
        header[(static_cast<std::size_t>(tile_id) * 32 + static_cast<std::size_t>(m)) * 2 + 0]);
}

int header_count(const std::vector<int64_t>& header, int tile_id, int m) {
    return static_cast<int>(
        header[(static_cast<std::size_t>(tile_id) * 32 + static_cast<std::size_t>(m)) * 2 + 1]);
}

std::string hero_fixture_dir() {
#ifdef GSPLAT_HERO_FIXTURE_DIR
    return GSPLAT_HERO_FIXTURE_DIR;
#else
    return "tests/fixtures/hero";
#endif
}

double compute_psnr(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double mse = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += d * d;
    }
    mse /= static_cast<double>(a.size());
    if (mse <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return 10.0 * std::log10(1.0 / mse);
}

}  // namespace

TEST_CASE("blend_microblock: single tile single Gaussian at center", "[blend_microblock]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;
    const int tiles_x = 1;
    const int tiles_y = 1;

    const float means_2d[] = {16.0f, 16.0f};
    const float covs_2d[] = {50.0f, 0.0f, 0.0f, 50.0f};
    const float colors[] = {1.0f, 0.0f, 0.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 1};

    const gsplat_cpu::MicroblockCullResult cull = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d, covs_2d, colors, opacities, cull.mb_header.data(), cull.mb_stream.data(), 1,
        cull.mb_stream.size(), H, W, tile_size, test_pool());

    REQUIRE(result.image.size() == static_cast<std::size_t>(H * W * 3));
    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    const float g_center = pixel_at(result.image, H, W, 16, 16, 1);
    const float b_center = pixel_at(result.image, H, W, 16, 16, 2);
    REQUIRE_THAT(r_center, WithinAbs(0.99f, 1e-2f));
    REQUIRE_THAT(g_center, WithinAbs(0.0f, 1e-2f));
    REQUIRE_THAT(b_center, WithinAbs(0.0f, 1e-2f));
}

TEST_CASE("blend_microblock: empty tile stays zero", "[blend_microblock]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;

    const float means_2d[] = {16.0f, 16.0f};
    const float covs_2d[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float colors[] = {1.0f, 1.0f, 1.0f};
    const float opacities[] = {1.0f};
    const std::vector<int64_t> mb_header(32 * 2, 0);
    const std::vector<int64_t> mb_stream;

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d, covs_2d, colors, opacities, mb_header.data(), mb_stream.data(), 1, 0, H, W,
        tile_size, test_pool());

    REQUIRE(std::all_of(result.image.begin(), result.image.end(),
                        [](const float v) { return v == 0.0f; }));
}

TEST_CASE("blend_microblock: edge tile writes only in-bounds region", "[blend_microblock]") {
    const int H = 40;
    const int W = 40;
    const int tile_size = 32;
    const int tiles_x = (W + tile_size - 1) / tile_size;
    const int tiles_y = (H + tile_size - 1) / tile_size;

    const float means_2d[] = {35.0f, 35.0f};
    const float covs_2d[] = {20.0f, 0.0f, 0.0f, 20.0f};
    const float colors[] = {0.0f, 0.0f, 1.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};

    const int num_tiles = tiles_x * tiles_y;
    std::vector<int64_t> tile_ranges(static_cast<std::size_t>(num_tiles) * 2, 0);
    const int corner_tile = (tiles_y - 1) * tiles_x + (tiles_x - 1);
    tile_ranges[static_cast<std::size_t>(corner_tile) * 2 + 0] = 0;
    tile_ranges[static_cast<std::size_t>(corner_tile) * 2 + 1] = 1;

    const gsplat_cpu::MicroblockCullResult cull = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges.data(), 1, 1, tiles_x,
        tiles_y, tile_size, kMbFloor, test_pool());

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d, covs_2d, colors, opacities, cull.mb_header.data(), cull.mb_stream.data(), 1,
        cull.mb_stream.size(), H, W, tile_size, test_pool());

    REQUIRE(pixel_at(result.image, H, W, 35, 35, 2) > 0.5f);
    REQUIRE(pixel_at(result.image, H, W, 0, 0, 0) == 0.0f);
    REQUIRE(pixel_at(result.image, H, W, 39, 39, 2) > 0.1f);
}

TEST_CASE("blend_microblock: two Gaussians front-to-back second dominates center",
          "[blend_microblock]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;
    const int tiles_x = 1;
    const int tiles_y = 1;

    const float means_2d[] = {16.0f, 16.0f, 16.0f, 16.0f};
    const float covs_2d[] = {50.0f, 0.0f, 0.0f, 50.0f, 50.0f, 0.0f, 0.0f, 50.0f};
    const float colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    const float opacities[] = {0.01f, 1.0f};
    const int64_t sorted_gaussian_ids[] = {0, 1};
    const int64_t tile_ranges[] = {0, 2};

    const gsplat_cpu::MicroblockCullResult cull = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 2, 2, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d, covs_2d, colors, opacities, cull.mb_header.data(), cull.mb_stream.data(), 2,
        cull.mb_stream.size(), H, W, tile_size, test_pool());

    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    const float g_center = pixel_at(result.image, H, W, 16, 16, 1);
    REQUIRE(r_center < 0.05f);
    REQUIRE(g_center > 0.95f);
}

TEST_CASE("blend_microblock: per-microblock early termination with stacked opaque Gaussians",
          "[blend_microblock]") {
    const int H = 32;
    const int W = 32;
    const int tile_size = 32;
    const int tiles_x = 1;
    const int tiles_y = 1;
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

    const gsplat_cpu::MicroblockCullResult cull = gsplat_cpu::microblock_cull(
        means_2d.data(), covs_2d.data(), opacities.data(), sorted_gaussian_ids.data(),
        tile_ranges, static_cast<std::size_t>(num_g), static_cast<std::size_t>(num_g), tiles_x,
        tiles_y, tile_size, kMbFloor, test_pool());

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d.data(), covs_2d.data(), colors.data(), opacities.data(), cull.mb_header.data(),
        cull.mb_stream.data(), static_cast<std::size_t>(num_g), cull.mb_stream.size(), H, W,
        tile_size, test_pool());

    const float r_center = pixel_at(result.image, H, W, 16, 16, 0);
    REQUIRE(r_center > 0.99f);
    REQUIRE(r_center <= 1.0f);
}

TEST_CASE("blend_microblock: hero fixture round-trip PSNR", "[blend_microblock][hero]") {
    const std::string fixture_dir = hero_fixture_dir();
    const std::string inputs_npz = fixture_dir + "/microblock_cull_inputs.npz";
    const std::string outputs_npz = fixture_dir + "/microblock_cull_outputs.npz";

    if (!std::filesystem::exists(inputs_npz) || !std::filesystem::exists(outputs_npz)) {
        SKIP("hero blend_microblock fixtures missing; run scripts/dump_microblock_fixture.py");
    }

    const cnpy::npz_t inputs = cnpy::npz_load(inputs_npz);
    const cnpy::npz_t outputs = cnpy::npz_load(outputs_npz);

    if (outputs.count("blend_image") == 0) {
        SKIP("blend_image missing from microblock_cull_outputs.npz; re-run dump_microblock_fixture.py");
    }

    const cnpy::NpyArray& means_arr = inputs.at("means_2d");
    const cnpy::NpyArray& covs_arr = inputs.at("covs_2d");
    const cnpy::NpyArray& opacities_arr = inputs.at("opacities");
    const cnpy::NpyArray& ref_header = outputs.at("mb_header");
    const cnpy::NpyArray& ref_stream = outputs.at("mb_stream");
    const cnpy::NpyArray& ref_arr = outputs.at("blend_image");

    const auto* means_2d = means_arr.data<float>();
    const auto* opacities = opacities_arr.data<float>();
    const auto* mb_header = ref_header.data<int64_t>();
    const auto* mb_stream = ref_stream.data<int64_t>();

    const std::size_t M = means_arr.shape[0];
    const int H = static_cast<int>(ref_arr.shape[0]);
    const int W = static_cast<int>(ref_arr.shape[1]);
    const int tile_size = 32;

    std::vector<float> covs_flat(M * 4);
    const float* covs_src = covs_arr.data<float>();
    if (covs_arr.shape.size() == 3) {
        for (std::size_t i = 0; i < M; ++i) {
            covs_flat[i * 4 + 0] = covs_src[i * 4 + 0];
            covs_flat[i * 4 + 1] = covs_src[i * 4 + 1];
            covs_flat[i * 4 + 2] = covs_src[i * 4 + 1];
            covs_flat[i * 4 + 3] = covs_src[i * 4 + 3];
        }
    } else {
        std::copy(covs_src, covs_src + M * 4, covs_flat.begin());
    }

    const cnpy::NpyArray& colors_arr = inputs.at("colors");
    std::vector<float> colors(M * 3);
    std::copy(colors_arr.data<float>(), colors_arr.data<float>() + M * 3, colors.begin());

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        means_2d, covs_flat.data(), colors.data(), opacities, mb_header, mb_stream, M,
        ref_stream.num_vals, H, W, tile_size, test_pool());

    const float* ref_data = ref_arr.data<float>();
    std::vector<float> ref_vec(ref_data, ref_data + ref_arr.num_vals);
    const double psnr = compute_psnr(result.image, ref_vec);
    REQUIRE(psnr >= 60.0);
}
