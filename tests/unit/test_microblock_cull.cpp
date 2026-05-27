#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <vector>

#include "cnpy.h"
#include "gsplat_cpu/microblock_cull.h"
#include "gsplat_cpu/thread_pool.h"

namespace {

gsplat_cpu::ThreadPool& test_pool() {
    static gsplat_cpu::ThreadPool pool(1);
    return pool;
}

constexpr float kMbFloor = 1.0f / 16384.0f;

int header_offset(const std::vector<int64_t>& header, int tile_id, int m) {
    return static_cast<int>(
        header[(static_cast<std::size_t>(tile_id) * 32 + static_cast<std::size_t>(m)) * 2 + 0]);
}

int header_count(const std::vector<int64_t>& header, int tile_id, int m) {
    return static_cast<int>(
        header[(static_cast<std::size_t>(tile_id) * 32 + static_cast<std::size_t>(m)) * 2 + 1]);
}

int total_kept(const std::vector<int64_t>& header, int tile_id) {
    int sum = 0;
    for (int m = 0; m < 32; ++m) {
        sum += header_count(header, tile_id, m);
    }
    return sum;
}

std::string hero_fixture_dir() {
#ifdef GSPLAT_HERO_FIXTURE_DIR
    return GSPLAT_HERO_FIXTURE_DIR;
#else
    return "tests/fixtures/hero";
#endif
}

}  // namespace

TEST_CASE("microblock_cull: single tile center Gaussian small sigma keeps one microblock",
          "[microblock_cull]") {
    const int tiles_x = 1;
    const int tiles_y = 1;
    const int tile_size = 32;

    const float means_2d[] = {16.0f, 16.0f};
    const float covs_2d[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 1};

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    REQUIRE(result.mb_header.size() == static_cast<std::size_t>(32 * 2));
    REQUIRE(total_kept(result.mb_header, 0) == 8);

    int center_m = -1;
    for (int m = 0; m < 32; ++m) {
        const int count = header_count(result.mb_header, 0, m);
        if (count == 1) {
            if (m == 18) {
                center_m = m;
            }
        } else {
            REQUIRE(count == 0);
        }
    }
    REQUIRE(center_m == 18);
    REQUIRE(result.mb_stream.size() == 8);
}

TEST_CASE("microblock_cull: single tile center Gaussian large sigma keeps many microblocks",
          "[microblock_cull]") {
    const int tiles_x = 1;
    const int tiles_y = 1;
    const int tile_size = 32;

    const float means_2d[] = {16.0f, 16.0f};
    const float covs_2d[] = {400.0f, 0.0f, 0.0f, 400.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 1};

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    REQUIRE(total_kept(result.mb_header, 0) >= 4);
}

TEST_CASE("microblock_cull: depth order preserved within microblock", "[microblock_cull]") {
    const int tiles_x = 1;
    const int tiles_y = 1;
    const int tile_size = 32;

    const float means_2d[] = {16.0f, 16.0f, 16.0f, 16.0f};
    const float covs_2d[] = {400.0f, 0.0f, 0.0f, 400.0f, 400.0f, 0.0f, 0.0f, 400.0f};
    const float opacities[] = {1.0f, 1.0f};
    const int64_t sorted_gaussian_ids[] = {0, 1};
    const int64_t tile_ranges[] = {0, 2};

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 2, 2, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    bool checked = false;
    for (int m = 0; m < 32; ++m) {
        const int count = header_count(result.mb_header, 0, m);
        if (count < 2) {
            continue;
        }
        checked = true;
        const int off = header_offset(result.mb_header, 0, m);
        for (int i = 0; i < count; ++i) {
            REQUIRE(result.mb_stream[static_cast<std::size_t>(off + i)] == static_cast<int64_t>(i));
        }
    }
    REQUIRE(checked);
}

TEST_CASE("microblock_cull: empty tile header stays zero", "[microblock_cull]") {
    const int tiles_x = 2;
    const int tiles_y = 1;
    const int tile_size = 32;

    const float means_2d[] = {16.0f, 16.0f};
    const float covs_2d[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float opacities[] = {1.0f};
    const int64_t sorted_gaussian_ids[] = {0};
    const int64_t tile_ranges[] = {0, 1, 0, 0};

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        means_2d, covs_2d, opacities, sorted_gaussian_ids, tile_ranges, 1, 1, tiles_x, tiles_y,
        tile_size, kMbFloor, test_pool());

    REQUIRE(result.mb_header.size() == static_cast<std::size_t>(2 * 32 * 2));
    for (int m = 0; m < 32; ++m) {
        REQUIRE(header_count(result.mb_header, 1, m) == 0);
        REQUIRE(header_offset(result.mb_header, 1, m) == 0);
    }
    REQUIRE(total_kept(result.mb_header, 0) == 8);
}

TEST_CASE("microblock_cull: hero fixture round-trip", "[microblock_cull][hero]") {
    const std::string fixture_dir = hero_fixture_dir();
    const std::string inputs_npz = fixture_dir + "/microblock_cull_inputs.npz";
    const std::string outputs_npz = fixture_dir + "/microblock_cull_outputs.npz";

    if (!std::filesystem::exists(inputs_npz) || !std::filesystem::exists(outputs_npz)) {
        SKIP("hero microblock_cull fixtures missing; run scripts/dump_microblock_fixture.py");
    }

    const cnpy::npz_t inputs = cnpy::npz_load(inputs_npz);
    const cnpy::npz_t outputs = cnpy::npz_load(outputs_npz);

    const cnpy::NpyArray& means_arr = inputs.at("means_2d");
    const cnpy::NpyArray& covs_arr = inputs.at("covs_2d");
    const cnpy::NpyArray& opacities_arr = inputs.at("opacities");
    const cnpy::NpyArray& sgids_arr = inputs.at("sorted_gaussian_ids");
    const cnpy::NpyArray& ranges_arr = inputs.at("tile_ranges");
    const cnpy::NpyArray& ref_header = outputs.at("mb_header");
    const cnpy::NpyArray& ref_stream = outputs.at("mb_stream");

    const auto* means_2d = means_arr.data<float>();
    const auto* opacities = opacities_arr.data<float>();
    const auto* sorted_gaussian_ids = sgids_arr.data<int64_t>();
    const auto* tile_ranges = ranges_arr.data<int64_t>();
    const auto* ref_header_data = ref_header.data<int64_t>();
    const auto* ref_stream_data = ref_stream.data<int64_t>();

    const std::size_t M = means_arr.shape[0];
    const std::size_t P = sgids_arr.shape[0];
    const int tiles_x = static_cast<int>(inputs.at("tiles_x").data<int64_t>()[0]);
    const int tiles_y = static_cast<int>(inputs.at("tiles_y").data<int64_t>()[0]);
    const float mb_contrib_floor =
        static_cast<float>(inputs.at("mb_contrib_floor").data<float>()[0]);

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

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        means_2d, covs_flat.data(), opacities, sorted_gaussian_ids, tile_ranges, M, P, tiles_x,
        tiles_y, 32, mb_contrib_floor, test_pool());

    REQUIRE(result.mb_header.size() == ref_header.num_vals);
    REQUIRE(result.mb_stream.size() == ref_stream.num_vals);
    for (std::size_t i = 0; i < result.mb_header.size(); ++i) {
        REQUIRE(result.mb_header[i] == ref_header_data[i]);
    }
    for (std::size_t i = 0; i < result.mb_stream.size(); ++i) {
        REQUIRE(result.mb_stream[i] == ref_stream_data[i]);
    }
}
