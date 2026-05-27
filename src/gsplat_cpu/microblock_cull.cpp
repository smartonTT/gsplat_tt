#include "gsplat_cpu/microblock_cull.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

namespace {

constexpr int kNumMicroblocks = 32;

constexpr int mb_origin_x(int m) { return (m & 3) * 8; }
constexpr int mb_origin_y(int m) { return (m >> 2) * 4; }

struct TileCullLocal {
    int64_t tile_pairs_dropped{0};
    std::array<int64_t, kNumMicroblocks> counts{};
    std::array<std::vector<int64_t>, kNumMicroblocks> stream_per_m{};
};

void cull_tile(
    const int tile_id,
    const int tiles_x,
    const int tile_size,
    const float* means_2d,
    const float* covs_2d,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const double mb_contrib_floor,
    TileCullLocal& out) {
    const int64_t start = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 0];
    const int64_t end = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 1];
    if (start == end) {
        return;
    }

    const int ty = tile_id / tiles_x;
    const int tx = tile_id % tiles_x;
    const double tx_tile = static_cast<double>(tx * tile_size);
    const double ty_tile = static_cast<double>(ty * tile_size);

    const int64_t L = end - start;
    std::vector<int64_t> tile_g_ids(static_cast<std::size_t>(L));
    for (int64_t i = 0; i < L; ++i) {
        tile_g_ids[static_cast<std::size_t>(i)] =
            sorted_gaussian_ids[static_cast<std::size_t>(start + i)];
    }

    std::vector<bool> keep_any(static_cast<std::size_t>(L), false);
    std::vector<std::array<bool, kNumMicroblocks>> keep_mask(static_cast<std::size_t>(L));

    for (int64_t l = 0; l < L; ++l) {
        const int64_t g = tile_g_ids[static_cast<std::size_t>(l)];
        const std::size_t gs = static_cast<std::size_t>(g);

        const double mean_x = static_cast<double>(means_2d[gs * 2 + 0]);
        const double mean_y = static_cast<double>(means_2d[gs * 2 + 1]);
        const double a = static_cast<double>(covs_2d[gs * 4 + 0]);
        const double b = static_cast<double>(covs_2d[gs * 4 + 1]);
        const double c = static_cast<double>(covs_2d[gs * 4 + 3]);
        const double g_op = static_cast<double>(opacities[gs]);

        const double det = std::max(a * c - b * b, 1e-6);
        const double ci_a = c / det;
        const double ci_b = -b / det;
        const double ci_c = a / det;

        for (int m = 0; m < kNumMicroblocks; ++m) {
            const double mb_ox = tx_tile + static_cast<double>(mb_origin_x(m));
            const double mb_oy = ty_tile + static_cast<double>(mb_origin_y(m));
            const double cx = std::clamp(mean_x, mb_ox, mb_ox + 8.0);
            const double cy = std::clamp(mean_y, mb_oy, mb_oy + 4.0);
            const double dx_c = cx - mean_x;
            const double dy_c = cy - mean_y;
            const double power_c =
                -0.5 * (ci_a * dx_c * dx_c + 2.0 * ci_b * dx_c * dy_c + ci_c * dy_c * dy_c);
            const double alpha_peak =
                std::min(g_op * std::exp(std::min(power_c, 0.0)), 0.99);
            const bool keep = alpha_peak >= mb_contrib_floor;
            keep_mask[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] = keep;
            if (keep) {
                keep_any[static_cast<std::size_t>(l)] = true;
            }
        }
    }

    for (int64_t l = 0; l < L; ++l) {
        if (!keep_any[static_cast<std::size_t>(l)]) {
            ++out.tile_pairs_dropped;
        }
    }

    for (int m = 0; m < kNumMicroblocks; ++m) {
        auto& kept = out.stream_per_m[static_cast<std::size_t>(m)];
        kept.reserve(static_cast<std::size_t>(L));
        for (int64_t l = 0; l < L; ++l) {
            if (keep_mask[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)]) {
                kept.push_back(tile_g_ids[static_cast<std::size_t>(l)]);
            }
        }
        out.counts[static_cast<std::size_t>(m)] = static_cast<int64_t>(kept.size());
    }
}

}  // namespace

MicroblockCullResult microblock_cull(
    const float* means_2d,
    const float* covs_2d,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    const std::size_t M,
    const std::size_t P,
    const int tiles_x,
    const int tiles_y,
    const int tile_size,
    const float mb_contrib_floor,
    ThreadPool& pool) {
    (void)M;

    const int num_tiles = tiles_x * tiles_y;
    const double floor = static_cast<double>(mb_contrib_floor);

    MicroblockCullResult result;
    result.pairs_in = static_cast<int64_t>(P);
    result.mb_header.assign(static_cast<std::size_t>(num_tiles) * kNumMicroblocks * 2, 0);

    std::vector<TileCullLocal> tile_locals(static_cast<std::size_t>(num_tiles));

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        pool.submit([&, tile_id]() {
            cull_tile(tile_id, tiles_x, tile_size, means_2d, covs_2d, opacities,
                      sorted_gaussian_ids, tile_ranges, floor, tile_locals[static_cast<std::size_t>(tile_id)]);
        });
    }
    pool.wait();

    int64_t stream_offset = 0;
    int64_t pairs_dropped = 0;
    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        const int64_t start = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 0];
        const int64_t end = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 1];
        if (start == end) {
            continue;
        }

        const TileCullLocal& local = tile_locals[static_cast<std::size_t>(tile_id)];
        pairs_dropped += local.tile_pairs_dropped;

        for (int m = 0; m < kNumMicroblocks; ++m) {
            const std::size_t header_base =
                (static_cast<std::size_t>(tile_id) * static_cast<std::size_t>(kNumMicroblocks) +
                 static_cast<std::size_t>(m)) *
                2;
            const int64_t count = local.counts[static_cast<std::size_t>(m)];
            result.mb_header[header_base + 0] = stream_offset;
            result.mb_header[header_base + 1] = count;
            if (count > 0) {
                const auto& chunk = local.stream_per_m[static_cast<std::size_t>(m)];
                result.mb_stream.insert(result.mb_stream.end(), chunk.begin(), chunk.end());
                stream_offset += count;
            }
        }
    }

    result.pairs_out = static_cast<int64_t>(result.mb_stream.size());
    result.pairs_dropped_in_all_mb = pairs_dropped;
    return result;
}

}  // namespace gsplat_cpu
