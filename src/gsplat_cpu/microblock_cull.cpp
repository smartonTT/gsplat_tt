#include "gsplat_cpu/microblock_cull.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace gsplat_cpu {

namespace {

constexpr int kNumMicroblocks = 32;

// M0 L1_RECORD debug: one-shot reporting of corrupt device-sort output
// (tile_ranges / sorted_gaussian_ids) consumed by the host microblock cull.
static std::atomic<int> g_cull_oob_reported{0};

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
    const float mb_contrib_floor,
    const std::size_t P,
    const std::size_t M,
    TileCullLocal& out) {
    const int64_t start = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 0];
    const int64_t end = tile_ranges[static_cast<std::size_t>(tile_id) * 2 + 1];
    if (start == end) {
        return;
    }
    if (start < 0 || end < start || static_cast<std::size_t>(end) > P) {
        if (g_cull_oob_reported.fetch_add(1) < 16) {
            std::fprintf(stderr,
                "[CULL_OOB] range tile=%d start=%lld end=%lld P=%zu\n",
                tile_id, (long long)start, (long long)end, P);
        }
        return;
    }

    const int ty = tile_id / tiles_x;
    const int tx = tile_id % tiles_x;
    const float tx_tile = static_cast<float>(tx * tile_size);
    const float ty_tile = static_cast<float>(ty * tile_size);

    const int64_t L = end - start;
    std::vector<int64_t> tile_g_ids(static_cast<std::size_t>(L));
    for (int64_t i = 0; i < L; ++i) {
        const int64_t gid = sorted_gaussian_ids[static_cast<std::size_t>(start + i)];
        if (gid < 0 || static_cast<std::size_t>(gid) >= M) {
            if (g_cull_oob_reported.fetch_add(1) < 16) {
                std::fprintf(stderr,
                    "[CULL_OOB] gid tile=%d start=%lld end=%lld i=%lld gid=%lld M=%zu\n",
                    tile_id, (long long)start, (long long)end, (long long)i,
                    (long long)gid, M);
            }
            tile_g_ids[static_cast<std::size_t>(i)] = 0;  // clamp so the cull can finish
            continue;
        }
        tile_g_ids[static_cast<std::size_t>(i)] = gid;
    }

    std::vector<bool> keep_any(static_cast<std::size_t>(L), false);
    std::vector<std::array<bool, kNumMicroblocks>> keep_mask(static_cast<std::size_t>(L));

    for (int64_t l = 0; l < L; ++l) {
        const int64_t g = tile_g_ids[static_cast<std::size_t>(l)];
        const std::size_t gs = static_cast<std::size_t>(g);

        const float mean_x = means_2d[gs * 2 + 0];
        const float mean_y = means_2d[gs * 2 + 1];
        const float a = covs_2d[gs * 4 + 0];
        const float b = covs_2d[gs * 4 + 1];
        const float c = covs_2d[gs * 4 + 3];
        const float g_op = opacities[gs];

        const float det = std::max(a * c - b * b, 1e-6f);
        const float ci_a = c / det;
        const float ci_b = -b / det;
        const float ci_c = a / det;

        // iter-015 BB prefilter + iter-017 exp-elimination:
        //   keep = (alpha_peak >= mb_contrib_floor)
        //        = (g_op * exp(power_c) >= floor)        // 0.99-cap is harmless: 0.99 >> floor
        //        = (power_c >= log(floor / g_op))
        //        = (power_c >= log_thresh)               // precomputed once per Gaussian
        // Note power_c <= 0 always (negative of a PSD quadratic form), and
        // log_thresh < 0 iff g_op > floor (precondition below).
        if (g_op <= mb_contrib_floor) {
            continue;
        }

        const float log_thresh = std::log(mb_contrib_floor / g_op);  // <= 0
        const float r_sq = -2.0f * log_thresh;                        // > 0
        const float r = std::sqrt(r_sq);
        const float x_half = r * std::sqrt(a);
        const float y_half = r * std::sqrt(c);

        const float bb_x_min = mean_x - x_half;
        const float bb_x_max = mean_x + x_half;
        const float bb_y_min = mean_y - y_half;
        const float bb_y_max = mean_y + y_half;

        const float tile_x_local_min = bb_x_min - tx_tile;
        const float tile_x_local_max = bb_x_max - tx_tile;
        const float tile_y_local_min = bb_y_min - ty_tile;
        const float tile_y_local_max = bb_y_max - ty_tile;

        const int mx_lo = std::max(0, static_cast<int>(std::floor(tile_x_local_min / 8.0f)));
        const int mx_hi = std::min(3, static_cast<int>(std::floor(tile_x_local_max / 8.0f)));
        const int my_lo = std::max(0, static_cast<int>(std::floor(tile_y_local_min / 4.0f)));
        const int my_hi = std::min(7, static_cast<int>(std::floor(tile_y_local_max / 4.0f)));

        if (mx_lo > mx_hi || my_lo > my_hi) {
            continue;
        }

        // Proper-min Mahalanobis cull (fixes tilted-Gaussian over-cull bug —
        // see cull_and_blend.cpp for the full explanation).
        const float thresh_m2 = -2.0f * log_thresh;  // m² ≤ thresh ⇔ keep
        const float ci_a_safe = std::max(ci_a, 1e-12f);
        const float ci_c_safe = std::max(ci_c, 1e-12f);

        for (int my = my_lo; my <= my_hi; ++my) {
            const float mb_oy = ty_tile + static_cast<float>(my * 4);
            const float v_lo = mb_oy - mean_y;
            const float v_hi = v_lo + 4.0f;
            const bool y_inside = (v_lo <= 0.0f) && (0.0f <= v_hi);
            const float v_fix = (v_lo > 0.0f) ? v_lo : v_hi;
            for (int mx = mx_lo; mx <= mx_hi; ++mx) {
                const int m = (my << 2) | mx;
                const float mb_ox = tx_tile + static_cast<float>(mx * 8);
                const float u_lo = mb_ox - mean_x;
                const float u_hi = u_lo + 8.0f;
                const bool x_inside = (u_lo <= 0.0f) && (0.0f <= u_hi);

                float m2_min;
                if (x_inside && y_inside) {
                    m2_min = 0.0f;
                } else {
                    float m2_v = std::numeric_limits<float>::infinity();
                    if (!x_inside) {
                        const float u_fix = (u_lo > 0.0f) ? u_lo : u_hi;
                        float v_star = -ci_b * u_fix / ci_c_safe;
                        v_star = std::clamp(v_star, v_lo, v_hi);
                        m2_v = ci_a * u_fix * u_fix
                               + 2.0f * ci_b * u_fix * v_star
                               + ci_c * v_star * v_star;
                    }
                    float m2_h = std::numeric_limits<float>::infinity();
                    if (!y_inside) {
                        float u_star = -ci_b * v_fix / ci_a_safe;
                        u_star = std::clamp(u_star, u_lo, u_hi);
                        m2_h = ci_a * u_star * u_star
                               + 2.0f * ci_b * u_star * v_fix
                               + ci_c * v_fix * v_fix;
                    }
                    m2_min = std::min(m2_v, m2_h);
                }
                const bool keep = m2_min <= thresh_m2;
                keep_mask[static_cast<std::size_t>(l)][static_cast<std::size_t>(m)] = keep;
                if (keep) {
                    keep_any[static_cast<std::size_t>(l)] = true;
                }
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

    const int num_tiles = tiles_x * tiles_y;
    const float floor = mb_contrib_floor;

    MicroblockCullResult result;
    result.pairs_in = static_cast<int64_t>(P);
    result.mb_header.assign(static_cast<std::size_t>(num_tiles) * kNumMicroblocks * 2, 0);

    std::vector<TileCullLocal> tile_locals(static_cast<std::size_t>(num_tiles));

    for (int tile_id = 0; tile_id < num_tiles; ++tile_id) {
        pool.submit([&, tile_id]() {
            cull_tile(tile_id, tiles_x, tile_size, means_2d, covs_2d, opacities,
                      sorted_gaussian_ids, tile_ranges, floor, P, M,
                      tile_locals[static_cast<std::size_t>(tile_id)]);
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
