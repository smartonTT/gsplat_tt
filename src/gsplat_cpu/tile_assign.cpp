#include "gsplat_cpu/tile_assign.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

namespace {

int clamp_int(int v, int lo, int hi) {
    return std::max(lo, std::min(v, hi));
}

float clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

}  // namespace

TileAssignResult tile_assign(
    const float* means_2d,
    const float* radii,
    const std::size_t M,
    const int image_height,
    const int image_width,
    const int tile_size,
    const float* covs_2d,
    const float* opacities,
    const float contrib_floor,
    ThreadPool* pool) {
    TileAssignResult result;
    result.tiles_per_gaussian.assign(M, 0);

    if (M == 0) {
        return result;
    }

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;

    // Phase 1: per-Gaussian BB rect (tile-space), parallel.
    // Stores tile rect [min_x..max_x] x [min_y..max_y] inclusive and the
    // pair count widths*heights into tiles_per_gaussian.
    std::vector<int> tile_min_x(M);
    std::vector<int> tile_max_x(M);
    std::vector<int> tile_min_y(M);
    std::vector<int> tile_max_y(M);
    std::vector<int> widths(M);
    std::vector<int> heights(M);

    const std::size_t W = (pool != nullptr) ? std::max<std::size_t>(1, pool->size()) : 1;

    auto bbox_one = [&](std::size_t m) {
        const float px = means_2d[m * 2 + 0];
        const float py = means_2d[m * 2 + 1];
        const float rx = radii[m * 2 + 0];
        const float ry = radii[m * 2 + 1];

        const int min_x =
            clamp_int(static_cast<int>((px - rx) / static_cast<float>(tile_size)), 0, tiles_x - 1);
        const int max_x =
            clamp_int(static_cast<int>((px + rx) / static_cast<float>(tile_size)), 0, tiles_x - 1);
        const int min_y =
            clamp_int(static_cast<int>((py - ry) / static_cast<float>(tile_size)), 0, tiles_y - 1);
        const int max_y =
            clamp_int(static_cast<int>((py + ry) / static_cast<float>(tile_size)), 0, tiles_y - 1);

        tile_min_x[m] = min_x;
        tile_max_x[m] = max_x;
        tile_min_y[m] = min_y;
        tile_max_y[m] = max_y;
        const int w = max_x - min_x + 1;
        const int h = max_y - min_y + 1;
        widths[m] = w;
        heights[m] = h;
        result.tiles_per_gaussian[m] = static_cast<int64_t>(w) * static_cast<int64_t>(h);
    };

    if (pool != nullptr && W > 1 && M >= 4096) {
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, M, &bbox_one]() {
                for (std::size_t m = w; m < M; m += W) bbox_one(m);
            });
        }
        pool->wait();
    } else {
        for (std::size_t m = 0; m < M; ++m) bbox_one(m);
    }

    // Phase 2: per-Gaussian write-offset prefix-sum (serial, but M=234k
    // single-pass scalar adds finish in ~0.2 ms; parallel scan adds more
    // overhead than it saves at this size).
    std::vector<std::size_t> offs(M + 1, 0);
    for (std::size_t m = 0; m < M; ++m) {
        offs[m + 1] = offs[m] + static_cast<std::size_t>(result.tiles_per_gaussian[m]);
    }
    const std::size_t P = offs[M];

    // Phase 3: parallel scatter of (g, tile_id) pairs into pre-sized arrays.
    // Each Gaussian writes its widths*heights pairs into [offs[m], offs[m+1])
    // independent of every other Gaussian. Replaces the previous serial
    // push_back loop (line 78-86 of iter-013 era) with a fully parallel
    // write — the dominant remaining serial cost in tile_assign.
    result.gaussian_ids.assign(P, 0);
    result.tile_ids.assign(P, 0);

    auto scatter_one = [&](std::size_t m) {
        const int min_x = tile_min_x[m];
        const int min_y = tile_min_y[m];
        const int wx = widths[m];
        const int hy = heights[m];
        std::size_t pos = offs[m];
        for (int dy = 0; dy < hy; ++dy) {
            const int64_t base = static_cast<int64_t>((min_y + dy) * tiles_x);
            for (int dx = 0; dx < wx; ++dx) {
                result.gaussian_ids[pos] = static_cast<int64_t>(m);
                result.tile_ids[pos] = base + static_cast<int64_t>(min_x + dx);
                ++pos;
            }
        }
    };

    if (pool != nullptr && W > 1 && M >= 4096) {
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, M, &scatter_one]() {
                for (std::size_t m = w; m < M; m += W) scatter_one(m);
            });
        }
        pool->wait();
    } else {
        for (std::size_t m = 0; m < M; ++m) scatter_one(m);
    }

    if (covs_2d != nullptr && opacities != nullptr && P > 0) {
        // Phase 4: parallel per-pair Mahalanobis cull. The cull is embarrassingly
        // parallel — each pair's contribution-at-tile-center depends only on
        // read-only inputs. Step 1: every worker writes its survival decision
        // into a shared keep_mask in a striped slice (no false sharing on
        // separate chunks). Step 2: parallel compaction via chunked
        // prefix-sum + parallel scatter (same pattern as project's iter-020).
        std::vector<uint8_t> keep_mask(P, 0);

        auto cull_stripe = [&](std::size_t w) {
            for (std::size_t p = w; p < P; p += W) {
                const int64_t g = result.gaussian_ids[p];
                const int64_t tile_id = result.tile_ids[p];

                const float a = covs_2d[static_cast<std::size_t>(g) * 4 + 0];
                const float b = covs_2d[static_cast<std::size_t>(g) * 4 + 1];
                const float c = covs_2d[static_cast<std::size_t>(g) * 4 + 3];
                const float det = std::max(a * c - b * b, 1e-6f);

                const float px = means_2d[static_cast<std::size_t>(g) * 2 + 0];
                const float py = means_2d[static_cast<std::size_t>(g) * 2 + 1];

                const float tx_tile =
                    static_cast<float>(tile_id % static_cast<int64_t>(tiles_x)) *
                    static_cast<float>(tile_size);
                const float ty_tile =
                    static_cast<float>(tile_id / static_cast<int64_t>(tiles_x)) *
                    static_cast<float>(tile_size);

                const float cx = clamp_float(px, tx_tile, tx_tile + static_cast<float>(tile_size));
                const float cy = clamp_float(py, ty_tile, ty_tile + static_cast<float>(tile_size));
                const float dx_c = cx - px;
                const float dy_c = cy - py;
                const float m2 =
                    (c * dx_c * dx_c - 2.0f * b * dx_c * dy_c + a * dy_c * dy_c) / det;
                const float contrib =
                    opacities[static_cast<std::size_t>(g)] * std::exp(-0.5f * m2);
                keep_mask[p] = contrib >= contrib_floor ? 1 : 0;
            }
        };

        if (pool != nullptr && W > 1) {
            for (std::size_t w = 0; w < W; ++w) {
                pool->submit([w, &cull_stripe]() { cull_stripe(w); });
            }
            pool->wait();
        } else {
            cull_stripe(0);
        }

        // Parallel compaction: chunked prefix-sum + parallel scatter.
        // Replaces the previous serial 294k-iteration push_back loop. Same
        // pattern as project's iter-020 gather_visible parallelization.
        // chunk_size chosen so that each worker has a contiguous span — this
        // preserves input ordering within each chunk and globally (because
        // chunks tile [0, P) in order).
        const std::size_t chunk_size = std::max<std::size_t>(1024, (P + W - 1) / W);
        const std::size_t num_chunks = (P + chunk_size - 1) / chunk_size;
        std::vector<std::size_t> chunk_counts(num_chunks, 0);

        auto count_chunk = [&](std::size_t c) {
            const std::size_t lo = c * chunk_size;
            const std::size_t hi = std::min(lo + chunk_size, P);
            std::size_t k = 0;
            for (std::size_t p = lo; p < hi; ++p) k += keep_mask[p];
            chunk_counts[c] = k;
        };
        if (pool != nullptr && W > 1 && num_chunks > 1) {
            for (std::size_t c = 0; c < num_chunks; ++c) {
                pool->submit([c, &count_chunk]() { count_chunk(c); });
            }
            pool->wait();
        } else {
            for (std::size_t c = 0; c < num_chunks; ++c) count_chunk(c);
        }

        std::vector<std::size_t> chunk_offs(num_chunks + 1, 0);
        for (std::size_t c = 0; c < num_chunks; ++c) {
            chunk_offs[c + 1] = chunk_offs[c] + chunk_counts[c];
        }
        const std::size_t total_kept = chunk_offs[num_chunks];

        std::vector<int64_t> kept_gids(total_kept);
        std::vector<int64_t> kept_tids(total_kept);

        auto scatter_chunk = [&](std::size_t c) {
            const std::size_t lo = c * chunk_size;
            const std::size_t hi = std::min(lo + chunk_size, P);
            std::size_t out = chunk_offs[c];
            for (std::size_t p = lo; p < hi; ++p) {
                if (keep_mask[p]) {
                    kept_gids[out] = result.gaussian_ids[p];
                    kept_tids[out] = result.tile_ids[p];
                    ++out;
                }
            }
        };
        if (pool != nullptr && W > 1 && num_chunks > 1) {
            for (std::size_t c = 0; c < num_chunks; ++c) {
                pool->submit([c, &scatter_chunk]() { scatter_chunk(c); });
            }
            pool->wait();
        } else {
            for (std::size_t c = 0; c < num_chunks; ++c) scatter_chunk(c);
        }

        result.gaussian_ids = std::move(kept_gids);
        result.tile_ids = std::move(kept_tids);

        // Recompute tiles_per_gaussian post-cull. Single serial pass over
        // total_kept (~234k); negligible.
        std::fill(result.tiles_per_gaussian.begin(), result.tiles_per_gaussian.end(), 0);
        for (const int64_t g : result.gaussian_ids) {
            ++result.tiles_per_gaussian[static_cast<std::size_t>(g)];
        }
    }

    return result;
}

}  // namespace gsplat_cpu
