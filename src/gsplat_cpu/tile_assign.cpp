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
    ThreadPool* pool,
    bool recompute_tiles_per_gaussian) {
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

    // iter-053: blocked dispatch (same rationale as iter-051/052). Strided
    // (m=w; m+=W) writes to tile_min_x/max_x/min_y/max_y/widths/heights/
    // tiles_per_gaussian — all per-Gaussian arrays — caused false sharing
    // between adjacent workers' writes to the same 64 B cache lines.
    if (pool != nullptr && W > 1 && M >= 4096) {
        const std::size_t chunk_m = (M + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, chunk_m, M, &bbox_one]() {
                const std::size_t lo = w * chunk_m;
                const std::size_t hi = std::min(lo + chunk_m, M);
                for (std::size_t m = lo; m < hi; ++m) bbox_one(m);
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
        const std::size_t chunk_m = (M + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, chunk_m, M, &scatter_one]() {
                const std::size_t lo = w * chunk_m;
                const std::size_t hi = std::min(lo + chunk_m, M);
                for (std::size_t m = lo; m < hi; ++m) scatter_one(m);
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

        // iter-042: precompute per-Gaussian m2_thresh and reformulate the
        // per-pair check to avoid both the div-by-det AND the std::exp on
        // every pair. Comparison
        //   opacity * exp(-0.5 * num/det) >= contrib_floor
        // is equivalent to
        //   num <= det * m2_thresh_g       where m2_thresh_g = -2 ln(contrib_floor / opacity_g)
        // for opacity > contrib_floor, and an unconditional drop when
        // opacity <= contrib_floor (the Gaussian can never contribute even
        // at its peak alpha). We pay 1 log per Gaussian (M = ~234k, parallel)
        // and save 1 div + 1 exp per pair (P = ~294k, parallel) -
        // approximately 60% of the per-pair cycles.
        std::vector<float> m2_thresh_g(M, 0.0f);
        std::vector<uint8_t> opacity_above_floor(M, 0);
        {
            auto precompute_one = [&](std::size_t m) {
                const float op = opacities[m];
                if (op > contrib_floor) {
                    m2_thresh_g[m] = -2.0f * std::log(contrib_floor / op);
                    opacity_above_floor[m] = 1;
                } else {
                    m2_thresh_g[m] = 0.0f;
                    opacity_above_floor[m] = 0;
                }
            };
            if (pool != nullptr && W > 1 && M >= 4096) {
                const std::size_t chunk_m = (M + W - 1) / W;
                for (std::size_t w = 0; w < W; ++w) {
                    pool->submit([w, chunk_m, M, &precompute_one]() {
                        const std::size_t lo = w * chunk_m;
                        const std::size_t hi = std::min(lo + chunk_m, M);
                        for (std::size_t m = lo; m < hi; ++m) precompute_one(m);
                    });
                }
                pool->wait();
            } else {
                for (std::size_t m = 0; m < M; ++m) precompute_one(m);
            }
        }

        // iter-053: cull dispatch changed from strided to block-chunked.
        // The strided pattern wrote keep_mask[p] (1 byte each) at p, p+W,
        // p+2W with W=10 — every 64 B cache line of keep_mask was touched
        // by every worker, causing intense cross-core line invalidation
        // on each store. Block-chunked dispatch makes each worker own a
        // contiguous span of keep_mask -> zero cross-core line traffic,
        // and hw-prefetcher-friendly sequential reads of gaussian_ids/
        // tile_ids.
        const std::size_t chunk_p = (P + W - 1) / W;
        auto cull_stripe = [&](std::size_t w) {
            const std::size_t lo = w * chunk_p;
            const std::size_t hi = std::min(lo + chunk_p, P);
            for (std::size_t p = lo; p < hi; ++p) {
                const int64_t g = result.gaussian_ids[p];
                if (!opacity_above_floor[static_cast<std::size_t>(g)]) {
                    keep_mask[p] = 0;
                    continue;
                }
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
                const float num =
                    c * dx_c * dx_c - 2.0f * b * dx_c * dy_c + a * dy_c * dy_c;
                const float scaled_thresh =
                    det * m2_thresh_g[static_cast<std::size_t>(g)];
                keep_mask[p] = (num <= scaled_thresh) ? 1 : 0;
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

        // iter-046: skip recomputing tiles_per_gaussian post-cull when
        // recompute_tiles_per_gaussian == false. render_full's fast path
        // never reads ta.tiles_per_gaussian (sort/blend don't need it),
        // and pipeline.py's standalone tile_assign caller is the only
        // consumer — that path uses recompute_tiles_per_gaussian=true so
        // verify_stage's bit-identity check still passes. Saves a 234k
        // serial increment loop in the hot path.
        if (recompute_tiles_per_gaussian) {
            std::fill(result.tiles_per_gaussian.begin(),
                      result.tiles_per_gaussian.end(), 0);
            for (const int64_t g : result.gaussian_ids) {
                ++result.tiles_per_gaussian[static_cast<std::size_t>(g)];
            }
        }
    }

    return result;
}

}  // namespace gsplat_cpu
