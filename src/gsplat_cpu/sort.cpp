#include "gsplat_cpu/sort.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

SortResult sort_and_bin(
    const int64_t* gaussian_ids,
    const int64_t* tile_ids,
    const float* depths,
    const std::size_t P,
    const std::size_t M,
    const int tiles_x,
    const int tiles_y,
    ThreadPool* pool) {
    (void)M;

    const int num_tiles = tiles_x * tiles_y;
    SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (P == 0) {
        return result;
    }

    // Pass 1a: count per-tile occupancy.
    std::vector<int64_t> counts(static_cast<std::size_t>(num_tiles), 0);
    for (std::size_t i = 0; i < P; ++i) {
        const int64_t t = tile_ids[i];
        ++counts[static_cast<std::size_t>(t)];
    }

    // Pass 1b: prefix-sum into starts. For tile_ranges output we match the
    // numpy reference convention: populated tiles get (prefix_start, prefix_end);
    // empty tiles stay at (0, 0). The internal `starts` array is the true prefix
    // sum used for scatter regardless of the published tile_ranges values.
    std::vector<int64_t> starts(static_cast<std::size_t>(num_tiles) + 1, 0);
    for (int t = 0; t < num_tiles; ++t) {
        const int64_t c = counts[static_cast<std::size_t>(t)];
        starts[static_cast<std::size_t>(t) + 1] = starts[static_cast<std::size_t>(t)] + c;
        if (c > 0) {
            result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] = starts[static_cast<std::size_t>(t)];
            result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] = starts[static_cast<std::size_t>(t) + 1];
        }
    }

    // Pass 2: scatter (gaussian_id, depth_bits) into per-tile slots in input
    // order. depth_bits is stored as a uint32 in `packed_keys`; the parallel
    // `packed_ids` array holds the matching gaussian_id. Scatter preserves
    // tile_assign's input order, which is what the numpy reference's stable
    // radix sort uses as the tie-break order for equal depth_bits.
    std::vector<uint32_t> packed_keys(P);
    std::vector<int64_t> packed_ids(P);
    std::vector<int64_t> cursors = starts;
    for (std::size_t i = 0; i < P; ++i) {
        const std::size_t t = static_cast<std::size_t>(tile_ids[i]);
        const int64_t g = gaussian_ids[i];
        const uint32_t db = std::bit_cast<uint32_t>(depths[static_cast<std::size_t>(g)]);
        const std::size_t pos = static_cast<std::size_t>(cursors[t]);
        packed_keys[pos] = db;
        packed_ids[pos] = g;
        cursors[t] = static_cast<int64_t>(pos + 1);
    }

    // Pass 3: per-tile STABLE sort by depth_bits (parallel across tiles via pool).
    // Stable is required so tied depths preserve input order, matching numpy's
    // radix argsort tie-breaking exactly.
    auto sort_one_tile = [&](int t) {
        const std::size_t lo = static_cast<std::size_t>(starts[static_cast<std::size_t>(t)]);
        const std::size_t hi = static_cast<std::size_t>(starts[static_cast<std::size_t>(t) + 1]);
        const std::size_t n = hi - lo;
        if (n <= 1) return;
        // Index permutation over [0, n). std::stable_sort on indices is stable
        // in the input order, which matches numpy radix-sort tie-breaking.
        std::vector<uint32_t> idx(n);
        for (std::size_t k = 0; k < n; ++k) idx[k] = static_cast<uint32_t>(k);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](uint32_t a, uint32_t b) {
                             return packed_keys[lo + a] < packed_keys[lo + b];
                         });
        std::vector<uint32_t> out_keys(n);
        std::vector<int64_t> out_ids(n);
        for (std::size_t k = 0; k < n; ++k) {
            out_keys[k] = packed_keys[lo + idx[k]];
            out_ids[k] = packed_ids[lo + idx[k]];
        }
        for (std::size_t k = 0; k < n; ++k) {
            packed_keys[lo + k] = out_keys[k];
            packed_ids[lo + k] = out_ids[k];
        }
    };

    if (pool != nullptr && num_tiles > 1) {
        for (int t = 0; t < num_tiles; ++t) {
            pool->submit([t, &sort_one_tile]() { sort_one_tile(t); });
        }
        pool->wait();
    } else {
        for (int t = 0; t < num_tiles; ++t) {
            sort_one_tile(t);
        }
    }

    // Pass 4: gather final gaussian_ids in sorted order.
    result.sorted_gaussian_ids.resize(P);
    for (std::size_t i = 0; i < P; ++i) {
        result.sorted_gaussian_ids[i] = packed_ids[i];
    }

    return result;
}

}  // namespace gsplat_cpu
