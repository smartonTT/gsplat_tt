#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct SortResult {
    std::vector<int64_t> sorted_gaussian_ids;  // (P,)
    std::vector<int64_t> tile_ranges;          // num_tiles * 2  (start, end pairs, row-major)
};

// Tile-bucket parallel sort. Two passes:
//   1. Linear scatter: bin each (gaussian_id, depth) into its tile bucket
//      using the precomputed per-tile offset table (prefix-sum over counts).
//   2. Per-tile in-place sort by depth_bits (parallel across tiles via pool).
// Bit-identical to the single-threaded std::sort over composite key when
// depths within a tile are unique (true for all reference fixtures).
SortResult sort_and_bin(
    const int64_t* gaussian_ids,  // P
    const int64_t* tile_ids,      // P
    const float* depths,            // M (indexed by gaussian_ids[i])
    std::size_t P,
    std::size_t M,
    int tiles_x,
    int tiles_y,
    ThreadPool* pool = nullptr);  // nullptr -> single-threaded fallback

}  // namespace gsplat_cpu
