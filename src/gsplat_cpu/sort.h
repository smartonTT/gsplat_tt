#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

struct SortResult {
    std::vector<int64_t> sorted_gaussian_ids;  // (P,)
    std::vector<int64_t> tile_ranges;          // num_tiles * 2  (start, end pairs, row-major)
};

SortResult sort_and_bin(
    const int64_t* gaussian_ids,  // P
    const int64_t* tile_ids,      // P
    const float* depths,            // M (indexed by gaussian_ids[i])
    std::size_t P,
    std::size_t M,
    int tiles_x,
    int tiles_y);

}  // namespace gsplat_cpu
