#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct MicroblockCullResult {
    std::vector<int64_t> mb_header;   // num_tiles * 32 * 2  (row-major: tile, m, [offset|count])
    std::vector<int64_t> mb_stream;   // L_prime (sum of all counts)
    int64_t pairs_in;
    int64_t pairs_out;                // == L_prime
    int64_t pairs_dropped_in_all_mb;  // for drop_pct stat
};

MicroblockCullResult microblock_cull(
    const float* means_2d,            // M * 2
    const float* covs_2d,             // M * 4 (a, b, b, c)
    const float* opacities,           // M
    const int64_t* sorted_gaussian_ids,  // P
    const int64_t* tile_ranges,       // num_tiles * 2 (start, end)
    std::size_t M,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,                    // 32
    float mb_contrib_floor,
    ThreadPool& pool);

}  // namespace gsplat_cpu
