#pragma once

#include <cstddef>
#include <cstdint>

#include "gsplat_cpu/blend.h"

namespace gsplat_cpu {

class ThreadPool;

BlendResult blend_microblock(
    const float* means_2d,            // M * 2
    const float* covs_2d,             // M * 4 (a, b, b, c)
    const float* colors,              // M * 3
    const float* opacities,           // M
    const int64_t* mb_header,         // num_tiles * 32 * 2 (offset, count)
    const int64_t* mb_stream,         // L_prime
    std::size_t M,
    std::size_t L_prime,
    int image_height,
    int image_width,
    int tile_size,                    // 32
    ThreadPool& pool);

}  // namespace gsplat_cpu
