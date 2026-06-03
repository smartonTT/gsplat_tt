#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct BlendResult {
    std::vector<float> image;  // H * W * 3, row-major, fp32, in [0, 1]
};

/// Write rendered image into \p image_out (H * W * 3, zeroed by caller).
void blend(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t M,
    std::size_t P,
    int image_height,
    int image_width,
    int tile_size,
    float* image_out,
    ThreadPool& pool);

BlendResult blend(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t M,
    std::size_t P,
    int image_height,
    int image_width,
    int tile_size,
    ThreadPool& pool);

}  // namespace gsplat_cpu
