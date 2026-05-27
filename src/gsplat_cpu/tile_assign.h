#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

struct TileAssignResult {
    std::vector<int64_t> gaussian_ids;        // (P,)
    std::vector<int64_t> tile_ids;            // (P,)
    std::vector<int64_t> tiles_per_gaussian;  // (M,)
};

TileAssignResult tile_assign(
    const float* means_2d,  // M * 2
    const float* radii,     // M * 2  (rx, ry per row)
    std::size_t M,
    int image_height,
    int image_width,
    int tile_size,
    const float* covs_2d,  // nullable; M * 4 (a, b, b, c) when provided
    const float* opacities,  // nullable; M
    float contrib_floor);

}  // namespace gsplat_cpu
