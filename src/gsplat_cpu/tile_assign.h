#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct TileAssignResult {
    std::vector<int64_t> gaussian_ids;        // (P,)
    std::vector<int64_t> tile_ids;            // (P,)
    std::vector<int64_t> tiles_per_gaussian;  // (M,)
};

// recompute_tiles_per_gaussian: when true (default for Python bindings that
// expose tiles_per_gaussian), perform a final bincount over the post-cull
// gaussian_ids. The render_full fast path passes false to skip the ~230k
// serial increment loop since no downstream stage reads it.
TileAssignResult tile_assign(
    const float* means_2d,  // M * 2
    const float* radii,     // M * 2  (rx, ry per row)
    std::size_t M,
    int image_height,
    int image_width,
    int tile_size,
    const float* covs_2d,  // nullable; M * 4 (a, b, b, c) when provided
    const float* opacities,  // nullable; M
    float contrib_floor,
    ThreadPool* pool = nullptr,
    bool recompute_tiles_per_gaussian = true);

}  // namespace gsplat_cpu
