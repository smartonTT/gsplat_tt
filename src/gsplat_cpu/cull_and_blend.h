#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct CullAndBlendResult {
    std::vector<float> image;        // H * W * 3, row-major
    int64_t pairs_in{0};             // total (Gaussian, tile) pairs in
    int64_t pairs_dropped_all_mb{0}; // Gaussians dropped in ALL 32 microblocks
    int64_t pairs_kept_per_mb{0};    // total surviving (Gaussian, microblock) pairs
};

// Fused microblock_cull + blend_microblock. The previous two-pass pipeline
// allocated a global mb_stream / mb_header buffer (~MB-scale), required two
// pybind calls and walked every tile's Gaussians twice. cull_and_blend runs
// both passes in a single per-tile task: walks the sorted Gaussian list,
// keeps an in-tile per-microblock kept-id buffer (small, hot in L1), and
// blends each microblock immediately into the output image. Saves global
// memory traffic and a pybind round-trip.
CullAndBlendResult cull_and_blend(
    const float* means_2d,
    const float* covs_2d,
    const float* colors,
    const float* opacities,
    const int64_t* sorted_gaussian_ids,
    const int64_t* tile_ranges,
    std::size_t M,
    std::size_t P,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int image_height,
    int image_width,
    float mb_contrib_floor,
    ThreadPool& pool);

}  // namespace gsplat_cpu
