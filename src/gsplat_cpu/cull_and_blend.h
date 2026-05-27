#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

struct CullAndBlendResult {
    std::vector<float> image;        // H * W * 3, row-major. Empty when caller
                                     // supplies image_out_external (iter-049).
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
//
// iter-049: optional `image_out_external` lets the caller supply a
// pre-allocated (and pre-zeroed) buffer of size H*W*3. When non-null, the
// kernel writes pixels directly into it and the returned result.image stays
// empty — eliminates a ~3 MB std::vector alloc+zero plus the same-sized
// memcpy out of the C++ stage. The Python wrapper (`render_full_py`) zeros
// the pybind numpy buffer once and threads it through.
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
    ThreadPool& pool,
    float* image_out_external = nullptr);

}  // namespace gsplat_cpu
