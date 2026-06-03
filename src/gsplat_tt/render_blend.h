#pragma once

#include <cstddef>
#include <cstdint>

#include "gsplat_cpu/cull_and_blend.h"  // gsplat_cpu::CullAndBlendResult

namespace gsplat_cpu {
class ThreadPool;
}

namespace gsplat_tt {

// amendment-003: TT blend stage invoked from the fused C++ render loop
// (render_full_py, blend_mode == 1). Same ABI as gsplat_cpu::cull_and_blend so
// it is a drop-in inside render_full. The microblock cull runs on CPU and the
// kept-Gaussian-per-microblock structure feeds the device kernel.
//
// blend_mode selects the implementation:
//   1 = CPU reference blend from the microblock basis payload (validates the
//       basis math + coeff_table/mb_header/mb_stream layout the device kernel
//       consumes; an oracle for debugging the SFPU kernel).
//   2 = TT device mb-major 4x8 SFPU kernel (the goal).
//   other = delegate to gsplat_cpu::cull_and_blend (bit-identical fallback).
gsplat_cpu::CullAndBlendResult render_blend_tt(
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
    gsplat_cpu::ThreadPool& pool,
    float* image_out_external,
    bool cull_disabled,
    float transmittance_threshold,
    int blend_mode);

}  // namespace gsplat_tt
