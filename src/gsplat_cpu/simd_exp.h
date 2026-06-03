#pragma once

// Vectorised exp(x) for x in [-30, 0] tuned for Gaussian-splat alpha weights.
// Only compiled when NEON is active; scalar and AVX2 paths use their own
// implementations in blend_microblock.cpp / cull_and_blend.cpp.

#include "gsplat_cpu/simd_config.h"

#if GSPLAT_HAS_NEON

#include <arm_neon.h>

namespace gsplat_cpu {

inline float32x4_t simd_exp_f32x4(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(1.44269504088896f);
    const float32x4_t LN2   = vdupq_n_f32(0.69314718055994530942f);

    const float32x4_t y = vmulq_f32(x, LOG2E);
    const int32x4_t   n = vcvtnq_s32_f32(y);
    const float32x4_t fn = vcvtq_f32_s32(n);

    const float32x4_t r = vfmsq_f32(x, fn, LN2);

    float32x4_t p = vdupq_n_f32(0.008333333333f);
    p = vfmaq_f32(vdupq_n_f32(0.041666666667f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.166666666667f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f),            p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);

    const int32x4_t two_n_bits =
        vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    const float32x4_t two_n = vreinterpretq_f32_s32(two_n_bits);

    return vmulq_f32(two_n, p);
}

inline float32x4_t simd_exp_f32x4_fast(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(1.44269504088896f);
    const float32x4_t LN2   = vdupq_n_f32(0.69314718055994530942f);

    const float32x4_t y = vmulq_f32(x, LOG2E);
    const int32x4_t   n = vcvtnq_s32_f32(y);
    const float32x4_t fn = vcvtq_f32_s32(n);

    const float32x4_t r = vfmsq_f32(x, fn, LN2);

    const float32x4_t r2 = vmulq_f32(r, r);
    const float32x4_t p0 = vfmaq_f32(vdupq_n_f32(1.0f), r, vdupq_n_f32(1.0f));
    float32x4_t p1 = vfmaq_f32(vdupq_n_f32(0.5f), r, vdupq_n_f32(0.166666666667f));
    p1 = vfmaq_f32(p1, r2, vdupq_n_f32(0.041666666667f));
    const float32x4_t p = vfmaq_f32(p0, r2, p1);

    const int32x4_t two_n_bits =
        vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    const float32x4_t two_n = vreinterpretq_f32_s32(two_n_bits);

    return vmulq_f32(two_n, p);
}

}  // namespace gsplat_cpu

#endif  // GSPLAT_HAS_NEON
