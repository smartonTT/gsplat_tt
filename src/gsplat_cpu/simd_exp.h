#pragma once

// Vectorised exp(x) for x in [-30, 0] tuned for Gaussian-splat alpha weights.
//
//   simd_exp_f32x4       degree-5 Maclaurin, ~6e-6 relative error
//                        (legacy, kept for non-blend callers).
//
//   simd_exp_f32x4_fast  degree-4 Maclaurin, ~5e-5 relative error,
//                        used by the blend SIMD path. One FMA shorter
//                        polynomial chain. Now that iter-035 removed the
//                        per-Gaussian max_t barrier, the simd_exp chain
//                        is on the critical path between Gaussians; each
//                        FMA we drop directly shortens the dependency
//                        chain. Error budget: alpha is clipped to 0.99
//                        and the per-microblock max_t gate is 1e-4, so a
//                        5e-5 fractional perturbation is below quantisation.
//
// Range reduction:  exp(x) = 2^n * exp(r),  n = round(x * log2(e)),
//                   r = x - n * ln(2)         with r in [-ln2/2, ln2/2]
// 2^n:              bit-twiddle: float_bits = (n + 127) << 23

#include <arm_neon.h>

namespace gsplat_cpu {

inline float32x4_t simd_exp_f32x4(float32x4_t x) {
    const float32x4_t LOG2E = vdupq_n_f32(1.44269504088896f);
    const float32x4_t LN2   = vdupq_n_f32(0.69314718055994530942f);

    const float32x4_t y = vmulq_f32(x, LOG2E);
    const int32x4_t   n = vcvtnq_s32_f32(y);
    const float32x4_t fn = vcvtq_f32_s32(n);

    const float32x4_t r = vfmsq_f32(x, fn, LN2);

    float32x4_t p = vdupq_n_f32(0.008333333333f);   // 1/120
    p = vfmaq_f32(vdupq_n_f32(0.041666666667f), p, r);  // 1/24 + r/120
    p = vfmaq_f32(vdupq_n_f32(0.166666666667f), p, r);  // 1/6 + r*(...)
    p = vfmaq_f32(vdupq_n_f32(0.5f),            p, r);  // 1/2 + r*(...)
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);  // 1   + r*(...)
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);  // 1   + r*(...)

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

    float32x4_t p = vdupq_n_f32(0.041666666667f);              // 1/24
    p = vfmaq_f32(vdupq_n_f32(0.166666666667f), p, r);          // 1/6 + r/24
    p = vfmaq_f32(vdupq_n_f32(0.5f),            p, r);          // 1/2 + r*(...)
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);          // 1   + r*(...)
    p = vfmaq_f32(vdupq_n_f32(1.0f),            p, r);          // 1   + r*(...)

    const int32x4_t two_n_bits =
        vshlq_n_s32(vaddq_s32(n, vdupq_n_s32(127)), 23);
    const float32x4_t two_n = vreinterpretq_f32_s32(two_n_bits);

    return vmulq_f32(two_n, p);
}

}  // namespace gsplat_cpu
