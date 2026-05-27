#pragma once

// Vectorised exp(x) for x in [-30, 0] tuned for Gaussian-splat alpha weights.
// Bit-identical with std::exp only to ~6e-6 relative; well above the PSNR
// budget (>60 dB North Star) when used in the blend inner loop.
//
// Range reduction:  exp(x) = 2^n * exp(r),  n = round(x * log2(e)),
//                   r = x - n * ln(2)         with r in [-ln2/2, ln2/2]
// Polynomial:       degree-5 Maclaurin of exp(r)
// 2^n:              bit-twiddle: float_bits = (n + 127) << 23
//
// Apple Silicon has FMA + 4-wide NEON; the kernel becomes ~6 fmas + 1
// cvt + 1 shift per 4 elements ≈ 1.5 cycles / element.

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

}  // namespace gsplat_cpu
