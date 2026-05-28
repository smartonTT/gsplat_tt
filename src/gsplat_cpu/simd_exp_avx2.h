#pragma once

#include "gsplat_cpu/simd_config.h"

#if GSPLAT_HAS_AVX2

#include <immintrin.h>

namespace gsplat_cpu {

// Degree-4 Maclaurin exp(r), ~5e-5 relative error — mirrors simd_exp_f32x4_fast.
inline __m128 simd_exp_f32x4_fast_avx2(__m128 x) {
    const __m128 LOG2E = _mm_set1_ps(1.44269504088896f);
    const __m128 LN2   = _mm_set1_ps(0.69314718055994530942f);

    const __m128 y = _mm_mul_ps(x, LOG2E);
    const __m128 fn = _mm_round_ps(y, _MM_FROUND_CUR_DIRECTION);
    const __m128i n = _mm_cvtps_epi32(fn);

    const __m128 r = _mm_fnmadd_ps(fn, LN2, x);

    const __m128 r2 = _mm_mul_ps(r, r);
    const __m128 p0 = _mm_fmadd_ps(r, _mm_set1_ps(1.0f), _mm_set1_ps(1.0f));
    __m128 p1 = _mm_fmadd_ps(r, _mm_set1_ps(0.166666666667f), _mm_set1_ps(0.5f));
    p1 = _mm_fmadd_ps(r2, _mm_set1_ps(0.041666666667f), p1);
    const __m128 p = _mm_fmadd_ps(r2, p1, p0);

    const __m128i two_n_bits = _mm_slli_epi32(_mm_add_epi32(n, _mm_set1_epi32(127)), 23);
    const __m128 two_n = _mm_castsi128_ps(two_n_bits);

    return _mm_mul_ps(two_n, p);
}

}  // namespace gsplat_cpu

#endif  // GSPLAT_HAS_AVX2
