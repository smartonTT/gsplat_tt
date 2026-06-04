// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar conic coeffs A,B,C from raw 2D covariance (host / reader / compute
// scalar paths). Matches alpha_blend_compute_mb DEVCONIC: det floor 1e-6,
// approx_recip + 2 Newton-Raphson iters, A=-0.5*cov_c/det, B=cov_b/det,
// C=-0.5*cov_a/det.

#pragma once

#include <cstdint>

namespace mb_cov_preconic {

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline uint32_t f_to_bits(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

inline void cov_to_abc(
    float cov_a, float cov_b, float cov_c, float& out_a, float& out_b, float& out_c) {
    float det = cov_a * cov_c - cov_b * cov_b;
    if (det < 1e-6f) {
        det = 1e-6f;
    }
    float inv = 1.0f / det;
    inv = inv * (2.0f - det * inv);
    inv = inv * (2.0f - det * inv);
    out_a = -0.5f * (cov_c * inv);
    out_b = cov_b * inv;
    out_c = -0.5f * (cov_a * inv);
}

inline void cov_bits_to_abc(
    uint32_t cov_a_bits, uint32_t cov_b_bits, uint32_t cov_c_bits,
    uint32_t& a_bits, uint32_t& b_bits, uint32_t& c_bits) {
    float a, b, c;
    cov_to_abc(
        bits_to_f(cov_a_bits), bits_to_f(cov_b_bits), bits_to_f(cov_c_bits), a, b, c);
    a_bits = f_to_bits(a);
    b_bits = f_to_bits(b);
    c_bits = f_to_bits(c);
}

}  // namespace mb_cov_preconic
