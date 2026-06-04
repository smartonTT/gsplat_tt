// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared SFPU microblock-cull math for tile-local L1 records (iter 61 fuse).
// Included from alpha_blend_compute_mb.cpp when MB_FUSE_TILE_L1_CULL=1.

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

namespace tile_l1_cull_sfpu {

constexpr uint32_t BATCH = 32;
constexpr uint32_t NUM_MB = 32;
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;

constexpr uint32_t DR_BOX_OX = 0 * 32;
constexpr uint32_t DR_BOX_OY = 1 * 32;
constexpr uint32_t DR_KEEP   = 2 * 32;
constexpr uint32_t DR_QV     = 3 * 32;
constexpr uint32_t DR_QH     = 4 * 32;

#ifdef TRISC_MATH

template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_x(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat u_c = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    { vFloat uh = u_c + vFloat(8.0f); vFloat z = 0.0f; vec_min_max(z, u_c); vec_min_max(u_c, uh); }
    vFloat v_lo = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    vFloat v_hi = v_lo + vFloat(4.0f);
    vFloat ra = approx_recip(cov_a);
    ra = ra * (vFloat(2.0f) - cov_a * ra);
    ra = ra * (vFloat(2.0f) - cov_a * ra);
    vFloat vs = (cov_b * u_c) * ra;
    vec_min_max(v_lo, vs);
    vec_min_max(vs, v_hi);
    dst_reg[DR_QV + V] = cov_c * (u_c * u_c) + (vFloat(-2.0f) * cov_b) * (u_c * vs) +
                         cov_a * (vs * vs);
}

template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_y(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat v_c = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    { vFloat vh = v_c + vFloat(4.0f); vFloat z = 0.0f; vec_min_max(z, v_c); vec_min_max(v_c, vh); }
    vFloat u_lo = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    vFloat u_hi = u_lo + vFloat(8.0f);
    vFloat rc = approx_recip(cov_c);
    rc = rc * (vFloat(2.0f) - cov_c * rc);
    rc = rc * (vFloat(2.0f) - cov_c * rc);
    vFloat us = (cov_b * v_c) * rc;
    vec_min_max(u_lo, us);
    vec_min_max(us, u_hi);
    dst_reg[DR_QH + V] = cov_c * (us * us) + (vFloat(-2.0f) * cov_b) * (us * v_c) +
                         cov_a * (v_c * v_c);
}

template <uint32_t V>
__attribute__((noinline, noipa)) void cull_combine(
    uint32_t keep_base, uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t thr_bits, bool cull_disabled, uint32_t pos_base) {
    using namespace sfpi;
    (void)pos_base;
    namespace cs = ckernel::sfpu;
    vFloat qmin = dst_reg[DR_QV + V];
    { vFloat qh = dst_reg[DR_QH + V]; vec_min_max(qmin, qh); }
    if (cull_disabled) { qmin = vFloat(0.0f); }
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    { vFloat det_floor = 1e-6f; vec_min_max(det_floor, det); }
    vFloat thr = cs::Converter::as_float(thr_bits);
    vFloat scaled = det * thr;
    vFloat keepv = 0.0f;
    v_if(qmin <= scaled) { keepv = vFloat(1.0f); } v_endif;
    dst_reg[keep_base + V] = keepv;
}
#endif  // TRISC_MATH

template <uint32_t V>
inline void cull_phase_fx(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_face_x<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fx<V + 1>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    }
}

template <uint32_t V>
inline void cull_phase_fy(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_face_y<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fy<V + 1>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    }
}

template <uint32_t V>
inline void cull_phase_combine(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* thr,
    bool cull_disabled) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_combine<V>(keep_base, a[V], b[V], c[V], thr[V], cull_disabled, pos_base)));
        }
        cull_phase_combine<V + 1>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
    }
}

inline void cull_dispatch(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits, bool cull_disabled) {
    cull_phase_fx<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_fy<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_combine<0>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
}

inline uint32_t f_to_u32(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

inline const uint32_t* l1_splat_words(uint32_t buck, uint32_t g) {
    return reinterpret_cast<const uint32_t*>(
        buck + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
}

inline uint32_t thr_bits_from_l1(const uint32_t* rec, float contrib_floor) {
    constexpr float kUnormInv = 1.0f / 65535.0f;
    const float opf = static_cast<float>(rec[6] & 0xffffu) * kUnormInv;
    float thrf;
    if (opf <= contrib_floor) {
        thrf = -1.0f;
    } else {
        thrf = -2.0f * __builtin_logf(contrib_floor / opf);
    }
    return f_to_u32(thrf);
}

}  // namespace tile_l1_cull_sfpu
