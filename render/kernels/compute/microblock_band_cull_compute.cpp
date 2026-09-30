// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Band-extent SFPU microblock cull (task #59) for the tile-local L1 cull.
//
// WHY
// ---
// microblock_cull_compute.cpp put ONE gaussian in each SFPU vector (32 lanes =
// the 32 microblocks) and evaluated a box-constrained Mahalanobis minimum per
// lane: ~140 SFPU instructions plus ~100 scalar TRISC instructions per pair
// (runtime-immediate coefficient loads, noinline calls, record decode). At
// ~3.4 M pairs/frame that was ~15 ms/view of TRISC time, as long as the blend.
//
// This kernel puts one GAUSSIAN in each LANE instead (32 gaussians per vector,
// 4 vectors per coefficient tile) and walks the 8 microblock rows ("bands").
// Every coefficient is a per-lane value the reader transposed into a tile, so
// every constant in the SFPU code is a compile-time immediate and there is no
// per-pair scalar work on the TRISC.
//
// MATH (per lane; u = x - mx, v = y - my in tile-local pixels)
// ------------------------------------------------------------
//   m2(u,v) = ci_a u^2 + 2 ci_b u v + ci_c v^2,  ci = {-2A, -B, -2C}
//   keep microblock (j,k) iff the ellipse m2 <= t meets its pixel-centre box
//   [8k+0.5, 8k+7.5] x [4j+0.5, 4j+3.5], t = 2 ln(op/floor) + kThrMargin.
// That is the same test as the box-min cull; here it is organised by rows:
//   * For a fixed v the ellipse spans u in R v -/+ S sqrt(P - Q v^2), with
//     R = -ci_b/ci_a, S = 1/ci_a, P = t ci_a, Q = det.
//   * Band j (v in [v0, v1]) meets the ellipse iff P - Q vc^2 >= 0 with
//     vc = clamp(0, v0, v1).
//   * The right edge R v + S sqrt(.) is concave in v, maximal at the ellipse's
//     right-most point vR = -ci_b sqrt(t / (det ci_c)); the band's right-most
//     extent is at vr = clamp(vR, v0, v1). The left edge mirrors it at
//     vl = clamp(-vR, v0, v1).
//   * Column k is kept iff that extent meets [8k+0.5, 8k+7.5] - mx. Done on
//     squares, no sqrt:  R vr + S sqrt(dr) >= c  <=>  n|n| + S^2 dr >= 0 with
//     n = R vr - c  (and the mirrored test on the left edge).
// Keep bits are accumulated as 2^23 + mask16 in fp32 (exact), so the packed
// tile carries the 16-bit halves in the low mantissa bits and the writer needs
// no float code: mask = (lo & 0xffff) | (hi << 16).
//
// Superset: the only approximations are the SFPU log (reads ln up to 0.0035
// low) and fp32 rounding (~1e-6 relative in m2). kThrMargin (0.05 in m2) keeps
// the mask a superset of every pixel centre the blend's per-pixel floor can
// keep; extra keeps are zeroed by that floor, so the image is unchanged.
// Model + tests: opt/golden-check/cull_mask_check.py band_keep_f32,
// tests/spec/test_band_cull.py.
//
// DEST (fp32, dst_reg units, 32 vectors per tile):
//   tile 0      coefficient tile, group g field f at vector 6g+f (from reader)
//   tiles 1-2   per-lane derived values, group g at 16g + D_*
//   tile 3      per-band scratch, band j at 4j + B_*
//   tile 4      output, group g: vector 2g = 2^23 + bits 0-15, 2g+1 = bits 16-31
// Every DEST value is stored at least ~200 SFPU instructions before it is read
// back (all groups' derive first, then per group all bands' rows, then all
// bands' columns): a store followed closely by a load of the same DEST row
// returned stale data in microblock_cull_compute.cpp.

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

namespace {

constexpr uint32_t CB_COEFF      = 2;   // reader -> compute: fp32 coefficient tiles
constexpr uint32_t CB_CULL_COUNTS= 3;   // per-subchunk [L, tx_pix, ty_pix, base]
constexpr uint32_t CB_CORE_TILES = 7;   // subchunk count handoff from the reader
constexpr uint32_t CB_KEEP       = 16;  // compute -> writer: fp32 mask tiles

constexpr uint32_t GROUPS = 4;              // 32-gaussian groups per coefficient tile
constexpr uint32_t BATCH  = 32u * GROUPS;   // gaussians per coefficient tile

// Coefficient fields (vector 6g+f of the coefficient tile). Must match
// reader_tile_l1_cull.cpp.
constexpr uint32_t F_A = 0, F_B = 1, F_C = 2, F_OPQ = 3, F_MX = 4, F_MY = 5;
// Derived per-lane values (DR_DER + 16g + D_*).
constexpr uint32_t D_P = 0, D_NQ = 1, D_S2 = 2, D_R = 3, D_NR = 4, D_VR = 5, D_NVR = 6,
                   D_VB = 7, D_NC0 = 8, D_C07 = 9;
// Per-band scratch (DR_BAND + 4j + B_*).
constexpr uint32_t B_DR = 0, B_DL = 1, B_NW = 2, B_NZ = 3;

constexpr uint32_t DR_IN   = 0 * 32;
constexpr uint32_t DR_DER  = 1 * 32;
constexpr uint32_t DR_BAND = 3 * 32;
constexpr uint32_t DR_OUT  = 4 * 32;

// m2 slack added to thr so the mask stays a superset of the blend's per-pixel
// floor (see microblock_cull_compute.cpp and test_pixel_floor_seams.py).
constexpr float kThrMargin = 0.05f;
constexpr float kTwo23 = 8388608.0f;  // output bias and opacity field bias

#ifdef TRISC_MATH
// sqrt(x), x >= 0, ~23-bit (tt-llk ckernel_sfpu_sqrt.h "SQRT_23-bits" with its
// programmable constants inlined, so the kernel needs no SFPU constant init).
sfpi_inline sfpi::vFloat band_sqrt(sfpi::vFloat x) {
    using namespace sfpi;
    vInt i = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(vInt(0x5f1110a0) - i);
    vFloat xy = x * y;
    vFloat c = (-y) * xy;
    y = y * (vFloat(2.2825186f) + c * (vFloat(2.2533049f) + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) - y * xy;
    return one_minus_xyy * (xy * 0.5f) + xy;
}

// 1/x: hardware seed + two Newton steps (as microblock_cull_compute.cpp).
sfpi_inline sfpi::vFloat band_recip(sfpi::vFloat x) {
    using namespace sfpi;
    vFloat r = approx_recip(x);
    r = r * (vFloat(2.0f) - x * r);
    r = r * (vFloat(2.0f) - x * r);
    return r;
}

// Phase A: per-lane derived values of group G.
template <uint32_t G>
__attribute__((noinline)) void band_derive(uint32_t inv_floor_bits) {
    using namespace sfpi;
    constexpr uint32_t I = DR_IN + 6u * G;
    constexpr uint32_t D = DR_DER + 16u * G;
    // t = 2 ln(op/floor) + margin; the field holds 2^23 + q, op = q/65535.
    vFloat op = (vFloat(dst_reg[I + F_OPQ]) - vFloat(kTwo23)) * vFloat(1.0f / 65535.0f);
    vFloat inv_floor = ckernel::sfpu::Converter::as_float(inv_floor_bits);
    vFloat lnr = ckernel::sfpu::_calculate_log_body_no_init_(op * inv_floor);
    vFloat t = lnr + lnr + vFloat(kThrMargin);

    vFloat ci_a = vFloat(dst_reg[I + F_A]) * vFloat(-2.0f);
    dst_reg[D + D_P] = t * ci_a;
    vFloat s = band_recip(ci_a);
    dst_reg[D + D_S2] = s * s;
    vFloat b = dst_reg[I + F_B];  // b = -ci_b
    vFloat r = b * s;             // R = -ci_b / ci_a
    dst_reg[D + D_R] = r;
    dst_reg[D + D_NR] = -r;

    vFloat ci_c = vFloat(dst_reg[I + F_C]) * vFloat(-2.0f);
    vFloat det = ci_a * ci_c - b * b;
    dst_reg[D + D_NQ] = -det;
    // vR = -ci_b sqrt(max(t,0) / (det ci_c)).
    vFloat rd = band_recip(det * ci_c);
    vFloat tz = 0.0f;
    vec_min_max(tz, t);  // t = max(t, 0)
    vFloat vr = b * band_sqrt(t * rd);
    dst_reg[D + D_VR] = vr;
    dst_reg[D + D_NVR] = -vr;

    vFloat mx = dst_reg[I + F_MX];
    dst_reg[D + D_NC0] = mx - vFloat(0.5f);  // -(column-0 left pixel centre - mx)
    dst_reg[D + D_C07] = vFloat(7.5f) - mx;  //  column-0 right pixel centre - mx
    dst_reg[D + D_VB] = vFloat(0.5f) - vFloat(dst_reg[I + F_MY]);  // band-0 top - my
}

// Phase B1: band J of group G -> the band's right/left extent terms.
template <uint32_t G, uint32_t J>
inline void band_row(void) {
    using namespace sfpi;
    constexpr uint32_t D = DR_DER + 16u * G;
    constexpr uint32_t S = DR_BAND + 4u * J;
    vFloat v0 = vFloat(dst_reg[D + D_VB]) + vFloat(static_cast<float>(4u * J));
    vFloat v1 = v0 + vFloat(3.0f);
    // vr = clamp(vR, v0, v1), vl = clamp(-vR, v0, v1)
    vFloat vr = dst_reg[D + D_VR];
    { vFloat a = v0; vec_min_max(a, vr); }
    { vFloat h = v1; vec_min_max(vr, h); }
    vFloat vl = dst_reg[D + D_NVR];
    { vFloat a = v0; vec_min_max(a, vl); }
    { vFloat h = v1; vec_min_max(vl, h); }
    // vc = clamp(0, v0, v1); the band meets the ellipse iff P - Q vc^2 >= 0.
    vFloat vc = 0.0f;
    vec_min_max(v0, vc);
    vec_min_max(vc, v1);
    vFloat dc = vFloat(dst_reg[D + D_NQ]) * (vc * vc) + vFloat(dst_reg[D + D_P]);
    vFloat dr = vFloat(dst_reg[D + D_S2]) *
                (vFloat(dst_reg[D + D_NQ]) * (vr * vr) + vFloat(dst_reg[D + D_P]));
    vFloat dl = vFloat(dst_reg[D + D_S2]) *
                (vFloat(dst_reg[D + D_NQ]) * (vl * vl) + vFloat(dst_reg[D + D_P]));
    v_if(dc < 0.0f) {
        dr = -1e30f;
        dl = -1e30f;
    }
    v_endif;
    dst_reg[S + B_DR] = dr;
    dst_reg[S + B_DL] = dl;
    dst_reg[S + B_NW] = vFloat(dst_reg[D + D_R]) * vr + vFloat(dst_reg[D + D_NC0]);
    dst_reg[S + B_NZ] = vFloat(dst_reg[D + D_NR]) * vl + vFloat(dst_reg[D + D_C07]);
}

// Phase B2: column K of band J -> keep bit 4J+K into acc (2^23 + 16-bit half).
template <uint32_t J, uint32_t K>
inline void band_col(sfpi::vFloat& acc) {
    using namespace sfpi;
    constexpr uint32_t S = DR_BAND + 4u * J;
    constexpr float kBit = static_cast<float>(1u << ((4u * J + K) & 15u));
    vFloat nw = dst_reg[S + B_NW];
    vFloat nz = dst_reg[S + B_NZ];
    if constexpr (K != 0u) {
        nw = nw - vFloat(static_cast<float>(8u * K));
        nz = nz + vFloat(static_cast<float>(8u * K));
    }
    vFloat tr = nw * sfpi::abs(nw) + vFloat(dst_reg[S + B_DR]);
    vFloat tl = nz * sfpi::abs(nz) + vFloat(dst_reg[S + B_DL]);
    vec_min_max(tr, tl);  // tr = min
    v_if(tr >= 0.0f) { acc = acc + vFloat(kBit); }
    v_endif;
}

template <uint32_t G, uint32_t J>
inline void band_rows_from(void) {
    if constexpr (J < 8u) {
        band_row<G, J>();
        band_rows_from<G, J + 1u>();
    }
}

// Bands [J, JEND) of the current 16-bit half into acc.
template <uint32_t J, uint32_t JEND>
inline void band_cols_from(sfpi::vFloat& acc) {
    if constexpr (J < JEND) {
        band_col<J, 0>(acc);
        band_col<J, 1>(acc);
        band_col<J, 2>(acc);
        band_col<J, 3>(acc);
        band_cols_from<J + 1u, JEND>(acc);
    }
}

// Phase B for group G: all 8 bands' rows (B1), then all columns (B2).
template <uint32_t G>
__attribute__((noinline)) void band_group(void) {
    using namespace sfpi;
    band_rows_from<G, 0>();
    vFloat lo = kTwo23;
    band_cols_from<0, 4>(lo);
    dst_reg[DR_OUT + 2u * G] = lo;
    vFloat hi = kTwo23;
    band_cols_from<4, 8>(hi);
    dst_reg[DR_OUT + 2u * G + 1u] = hi;
}

// GSPLAT cull_disabled: keep every microblock (the blend's per-pixel floor
// still drops what is below it).
__attribute__((noinline)) void band_keep_all(void) {
    using namespace sfpi;
    vFloat all = kTwo23 + 65535.0f;
    dst_reg[DR_OUT + 0] = all;
    dst_reg[DR_OUT + 1] = all;
    dst_reg[DR_OUT + 2] = all;
    dst_reg[DR_OUT + 3] = all;
    dst_reg[DR_OUT + 4] = all;
    dst_reg[DR_OUT + 5] = all;
    dst_reg[DR_OUT + 6] = all;
    dst_reg[DR_OUT + 7] = all;
}

inline void band_batch(uint32_t inv_floor_bits) {
    band_derive<0>(inv_floor_bits);
    band_derive<1>(inv_floor_bits);
    band_derive<2>(inv_floor_bits);
    band_derive<3>(inv_floor_bits);
    band_group<0>();
    band_group<1>();
    band_group<2>();
    band_group<3>();
}
#endif  // TRISC_MATH

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_mb_mask");
    const uint32_t floor_bits = get_arg_val<uint32_t>(1);
    const bool cull_disabled  = get_arg_val<uint32_t>(2) != 0;
    // inv_floor = 1/floor once per core (one soft-float divide, not per pair).
    float floor_f;
    __builtin_memcpy(&floor_f, &floor_bits, 4);
    const float inv_floor_f = (floor_f > 0.0f) ? (1.0f / floor_f) : 0.0f;
    uint32_t inv_floor_bits;
    __builtin_memcpy(&inv_floor_bits, &inv_floor_f, 4);
    (void)inv_floor_bits;

    cb_wait_front(CB_CORE_TILES, 1);
    const uint32_t num_work =
        reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
    cb_pop_front(CB_CORE_TILES, 1);

    init_sfpu(CB_COEFF, CB_KEEP);
    if (num_work == 0) {
        return;
    }

    for (uint32_t w = 0; w < num_work; w++) {
        cb_wait_front(CB_CULL_COUNTS, 1);
        const uint32_t L =
            reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CULL_COUNTS, 0))[0];
        cb_pop_front(CB_CULL_COUNTS, 1);

        const uint32_t nbatch = (L + BATCH - 1u) / BATCH;
        for (uint32_t b = 0; b < nbatch; b++) {
            cb_wait_front(CB_COEFF, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_COEFF);
            copy_tile(CB_COEFF, 0, DR_IN / 32);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            if (cull_disabled) {
                MATH((band_keep_all()));
            } else {
                MATH((band_batch(inv_floor_bits)));
            }
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_KEEP, 1);
            pack_tile(DR_OUT / 32, CB_KEEP);
            cb_push_back(CB_KEEP, 1);
            tile_regs_release();
            cb_pop_front(CB_COEFF, 1);
        }
    }
}
