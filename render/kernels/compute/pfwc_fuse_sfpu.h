// SPDX-License-Identifier: Apache-2.0
//
// Task #489 (GSPLAT_TT_PFWC_ACQ_FUSE, math thread): fewer pfwc acquires.
//
// Bit 0 (projection): steps 1-5 (world->camera transform, 1/tz, depth, mean_x,
// mean_y) were seven acquires of tile ops with 15 copy_tile and 9 packs. Here they
// are one acquire: 3 copy_tile (mx, my, mz -> tiles 0..2), run_xform (3 passes),
// the 1/tz pass and run_mean, 9 packs from DEST.
//   xform x3: tiles 0 mx, 1 my, 2 mz -> 3 tx, 4 ty, 5 tz, 6 tz (1/tz in place).
//   run_mean:  tiles 3 tx, 4 ty, 6 inv_tz -> 0 mean_x, 1 mean_y.
// Bit 1 (cov): cov_cam (pfwc_covcam_sfpu.h) and cov2d a / c + radii (S_AC) share
// one acquire: 6 cov3d copies, run (cc00..cc22 in tiles 0..5), copy inv_tz -> 6,
// tx -> 7, run_a, copy ty -> 7, run_c, then the radii tile ops on 6 / 7. cc00 and
// cc11 never leave DEST (S_AC was their only reader).
//   run_a: tiles 0 cc00, 2 cc02, 5 cc22, 6 inv_tz, 7 tx -> a in tile 0.
//   run_c: tiles 3 cc11, 4 cc12, 5 cc22, 6 inv_tz, 7 ty -> c in tiles 3 and 7,
//          a copied 0 -> 6 (inv_tz / ty are consumed).
//
// Every product (SFPMUL, + 0.0) and sum (SFPADD, * 1.0) is rounded on its own, in
// the op order of the tile-op path (mul_unary_tile, add_binary_tile, add_unary_tile,
// mul_binary_tile): bit-identical (tests/unit/test_pfwc_fuse_sfpu.cpp). Same
// conventions as pfwc_cov2d_sfpu.h: DEST rows, tile t at row 64t, sfpi::dst_reg++
// moves the counter 2 rows, call each run between SFPU start(0) and done.
#pragma once

#include <cstdint>

namespace pfwc_fuse {

using P = ckernel::p_sfpu;
constexpr uint32_t LOADI_UPPER = 8, LOADI_LOWER = 10;  // SFPLOADI_MOD0_UPPER / _LOWER
constexpr uint32_t TWO = 0x40000000u, PT3 = 0x3E99999Au;  // 2.0f, 0.3f
constexpr uint32_t T = 64;                               // DEST rows per tile

#define PFWC_FUSE_LD(l, t) TTI_SFPLOAD((l), 0, ckernel::ADDR_MOD_7, (t) * T)
#define PFWC_FUSE_ST(l, t) TTI_SFPSTORE((l), 0, ckernel::ADDR_MOD_7, (t) * T)
#define PFWC_FUSE_MUL(a, b, d) TTI_SFPMUL((a), (b), P::LCONST_0, (d), 0)
#define PFWC_FUSE_ADD(a, c, d) TTI_SFPADD((a), P::LCONST_1, (c), (d), 0)

template <uint32_t L>
inline void loadi(uint32_t bits) {
    TT_SFPLOADI(L, LOADI_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(L, LOADI_UPPER, bits >> 16);
}

// One output j of step 1: ((mx r_j0 + my r_j1) + mz r_j2) + t_j -> tile OUT
// (and tile OUT2 when OUT2 != OUT). Called three times (j = 0, 1, 2 -> tiles
// 3, 4, 5 + 6), each between SFPU start(0) and done.
template <uint32_t OUT, uint32_t OUT2>
inline void xform(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t t) {
    loadi<P::LREG3>(r0);
    loadi<P::LREG4>(r1);
    loadi<P::LREG5>(r2);
    loadi<P::LREG6>(t);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_FUSE_LD(P::LREG0, 0);
        PFWC_FUSE_LD(P::LREG1, 1);
        PFWC_FUSE_LD(P::LREG2, 2);
        PFWC_FUSE_MUL(P::LREG0, P::LREG3, P::LREG7);    // mx * r0
        PFWC_FUSE_MUL(P::LREG1, P::LREG4, P::LREG0);    // my * r1
        PFWC_FUSE_ADD(P::LREG7, P::LREG0, P::LREG7);
        PFWC_FUSE_MUL(P::LREG2, P::LREG5, P::LREG0);    // mz * r2
        PFWC_FUSE_ADD(P::LREG7, P::LREG0, P::LREG7);
        PFWC_FUSE_ADD(P::LREG7, P::LREG6, P::LREG7);    // + t
        PFWC_FUSE_ST(P::LREG7, OUT);
        if constexpr (OUT2 != OUT) PFWC_FUSE_ST(P::LREG7, OUT2);
        sfpi::dst_reg++;
    }
}

// mean_x = (tx inv) fx + cx -> tile 0, mean_y = (ty inv) fy + cy -> tile 1.
inline void run_mean(uint32_t fx, uint32_t cx, uint32_t fy, uint32_t cy) {
    loadi<P::LREG2>(fx);
    loadi<P::LREG3>(cx);
    loadi<P::LREG5>(fy);
    loadi<P::LREG6>(cy);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_FUSE_LD(P::LREG1, 6);                      // inv_tz
        PFWC_FUSE_LD(P::LREG0, 3);                      // tx
        PFWC_FUSE_MUL(P::LREG0, P::LREG1, P::LREG0);
        PFWC_FUSE_MUL(P::LREG0, P::LREG2, P::LREG0);
        PFWC_FUSE_ADD(P::LREG0, P::LREG3, P::LREG0);
        PFWC_FUSE_ST(P::LREG0, 0);
        PFWC_FUSE_LD(P::LREG4, 4);                      // ty
        PFWC_FUSE_MUL(P::LREG4, P::LREG1, P::LREG4);
        PFWC_FUSE_MUL(P::LREG4, P::LREG5, P::LREG4);
        PFWC_FUSE_ADD(P::LREG4, P::LREG6, P::LREG4);
        PFWC_FUSE_ST(P::LREG4, 1);
        sfpi::dst_reg++;
    }
}

// a = ((cc00 j00) j00 + ((cc02 j00) j02) 2) + (cc22 j02) j02 + 0.3 (run_ac's a).
inline void run_a(uint32_t fx, uint32_t nfx) {
    TTI_SFPLOADI(P::LREG0, LOADI_LOWER, TWO & 0xFFFF);
    TTI_SFPLOADI(P::LREG0, LOADI_UPPER, TWO >> 16);
    TTI_SFPLOADI(P::LREG7, LOADI_LOWER, PT3 & 0xFFFF);
    TTI_SFPLOADI(P::LREG7, LOADI_UPPER, PT3 >> 16);
    loadi<P::LREG6>(fx);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_FUSE_LD(P::LREG1, 6);                      // L1 = inv_tz
        PFWC_FUSE_MUL(P::LREG1, P::LREG6, P::LREG2);    // L2 = j00 = inv * fx
        PFWC_FUSE_LD(P::LREG3, 7);                      // tx
        loadi<P::LREG4>(nfx);
        PFWC_FUSE_MUL(P::LREG3, P::LREG4, P::LREG3);    // tx * -fx
        PFWC_FUSE_MUL(P::LREG3, P::LREG1, P::LREG3);    // * inv
        PFWC_FUSE_MUL(P::LREG3, P::LREG1, P::LREG3);    // L3 = j02
        PFWC_FUSE_LD(P::LREG4, 0);
        PFWC_FUSE_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_FUSE_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_FUSE_LD(P::LREG5, 2);
        PFWC_FUSE_MUL(P::LREG5, P::LREG2, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG0, P::LREG5);
        PFWC_FUSE_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_FUSE_LD(P::LREG5, 5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_FUSE_ADD(P::LREG4, P::LREG7, P::LREG4);
        PFWC_FUSE_ST(P::LREG4, 0);
        sfpi::dst_reg++;
    }
}

// c = ((cc11 j11) j11 + ((cc12 j11) j12) 2) + (cc22 j12) j12 + 0.3 (run_ac's c).
inline void run_c(uint32_t fy, uint32_t nfy) {
    TTI_SFPLOADI(P::LREG0, LOADI_LOWER, TWO & 0xFFFF);
    TTI_SFPLOADI(P::LREG0, LOADI_UPPER, TWO >> 16);
    TTI_SFPLOADI(P::LREG7, LOADI_LOWER, PT3 & 0xFFFF);
    TTI_SFPLOADI(P::LREG7, LOADI_UPPER, PT3 >> 16);
    loadi<P::LREG6>(fy);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_FUSE_LD(P::LREG1, 6);                      // L1 = inv_tz
        PFWC_FUSE_MUL(P::LREG1, P::LREG6, P::LREG2);    // L2 = j11 = inv * fy
        PFWC_FUSE_LD(P::LREG3, 7);                      // ty
        loadi<P::LREG4>(nfy);
        PFWC_FUSE_MUL(P::LREG3, P::LREG4, P::LREG3);
        PFWC_FUSE_MUL(P::LREG3, P::LREG1, P::LREG3);
        PFWC_FUSE_MUL(P::LREG3, P::LREG1, P::LREG3);    // L3 = j12
        PFWC_FUSE_LD(P::LREG4, 3);
        PFWC_FUSE_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_FUSE_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_FUSE_LD(P::LREG5, 4);
        PFWC_FUSE_MUL(P::LREG5, P::LREG2, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG0, P::LREG5);
        PFWC_FUSE_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_FUSE_LD(P::LREG5, 5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_FUSE_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_FUSE_ADD(P::LREG4, P::LREG7, P::LREG4);
        PFWC_FUSE_ST(P::LREG4, 3);
        PFWC_FUSE_ST(P::LREG4, 7);
        PFWC_FUSE_LD(P::LREG5, 0);                      // a -> tile 6 (radii.x)
        PFWC_FUSE_ST(P::LREG5, 6);
        sfpi::dst_reg++;
    }
}

#undef PFWC_FUSE_LD
#undef PFWC_FUSE_ST
#undef PFWC_FUSE_MUL
#undef PFWC_FUSE_ADD

}  // namespace pfwc_fuse
