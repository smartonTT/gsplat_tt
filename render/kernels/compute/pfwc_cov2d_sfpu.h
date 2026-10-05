// SPDX-License-Identifier: Apache-2.0
//
// Task #228 (P2, docs/pfwc-trisc-model-t226): cov2d a / b / c of steps 7-11 as
// two looped SFPU passes over DEST (GSPLAT_TT_PFWC_COV2D_SFPU=1, math thread).
// The default path spends ~150 tile ops per chunk there: it rebuilds the
// Jacobian terms j00, j02, j11, j12 in each of five acquires and a and c twice.
// Here each section loads its inputs once and keeps everything in LREGs.
//
// run_ac(): DEST tiles 0 cc00, 1 cc02, 2 cc22, 3 cc11, 4 cc12, 5 inv_tz, 6 tx,
// 7 ty. Out: a -> tiles 0 and 6, c -> tiles 3 and 7 (tx / ty are consumed).
// run_b(): DEST tiles 0 cc02, 1 cc01, 2 cc12, 3 cc22, 4 inv_tz, 5 tx, 6 ty.
// Out: b -> tiles 1 and 7.
//
// Per lane every product (SFPMUL, + 0.0) and every sum (SFPADD, * 1.0) is
// rounded on its own, in the order mul_unary_tile, mul_binary_tile,
// add_binary_tile and add_unary_tile use in the default path: bit-identical to
// it (tests/unit/test_cov2d_sfpu.cpp). No SFPMAD fusion: it would change the
// rounding and the md5. Addresses are DEST rows of vector 0 (tile t at row
// 64t); sfpi::dst_reg++ moves the SFPU DEST counter to the next vector (2
// rows). Two loop-invariant constants per pass stay in LREG0 / LREG7; the rest
// are loaded per vector with SFPLOADI (8 LREGs, no spill). Blackhole stalls on
// SFPMAD results, so no SFPNOPs. Call each run between SFPU start(0) and done:
// done resets the counter.
#pragma once

#include <cstdint>

namespace pfwc_cov2d {

using P = ckernel::p_sfpu;
constexpr uint32_t LOADI_UPPER = 8, LOADI_LOWER = 10;  // SFPLOADI_MOD0_UPPER / _LOWER
constexpr uint32_t TWO = 0x40000000u, PT3 = 0x3E99999Au;  // 2.0f, 0.3f
constexpr uint32_t T = 64;                               // DEST rows per tile

#define PFWC_COV2D_LD(l, t) TTI_SFPLOAD((l), 0, ckernel::ADDR_MOD_7, (t) * T)
#define PFWC_COV2D_ST(l, t) TTI_SFPSTORE((l), 0, ckernel::ADDR_MOD_7, (t) * T)
#define PFWC_COV2D_MUL(a, b, d) TTI_SFPMUL((a), (b), P::LCONST_0, (d), 0)
#define PFWC_COV2D_ADD(a, c, d) TTI_SFPADD((a), P::LCONST_1, (c), (d), 0)

template <uint32_t L>
inline void loadi(uint32_t bits) {
    TT_SFPLOADI(L, LOADI_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(L, LOADI_UPPER, bits >> 16);
}

// fx, fy and their negations as fp32 bits (runtime args 13, 14, 54, 55).
inline void run_ac(uint32_t fx, uint32_t nfx, uint32_t fy, uint32_t nfy) {
    TTI_SFPLOADI(P::LREG0, LOADI_LOWER, TWO & 0xFFFF);
    TTI_SFPLOADI(P::LREG0, LOADI_UPPER, TWO >> 16);
    TTI_SFPLOADI(P::LREG7, LOADI_LOWER, PT3 & 0xFFFF);
    TTI_SFPLOADI(P::LREG7, LOADI_UPPER, PT3 >> 16);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_COV2D_LD(P::LREG1, 5);                     // L1 = inv_tz
        loadi<P::LREG2>(fx);
        PFWC_COV2D_MUL(P::LREG1, P::LREG2, P::LREG2);   // L2 = j00 = inv * fx
        PFWC_COV2D_LD(P::LREG3, 6);                     // tx
        loadi<P::LREG4>(nfx);
        PFWC_COV2D_MUL(P::LREG3, P::LREG4, P::LREG3);   // tx * -fx
        PFWC_COV2D_MUL(P::LREG3, P::LREG1, P::LREG3);   // * inv
        PFWC_COV2D_MUL(P::LREG3, P::LREG1, P::LREG3);   // L3 = j02
        // a = ((cc00 j00) j00 + ((cc02 j00) j02) 2) + (cc22 j02) j02 + 0.3
        PFWC_COV2D_LD(P::LREG4, 0);
        PFWC_COV2D_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_COV2D_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_COV2D_LD(P::LREG5, 1);
        PFWC_COV2D_MUL(P::LREG5, P::LREG2, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG0, P::LREG5);
        PFWC_COV2D_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_COV2D_LD(P::LREG5, 2);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_COV2D_ADD(P::LREG4, P::LREG7, P::LREG4);
        PFWC_COV2D_ST(P::LREG4, 0);
        PFWC_COV2D_ST(P::LREG4, 6);
        loadi<P::LREG2>(fy);
        PFWC_COV2D_MUL(P::LREG1, P::LREG2, P::LREG2);   // L2 = j11 = inv * fy
        PFWC_COV2D_LD(P::LREG3, 7);                     // ty
        loadi<P::LREG4>(nfy);
        PFWC_COV2D_MUL(P::LREG3, P::LREG4, P::LREG3);
        PFWC_COV2D_MUL(P::LREG3, P::LREG1, P::LREG3);
        PFWC_COV2D_MUL(P::LREG3, P::LREG1, P::LREG3);   // L3 = j12
        // c = ((cc11 j11) j11 + ((cc12 j11) j12) 2) + (cc22 j12) j12 + 0.3
        PFWC_COV2D_LD(P::LREG4, 3);
        PFWC_COV2D_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_COV2D_MUL(P::LREG4, P::LREG2, P::LREG4);
        PFWC_COV2D_LD(P::LREG5, 4);
        PFWC_COV2D_MUL(P::LREG5, P::LREG2, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG0, P::LREG5);
        PFWC_COV2D_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_COV2D_LD(P::LREG5, 2);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG3, P::LREG5);
        PFWC_COV2D_ADD(P::LREG4, P::LREG5, P::LREG4);
        PFWC_COV2D_ADD(P::LREG4, P::LREG7, P::LREG4);
        PFWC_COV2D_ST(P::LREG4, 3);
        PFWC_COV2D_ST(P::LREG4, 7);
        sfpi::dst_reg++;
    }
}

inline void run_b(uint32_t fx, uint32_t nfx, uint32_t fy, uint32_t nfy) {
    loadi<P::LREG0>(fx);
    loadi<P::LREG7>(fy);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        PFWC_COV2D_LD(P::LREG1, 4);                     // L1 = inv_tz
        PFWC_COV2D_MUL(P::LREG1, P::LREG0, P::LREG2);   // L2 = j00
        PFWC_COV2D_MUL(P::LREG1, P::LREG7, P::LREG3);   // L3 = j11
        PFWC_COV2D_LD(P::LREG4, 5);                     // tx
        loadi<P::LREG5>(nfx);
        PFWC_COV2D_MUL(P::LREG4, P::LREG5, P::LREG4);
        PFWC_COV2D_MUL(P::LREG4, P::LREG1, P::LREG4);
        PFWC_COV2D_MUL(P::LREG4, P::LREG1, P::LREG4);   // L4 = j02
        PFWC_COV2D_LD(P::LREG5, 6);                     // ty
        loadi<P::LREG6>(nfy);
        PFWC_COV2D_MUL(P::LREG5, P::LREG6, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG1, P::LREG5);
        PFWC_COV2D_MUL(P::LREG5, P::LREG1, P::LREG5);   // L5 = j12
        // b = (((cc01 j00) j11 + (cc02 j00) j12) + (cc12 j02) j11) + (cc22 j02) j12
        PFWC_COV2D_LD(P::LREG1, 1);
        PFWC_COV2D_MUL(P::LREG1, P::LREG2, P::LREG1);
        PFWC_COV2D_MUL(P::LREG1, P::LREG3, P::LREG1);
        PFWC_COV2D_LD(P::LREG6, 0);
        PFWC_COV2D_MUL(P::LREG6, P::LREG2, P::LREG6);
        PFWC_COV2D_MUL(P::LREG6, P::LREG5, P::LREG6);
        PFWC_COV2D_ADD(P::LREG1, P::LREG6, P::LREG1);
        PFWC_COV2D_LD(P::LREG6, 2);
        PFWC_COV2D_MUL(P::LREG6, P::LREG4, P::LREG6);
        PFWC_COV2D_MUL(P::LREG6, P::LREG3, P::LREG6);
        PFWC_COV2D_ADD(P::LREG1, P::LREG6, P::LREG1);
        PFWC_COV2D_LD(P::LREG6, 3);
        PFWC_COV2D_MUL(P::LREG6, P::LREG4, P::LREG6);
        PFWC_COV2D_MUL(P::LREG6, P::LREG5, P::LREG6);
        PFWC_COV2D_ADD(P::LREG1, P::LREG6, P::LREG1);
        PFWC_COV2D_ST(P::LREG1, 1);
        PFWC_COV2D_ST(P::LREG1, 7);
        sfpi::dst_reg++;
    }
}

#undef PFWC_COV2D_LD
#undef PFWC_COV2D_ST
#undef PFWC_COV2D_MUL
#undef PFWC_COV2D_ADD

}  // namespace pfwc_cov2d
