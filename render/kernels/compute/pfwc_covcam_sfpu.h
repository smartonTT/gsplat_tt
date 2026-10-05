// SPDX-License-Identifier: Apache-2.0
//
// Task #206: cov_cam (cc_e = sum_k s_ek * c_k, the per-view 6x6 map) as one SFPU
// pass over the six cov3d tiles in DEST (GSPLAT_TT_PFWC_COVCAM_SFPU=1, math thread).
// The default path spends 36 copy_tile, 36 mul_unary_tile and 30 add_binary_tile
// per chunk on it; this path needs 6 copy_tile and run() below.
//
// DEST: tile k (0..5) holds c_k (COV3D_CB order); s_ek (runtime arg 17 + 6e + k)
// is staged as a lane-broadcast vector at DEST vector SV + 6e + k (tiles 6..7,
// once per acquire: the pack-side release clears DEST). Per vector V: LREG0..5 =
// c_0..c_5, then for each entry e
//   LREG6 = c_0 * s_e0;  LREG6 = LREG6 + c_k * s_ek  (k = 1..5)
// with every product (SFPMUL, + 0.0) and every sum (SFPADD, * 1.0) rounded on its
// own, in the order mul_unary_tile and add_binary_tile use: bit-identical to the
// default path (tests/unit/test_covcam_sfpu.cpp). No SFPMAD fusion: it would change
// the rounding and the md5. LREG6 then overwrites tile e at V; the inputs at V are
// already in LREGs. Addresses are DEST rows: sfpi dst_reg[i] is row 2i, so tile t
// vector V is row 64t + 2V. Blackhole stalls on SFPMAD results, so no SFPNOPs.
#pragma once

#include <cstdint>

namespace pfwc_covcam {

using P = ckernel::p_sfpu;
constexpr uint32_t SV = 6 * 32;                       // DEST vector of s_00
constexpr uint32_t LOADI_UPPER = 8, LOADI_LOWER = 10;  // SFPLOADI_MOD0_UPPER / _LOWER

// LREG6 += c_K * s_EK (LREG7 = s_EK, then the product in place).
template <uint32_t E, uint32_t K>
inline void term() {
    TTI_SFPLOAD(P::LREG7, 0, ckernel::ADDR_MOD_7, 2 * (SV + 6 * E + K));
    TTI_SFPMUL(P::LREG0 + K, P::LREG7, P::LCONST_0, P::LREG7, 0);
    TTI_SFPADD(P::LREG6, P::LCONST_1, P::LREG7, P::LREG6, 0);
}

// Entry E at DEST row `row` of tile E.
template <uint32_t E>
inline void entry(uint32_t row) {
    TTI_SFPLOAD(P::LREG6, 0, ckernel::ADDR_MOD_7, 2 * (SV + 6 * E));
    TTI_SFPMUL(P::LREG0, P::LREG6, P::LCONST_0, P::LREG6, 0);
    term<E, 1>();
    term<E, 2>();
    term<E, 3>();
    term<E, 4>();
    term<E, 5>();
    TT_SFPSTORE(P::LREG6, 0, ckernel::ADDR_MOD_7, 64 * E + row);
}

// scale_bits(j) = fp32 bits of s_ek, j = 6e + k. Call between SFPU start(0) and done.
template <class ScaleBits>
inline void run(ScaleBits scale_bits) {
    for (uint32_t j = 0; j < 36; j++) {
        const uint32_t b = scale_bits(j);
        TT_SFPLOADI(P::LREG7, LOADI_LOWER, b & 0xFFFF);
        TT_SFPLOADI(P::LREG7, LOADI_UPPER, b >> 16);
        TT_SFPSTORE(P::LREG7, 0, ckernel::ADDR_MOD_7, 2 * (SV + j));
    }
    for (uint32_t row = 0; row < 64; row += 2) {
        TT_SFPLOAD(P::LREG0, 0, ckernel::ADDR_MOD_7, 0 * 64 + row);
        TT_SFPLOAD(P::LREG1, 0, ckernel::ADDR_MOD_7, 1 * 64 + row);
        TT_SFPLOAD(P::LREG2, 0, ckernel::ADDR_MOD_7, 2 * 64 + row);
        TT_SFPLOAD(P::LREG3, 0, ckernel::ADDR_MOD_7, 3 * 64 + row);
        TT_SFPLOAD(P::LREG4, 0, ckernel::ADDR_MOD_7, 4 * 64 + row);
        TT_SFPLOAD(P::LREG5, 0, ckernel::ADDR_MOD_7, 5 * 64 + row);
        entry<0>(row);
        entry<1>(row);
        entry<2>(row);
        entry<3>(row);
        entry<4>(row);
        entry<5>(row);
    }
}

}  // namespace pfwc_covcam
