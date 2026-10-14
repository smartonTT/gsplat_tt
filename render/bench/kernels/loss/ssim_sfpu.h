// Task #736 (T7 fused L1 + D-SSIM): the #720 SFPU fp32 FMA 11-tap filter (sfilt_compute.cpp, same
// op order, SF_CMODE 1 hoisted taps) with any input stack B..B+2 and output tile O, so a full-sync
// fp32 dest (8 tiles) holds two stacks and their outputs; plus the pointwise SFPU helpers.
// sfilt_compute.cpp stays as it is (the unfused A/B arm).
#pragma once
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/transpose_wh_dest.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif

#define SF_INL inline __attribute__((always_inline))

// Task #743 timing ablation (profile only, results are garbage when set): bit 0 skips the filter
// rows (FiltRows), 8 the row-layout conversions (TConv) of vfilt_body, 2 the dest transposes,
// 4 the pointwise SFPU passes. Host: LOSS_SSIM_ABL. 0 = the real kernel.
#ifndef SSIM_ABL
#define SSIM_ABL 0
#endif
// Task #753: strip width in tile columns (reader_ssim.cpp); host LOSS_SSIM_K.
#ifndef SSIM_K
#define SSIM_K 1
#endif

#ifdef TRISC_MATH
#ifndef SF_G0
#error "SF_G0..SF_G5 must be defined"
#endif

// dst_reg index of image row r of dest tile t in row layout (sfilt_compute.cpp rowslot).
constexpr uint32_t rowslot(uint32_t t, uint32_t r) {
    return 32 * t + 16 * ((r >> 2) >> 2) + 8 * ((r & 3) >> 1) + 2 * ((r >> 2) & 3) + (r & 1);
}
// Row R in [-32, 64) of the stack [B; B+1; B+2], relative to tile B+1.
template <uint32_t B>
constexpr uint32_t stackslot(int R) {
    return R < 0 ? rowslot(B, uint32_t(R + 32)) : R < 32 ? rowslot(B + 1, uint32_t(R)) : rowslot(B + 2, uint32_t(R - 32));
}

template <uint32_t t, uint32_t g>
SF_INL void tconv_group() {
    using namespace sfpi;
    constexpr uint32_t b = 32 * t + 16 * (g >> 2) + 2 * (g & 3);
    vFloat a0 = dst_reg[b], a1 = dst_reg[b + 1], a2 = dst_reg[b + 8], a3 = dst_reg[b + 9];
    subvec_transp(a0, a1, a2, a3);
    dst_reg[b] = a0;
    dst_reg[b + 1] = a1;
    dst_reg[b + 8] = a2;
    dst_reg[b + 9] = a3;
}
template <uint32_t t, uint32_t g, uint32_t g1>
struct TConv {
    static SF_INL void run() {
        tconv_group<t, g>();
        TConv<t, g + 1, g1>::run();
    }
};
template <uint32_t t, uint32_t g1>
struct TConv<t, g1, g1> {
    static SF_INL void run() {}
};

template <uint32_t B, uint32_t O, int r>
SF_INL void filt_row(const sfpi::vFloat c3, const sfpi::vFloat c4, const sfpi::vFloat c5) {
    using namespace sfpi;
    vFloat acc = (vFloat(dst_reg[stackslot<B>(r - 5)]) + dst_reg[stackslot<B>(r + 5)]) * vConstFloatPrgm0;
    acc = (vFloat(dst_reg[stackslot<B>(r - 4)]) + dst_reg[stackslot<B>(r + 4)]) * vConstFloatPrgm1 + acc;
    acc = (vFloat(dst_reg[stackslot<B>(r - 3)]) + dst_reg[stackslot<B>(r + 3)]) * vConstFloatPrgm2 + acc;
    acc = (vFloat(dst_reg[stackslot<B>(r - 2)]) + dst_reg[stackslot<B>(r + 2)]) * c3 + acc;
    acc = (vFloat(dst_reg[stackslot<B>(r - 1)]) + dst_reg[stackslot<B>(r + 1)]) * c4 + acc;
    acc = vFloat(dst_reg[stackslot<B>(r)]) * c5 + acc;
    dst_reg[rowslot(O, r)] = acc;
}
template <uint32_t B, uint32_t O, int r>
struct FiltRows {
    static SF_INL void run(const sfpi::vFloat c3, const sfpi::vFloat c4, const sfpi::vFloat c5) {
        filt_row<B, O, r>(c3, c4, c5);
        FiltRows<B, O, r + 1>::run(c3, c4, c5);
    }
};
template <uint32_t B, uint32_t O>
struct FiltRows<B, O, 32> {
    static SF_INL void run(const sfpi::vFloat, const sfpi::vFloat, const sfpi::vFloat) {}
};

// Vertical filter of the natural-layout stack B..B+2 (B+1 = the output's rows) -> tile O
// (natural). The stack is left partly in row layout (no longer usable).
template <uint32_t B, uint32_t O>
SF_INL void vfilt_body() {
    using namespace sfpi;
    if constexpr (!(SSIM_ABL & 8)) {
        TConv<B, 6, 8>::run();
        TConv<B + 1, 0, 8>::run();
        TConv<B + 2, 0, 2>::run();
    }
    if constexpr (!(SSIM_ABL & 1)) {
        const vFloat c3 = SF_G3, c4 = SF_G4, c5 = SF_G5;
        FiltRows<B, O, 0>::run(c3, c4, c5);
    }
    if constexpr (!(SSIM_ABL & 8)) TConv<O, 0, 8>::run();
}

inline void sfpu_begin() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
    _llk_math_eltwise_unary_sfpu_start_(0);
    sfpi::vConstFloatPrgm0 = SF_G0;
    sfpi::vConstFloatPrgm1 = SF_G1;
    sfpi::vConstFloatPrgm2 = SF_G2;
}
inline void sfpu_end() { _llk_math_eltwise_unary_sfpu_done_(); }

// 1 / x: SFPARECIP seed + two Newton steps (pfwc_recip_nr.h form), ~1 ulp.
SF_INL sfpi::vFloat rcp(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    y = y * (sfpi::vFloat(2.0f) - x * y);
    y = y * (sfpi::vFloat(2.0f) - x * y);
    return y;
}

template <uint32_t T>
inline void zero_tile() {
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        sfpi::dst_reg[32 * T] = 0.0f;
        sfpi::dst_reg++;
    }
    sfpu_end();
}
#endif

// Copy n consecutive front pages of cb to dest d0.. (no pop).
inline void copy_n(uint32_t cb, uint32_t n, uint32_t d0) {
    copy_tile_to_dst_init_short(cb);
    for (uint32_t k = 0; k < n; k++) copy_tile(cb, k, d0 + k);
}

// Copy front pages i0 .. i0+n-1 of cb to dest d0.. (no pop).
inline void copy_at(uint32_t cb, uint32_t i0, uint32_t n, uint32_t d0) {
    copy_tile_to_dst_init_short(cb);
    for (uint32_t k = 0; k < n; k++) copy_tile(cb, i0 + k, d0 + k);
}

inline void transpose_tiles(uint32_t mask) {
    if constexpr (SSIM_ABL & 2) return;
    transpose_wh_dest_init_short<true>();
    for (uint32_t k = 0; k < 8; k++)
        if (mask & (1u << k)) transpose_wh_dest<true>(k);
}

// Pack dest tiles d[0..n) into cb (one reserve/push of n pages), then release dest.
inline void commit_pack(uint32_t cb, uint32_t n, const uint32_t* d) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, n);
    for (uint32_t k = 0; k < n; k++) pack_tile(d[k], cb);
    cb_push_back(cb, n);
}
inline void pack_more(uint32_t cb, uint32_t n, const uint32_t* d) {
    cb_reserve_back(cb, n);
    for (uint32_t k = 0; k < n; k++) pack_tile(d[k], cb);
    cb_push_back(cb, n);
}
inline void release() { tile_regs_release(); }
