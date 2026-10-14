// Task #720 (T7 D-SSIM): 11-tap separable Gaussian (sigma 1.5, zero padding) of fp32 planes on the
// SFPU in fp32 (no FPU accumulator: #714 showed its multi-term sums carry a 1e-5..2.9e-4 bias).
// One strip = one tile column J of one plane, walked down the tile rows I (as filt_compute.cpp).
//   H pass, tile row i: X[i, J-1..J+1] -> dest 0..2, each transposed in dest (exact for 32 bit),
//            so the horizontal filter becomes a vertical one over the stack [0; 1; 2]; the result
//            (dest 3) is transposed back and packed to row slot q (CB_H0 + q, fp32).
//   V pass:  H[i-1], H[i], H[i+1] (slots) -> dest 0..2, vertical filter -> dest 3 -> CB_OUT.
// Vertical filter on the SFPU: an SFPLOAD sees 4 rows x 8 (even or odd) columns. SFPTRANSP of
// the four vectors of a 4-row group (face 0/1, even/odd) turns each into one full 32-column row
// ("row layout", rowslot() below), so a vertical tap is a plain lane-wise FMA:
//   out[r] = g0 (x[r-5] + x[r+5]) + g1 (x[r-4] + x[r+4]) + ... + g5 x[r]   (fp32, this order)
// Model: opt/t720/sfilt_check.py emulate().
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

// CB_X: fp32 tiles unpacked straight to dest (u2d). Row-pass results rotate through three
// one-tile CBs (CB_H0..2, u2d too): copy_tile indexes a CB from its read pointer without wrapping.
constexpr uint32_t CB_X = 0, CB_H0 = 6, CB_OUT = 16;

#ifdef TRISC_MATH
// Taps g0 (outermost) .. g5 (centre), fp32 literals from the host (loss_bench.cpp run_sfilt).
#ifndef SF_G0
#error "SF_G0..SF_G5 must be defined"
#endif

// dst_reg index (2 dest rows per unit, 32 per fp32 tile) of image row r of dest tile t in row
// layout: row 4g + q lives where the SFPLOAD vector (group g, face half q >> 1, parity q & 1) was.
constexpr uint32_t rowslot(uint32_t t, uint32_t r) {
    return 32 * t + 16 * ((r >> 2) >> 2) + 8 * ((r & 3) >> 1) + 2 * ((r >> 2) & 3) + (r & 1);
}
// Row R of the stack [tile 0; tile 1; tile 2], R in [-32, 64) relative to tile 1.
constexpr uint32_t stackslot(int R) {
    return R < 0 ? rowslot(0, uint32_t(R + 32)) : R < 32 ? rowslot(1, uint32_t(R)) : rowslot(2, uint32_t(R - 32));
}

// Natural <-> row layout of 4-row group g of dest tile t (SFPTRANSP is its own inverse).
template <uint32_t t, uint32_t g>
inline void tconv_group() {
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
    static inline void run() {
        tconv_group<t, g>();
        TConv<t, g + 1, g1>::run();
    }
};
template <uint32_t t, uint32_t g1>
struct TConv<t, g1, g1> {
    static inline void run() {}
};

template <int r>
inline void filt_row(const sfpi::vFloat& c3, const sfpi::vFloat& c4, const sfpi::vFloat& c5) {
    using namespace sfpi;
    vFloat acc = (vFloat(dst_reg[stackslot(r - 5)]) + dst_reg[stackslot(r + 5)]) * vConstFloatPrgm0;
    acc = (vFloat(dst_reg[stackslot(r - 4)]) + dst_reg[stackslot(r + 4)]) * vConstFloatPrgm1 + acc;
    acc = (vFloat(dst_reg[stackslot(r - 3)]) + dst_reg[stackslot(r + 3)]) * vConstFloatPrgm2 + acc;
    acc = (vFloat(dst_reg[stackslot(r - 2)]) + dst_reg[stackslot(r + 2)]) * c3 + acc;
    acc = (vFloat(dst_reg[stackslot(r - 1)]) + dst_reg[stackslot(r + 1)]) * c4 + acc;
    acc = vFloat(dst_reg[stackslot(r)]) * c5 + acc;
    dst_reg[rowslot(3, r)] = acc;
}

template <int r>
struct FiltRows {
    static inline void run(const sfpi::vFloat& c3, const sfpi::vFloat& c4, const sfpi::vFloat& c5) {
        filt_row<r>(c3, c4, c5);
        FiltRows<r + 1>::run(c3, c4, c5);
    }
};
template <>
struct FiltRows<32> {
    static inline void run(const sfpi::vFloat&, const sfpi::vFloat&, const sfpi::vFloat&) {}
};

// dest 0..2 (natural layout, tile 1 = the output's rows) -> dest 3 = vertical filter (natural).
// Only the rows the filter reads are converted: tile 0 rows 24..31, tile 2 rows 0..7.
inline void vfilter_dest() {
    using namespace sfpi;
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
    _llk_math_eltwise_unary_sfpu_start_(0);
    vConstFloatPrgm0 = SF_G0;
    vConstFloatPrgm1 = SF_G1;
    vConstFloatPrgm2 = SF_G2;
    TConv<0, 6, 8>::run();
    TConv<1, 0, 8>::run();
    TConv<2, 0, 2>::run();
    const vFloat c3 = SF_G3, c4 = SF_G4, c5 = SF_G5;
    FiltRows<0>::run(c3, c4, c5);
    TConv<3, 0, 8>::run();
    _llk_math_eltwise_unary_sfpu_done_();
}

inline void zero_tile3() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
    _llk_math_eltwise_unary_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        sfpi::dst_reg[96] = 0.0f;
        sfpi::dst_reg++;
    }
    _llk_math_eltwise_unary_sfpu_done_();
}
#endif

inline void pack_out(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb, 1);
    pack_tile(3, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}

// H pass of tile row i (reader sends X[i, J-1..J+1]) -> row slot q.
inline void h_pass(uint32_t q) {
    cb_wait_front(CB_X, 3);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_X);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_X, k, k);
    transpose_wh_dest_init_short<true>();
    for (uint32_t k = 0; k < 3; k++) transpose_wh_dest<true>(k);
    MATH((vfilter_dest()));
    transpose_wh_dest_init_short<true>();
    transpose_wh_dest<true>(3);
    cb_pop_front(CB_X, 3);
    pack_out(CB_H0 + q);
}

// Off-plane tile row: a zero tile in slot q.
inline void push_zero(uint32_t q) {
    tile_regs_acquire();
    MATH((zero_tile3()));
    pack_out(CB_H0 + q);
}

// V pass: output = vertical filter of the slots a, b, c (rows i-1, i, i+1).
inline void v_pass(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t sl[3] = {a, b, c};
    for (uint32_t k = 0; k < 3; k++) cb_wait_front(CB_H0 + sl[k], 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_H0 + a);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_H0 + sl[k], 0, k);
    MATH((vfilter_dest()));
    cb_pop_front(CB_H0 + a, 1);
    pack_out(CB_OUT);
}

// Runtime args: t0, t1 (output tiles, strip-major: t = strip * nh + row), nh.
// Each strip segment [r0, r1) recomputes the H pass of its halo rows r0-1 and r1.
void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    unary_op_init_common(CB_X, CB_OUT);
    for (uint32_t t = t0; t < t1;) {
        const uint32_t r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        // rows r0-1 .. r1 -> slots q = 0, 1, ...; output row r0 + q - 2 once slot q is in.
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;  // wraps for r0 = 0
            if (row < nh) {
                h_pass(q % 3);
            } else {
                push_zero(q % 3);
            }
            if (q >= 2) v_pass((q - 2) % 3, (q - 1) % 3, q % 3);
        }
        q--;  // last slot written
        cb_pop_front(CB_H0 + (q - 1) % 3, 1);
        cb_pop_front(CB_H0 + q % 3, 1);
    }
}
