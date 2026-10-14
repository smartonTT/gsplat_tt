// Task #706 (T7 D-SSIM step 2): 11-tap separable Gaussian (sigma 1.5, zero padding) of fp32
// planes as banded tile matmuls on the FPU, split3 precision (docs/t690/DSSIM-design.md).
// One strip = one tile column J of one channel, walked down the tile rows I.
//   row pass:    row[I]  = sum_k X[I, J+k] @ R[k]          (reader sends X[I, J-1..J+1], zeros off-plane)
//   column pass: out[I]  = sum_k R[k]^T @ row[I+k]         (rolling 3-tile window, zero rows at the ends)
// split3: z = zh + zl (zh = z & HI_MASK, zl = z - zh exactly),
// R = Rh + Rl likewise (host constants): zh Rh + zl Rh + zh Rl, accumulated in fp32 dest.
// Model: gsplat/train_loss.py filt_tiles(..., "split3", "trunc").
// HI_MASK keeps 9 explicit mantissa bits: run 1971 (13-bit mask) gave ~5e-4 relative error,
// TF32 level, consistent with the HiFi4 multiplier covering only 10 of srcA's 11 bits.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif

// CB_X: fp32 tiles unpacked straight to dest (u2d). CB_R: 12 constant tiles
// Rh[-1,0,1], Rl[-1,0,1], RTh[-1,0,1], RTl[-1,0,1]. Row-pass results rotate through three
// one-tile CB pairs (RH0..2 / RL0..2): matmul_tiles indexes a CB from its read pointer
// without wrapping, so a 3-tile sliding window in one ring CB reads past its end.
constexpr uint32_t CB_X = 0, CB_R = 2, CB_XH = 4, CB_XL = 5, CB_RH0 = 6, CB_RL0 = 9, CB_OUT = 16;
constexpr uint32_t RH = 0, RL = 3, RTH = 6, RTL = 9;
#ifndef HI_MASK
#define HI_MASK -16384  // 0xFFFFC000: hi keeps 9 explicit mantissa bits (see the header note)
#endif

#ifdef TRISC_MATH
// dest tiles [0, n) -> hi in place, lo in [n, 2n). 32 vectors of 32 lanes per fp32 tile.
template <uint32_t n>
inline void split_tiles() {
    using namespace sfpi;
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
#pragma GCC unroll 3
        for (uint32_t t = 0; t < n; t++) {
            const vFloat z = dst_reg[32 * t];
            const vFloat h = reinterpret<vFloat>(reinterpret<vInt>(z) & vInt(HI_MASK));
            dst_reg[32 * t] = h;
            dst_reg[32 * (t + n)] = z - h;
        }
        dst_reg++;
    }
}

inline void zero_tiles2() {
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        sfpi::dst_reg[0] = 0.0f;
        sfpi::dst_reg[32] = 0.0f;
        sfpi::dst_reg++;
    }
}

inline void sfpu_begin() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>();
    _llk_math_eltwise_unary_sfpu_start_(0);
}
#endif

// Packs dest tiles 0 and 1 to row slot q (hi / lo).
inline void pack_row_pair(uint32_t q) {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_RH0 + q, 1);
    cb_reserve_back(CB_RL0 + q, 1);
    pack_tile(0, CB_RH0 + q);
    pack_tile(1, CB_RL0 + q);
    cb_push_back(CB_RH0 + q, 1);
    cb_push_back(CB_RL0 + q, 1);
    tile_regs_release();
}

inline void push_zero_row(uint32_t q) {
    tile_regs_acquire();
    MATH((sfpu_begin()));
    MATH((zero_tiles2()));
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    pack_row_pair(q);
}

// Row pass of tile row i (reader sends X[i, J-1..J+1]) -> row slot q.
inline void row_pass(uint32_t q) {
    cb_wait_front(CB_X, 3);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_X);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_X, k, k);
    MATH((sfpu_begin()));
    MATH((split_tiles<3>()));
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    tile_regs_commit();
    cb_pop_front(CB_X, 3);
    tile_regs_wait();
    cb_reserve_back(CB_XH, 3);
    cb_reserve_back(CB_XL, 3);
    for (uint32_t k = 0; k < 3; k++) pack_tile(k, CB_XH);
    for (uint32_t k = 0; k < 3; k++) pack_tile(3 + k, CB_XL);
    cb_push_back(CB_XH, 3);
    cb_push_back(CB_XL, 3);
    tile_regs_release();
    cb_wait_front(CB_XH, 3);
    cb_wait_front(CB_XL, 3);
    tile_regs_acquire();
    mm_init_short(CB_XH, CB_R);
    for (uint32_t k = 0; k < 3; k++) {
        matmul_tiles(CB_XH, CB_R, k, RH + k, 0);
        matmul_tiles(CB_XL, CB_R, k, RH + k, 0);
        matmul_tiles(CB_XH, CB_R, k, RL + k, 0);
    }
    MATH((sfpu_begin()));
    MATH((split_tiles<1>()));
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    cb_pop_front(CB_XH, 3);
    cb_pop_front(CB_XL, 3);
    pack_row_pair(q);
}

// Column pass: output = sum_k RT[k] @ row[slot(k)], slots a, b, c for k = -1, 0, 1.
inline void col_pass(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t sl[3] = {a, b, c};
    for (uint32_t k = 0; k < 3; k++) {
        cb_wait_front(CB_RH0 + sl[k], 1);
        cb_wait_front(CB_RL0 + sl[k], 1);
    }
    tile_regs_acquire();
    mm_init_short(CB_R, CB_RH0);
    for (uint32_t k = 0; k < 3; k++) {
        matmul_tiles(CB_R, CB_RH0 + sl[k], RTH + k, 0, 0);
        matmul_tiles(CB_R, CB_RH0 + sl[k], RTL + k, 0, 0);
        matmul_tiles(CB_R, CB_RL0 + sl[k], RTH + k, 0, 0);
    }
    tile_regs_commit();
    cb_pop_front(CB_RH0 + a, 1);
    cb_pop_front(CB_RL0 + a, 1);
    tile_regs_wait();
    cb_reserve_back(CB_OUT, 1);
    pack_tile(0, CB_OUT);
    cb_push_back(CB_OUT, 1);
    tile_regs_release();
}

// Runtime args: t0, t1 (output tiles, strip-major: t = strip * nh + row), nh.
// Each strip segment [r0, r1) recomputes the row pass of its halo rows r0-1 and r1.
void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    mm_init(CB_XH, CB_R, CB_OUT);
    cb_wait_front(CB_R, 12);
    for (uint32_t t = t0; t < t1;) {
        const uint32_t r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        // rows r0-1 .. r1 -> slots q = 0, 1, ...; output row r0 + q - 2 once slot q is in.
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;  // wraps for r0 = 0
            if (row < nh) {
                row_pass(q % 3);
            } else {
                push_zero_row(q % 3);
            }
            if (q >= 2) col_pass((q - 2) % 3, (q - 1) % 3, q % 3);
        }
        q--;  // last slot written
        cb_pop_front(CB_RH0 + (q - 1) % 3, 1);
        cb_pop_front(CB_RL0 + (q - 1) % 3, 1);
        cb_pop_front(CB_RH0 + q % 3, 1);
        cb_pop_front(CB_RL0 + q % 3, 1);
    }
}
