// Task #706 (T7 D-SSIM step 2): 11-tap separable Gaussian (sigma 1.5, zero padding) of fp32
// planes as banded tile matmuls on the FPU, split3 precision (docs/t690/DSSIM-design.md).
// One strip = one tile column J of one channel, walked down the tile rows I.
//   row pass:    row[I]  = sum_k X[I, J+k] @ R[k]          (reader sends X[I, J-1..J+1], zeros off-plane)
//   column pass: out[I]  = sum_k R[k]^T @ row[I+k]         (rolling 3-tile window, zero rows at the ends)
// split3: z = zh + zl (zh = z with the low 13 mantissa bits cleared, exact TF32; zl = z - zh),
// R = Rh + Rl likewise (host constants): zh Rh + zl Rh + zh Rl, accumulated in fp32 dest.
// Model: gsplat/train_loss.py filt_tiles(..., "split3", "trunc").
// Runtime args: n_strips, nh.
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
// Rh[-1,0,1], Rl[-1,0,1], RTh[-1,0,1], RTl[-1,0,1]. Others: plain fp32.
constexpr uint32_t CB_X = 0, CB_R = 2, CB_XH = 4, CB_XL = 5, CB_RH = 6, CB_RL = 7, CB_OUT = 16;
constexpr uint32_t RH = 0, RL = 3, RTH = 6, RTL = 9;

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
            const vFloat h = reinterpret<vFloat>(reinterpret<vInt>(z) & vInt(-8192));  // 0xFFFFE000
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

// Packs dest tiles 0 and 1 to CB_RH / CB_RL (one row-pass result, split).
inline void pack_row_pair() {
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_RH, 1);
    cb_reserve_back(CB_RL, 1);
    pack_tile(0, CB_RH);
    pack_tile(1, CB_RL);
    cb_push_back(CB_RH, 1);
    cb_push_back(CB_RL, 1);
    tile_regs_release();
}

inline void push_zero_row() {
    tile_regs_acquire();
    MATH((sfpu_begin()));
    MATH((zero_tiles2()));
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    pack_row_pair();
}

void kernel_main() {
    const uint32_t n_strips = get_arg_val<uint32_t>(0);
    const uint32_t nh = get_arg_val<uint32_t>(1);
    mm_init(CB_XH, CB_R, CB_OUT);
    cb_wait_front(CB_R, 12);
    for (uint32_t s = 0; s < n_strips; s++) {
        push_zero_row();  // row[-1]
        for (uint32_t i = 0; i <= nh; i++) {
            if (i < nh) {
                // split X[i, J-1..J+1] -> CB_XH, CB_XL
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
                // row pass -> dest 0, split -> CB_RH / CB_RL
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
                pack_row_pair();
            } else {
                push_zero_row();  // row[nh]
            }
            if (i == 0) continue;
            // column pass for output row i-1: window row[i-2..i]
            cb_wait_front(CB_RH, 3);
            cb_wait_front(CB_RL, 3);
            tile_regs_acquire();
            mm_init_short(CB_R, CB_RH);
            for (uint32_t k = 0; k < 3; k++) {
                matmul_tiles(CB_R, CB_RH, RTH + k, k, 0);
                matmul_tiles(CB_R, CB_RH, RTL + k, k, 0);
                matmul_tiles(CB_R, CB_RL, RTH + k, k, 0);
            }
            tile_regs_commit();
            cb_pop_front(CB_RH, 1);
            cb_pop_front(CB_RL, 1);
            tile_regs_wait();
            cb_reserve_back(CB_OUT, 1);
            pack_tile(0, CB_OUT);
            cb_push_back(CB_OUT, 1);
            tile_regs_release();
        }
        cb_pop_front(CB_RH, 2);
        cb_pop_front(CB_RL, 2);
    }
}
