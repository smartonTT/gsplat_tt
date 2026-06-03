// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DST-persistent full-tile alpha-blend (amendment-003 step 3).
//
// WHY THIS KERNEL
// ---------------
// The legacy kernel spills running R/G/B/T state to L1 CBs every Gaussian
// (pack_tile -> CB -> copy_tile). With fp32 state CBs + fp32_dest_acc on
// Blackhole that round-trip corrupts tile rows >= 8 (structured 0/1 bands);
// with bf16 state CBs it's row-stable but caps ~26-39 dB (accumulator
// swamping over dense full-frame tiles).
//
// This kernel keeps R/G/B/T in fp32 DEST slots 0..3 for the ENTIRE per-tile
// Gaussian loop inside one tile_regs_acquire. Per-Gaussian Mahalanobis/alpha
// math uses DEST scratch slots 4..7. No state ever touches an L1 CB until the
// final pack at tile end. This is the same fp32-DEST + SFPU pattern that took
// project/pfwc from 34 -> 82 dB (tt-007). Requires dst_full_sync_en=true so
// the DEST file exposes 8 fp32 tiles (0..7).
//
// All SFPU op descriptors are initialized ONCE before the tile loop and reused
// mid-acquire (the proven pfwc pattern); we do NOT re-init ops inside the
// Gaussian loop (that desyncs the TRISC threads and was the prior hang cause).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/binary_max_min.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t use_mb_shadow = get_arg_val<uint32_t>(1);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_COLOR_OUT = 16;
    constexpr uint32_t CB_CONST_099 = 23;
    constexpr uint32_t CB_MB_COEFF_SHADOW = 24;

    // Persistent fp32 state slots (live across the whole Gaussian loop).
    constexpr uint32_t DST_R = 0;
    constexpr uint32_t DST_G = 1;
    constexpr uint32_t DST_B = 2;
    constexpr uint32_t DST_T = 3;
    // Per-Gaussian scratch.
    constexpr uint32_t DST_S4 = 4;
    constexpr uint32_t DST_S5 = 5;
    constexpr uint32_t DST_S6 = 6;
    constexpr uint32_t DST_S7 = 7;

    constexpr uint32_t NEG_HALF_BITS = 0xBF000000u;  // -0.5
    constexpr uint32_t ONE_F_BITS    = 0x3F800000u;  //  1.0

    // One-time SFPU/FPU init (proven pfwc pattern: register every op descriptor
    // ONCE, reuse mid-acquire; never re-init inside the Gaussian loop).
    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);
    add_binary_tile_init();
    mul_binary_tile_init();
    binary_min_tile_init();
    exp_tile_init<false>();
    fill_tile_init();

    // CB_CONST_099 <- tile of 0.99 (alpha clamp ceiling). This kernel replaces
    // the legacy one, so it owns the constant fill.
    cb_reserve_back(CB_CONST_099, 1);
    tile_regs_acquire();
    fill_tile(0, 0.99f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_099);
    tile_regs_release();
    cb_push_back(CB_CONST_099, 1);

    cb_wait_front(CB_CONST_099, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_TILE_META, 1);
        const uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, 0, 0);
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        tile_regs_acquire();

        // R = G = B = 0, T = 1.0 (persistent across the Gaussian loop).
        fill_tile(DST_R, 0.0f);
        fill_tile(DST_G, 0.0f);
        fill_tile(DST_B, 0.0f);
        fill_tile(DST_T, 1.0f);

        for (uint32_t g = 0; g < g_count; g++) {
            if (use_mb_shadow) {
                cb_wait_front(CB_MB_COEFF_SHADOW, 1);
                cb_pop_front(CB_MB_COEFF_SHADOW, 1);
            }
            cb_wait_front(CB_SCALARS, 1);

            const uint32_t mean_x_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            const uint32_t mean_y_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            const uint32_t cov_a_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            const uint32_t two_cov_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            const uint32_t cov_c_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            const uint32_t color_r_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            const uint32_t color_g_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            const uint32_t color_b_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            const uint32_t opacity_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 8);

            // dx -> S4, dy -> S5  (tile-local px/py minus tile-local mean).
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, DST_S4);
            sub_unary_tile(DST_S4, mean_x_bits);
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, DST_S5);
            sub_unary_tile(DST_S5, mean_y_bits);

            // Q = a*dx^2 + c*dy^2 + 2b*dx*dy  -> S6
            mul_binary_tile(DST_S4, DST_S4, DST_S6);  // dx^2
            mul_unary_tile(DST_S6, cov_a_bits);       // a*dx^2
            mul_binary_tile(DST_S5, DST_S5, DST_S7);  // dy^2
            mul_unary_tile(DST_S7, cov_c_bits);       // c*dy^2
            add_binary_tile(DST_S6, DST_S7, DST_S6);  // a*dx^2 + c*dy^2
            mul_binary_tile(DST_S4, DST_S5, DST_S7);  // dx*dy
            mul_unary_tile(DST_S7, two_cov_b_bits);   // 2b*dx*dy
            add_binary_tile(DST_S6, DST_S7, DST_S6);  // Q

            // power = -0.5 * Q ; weight = exp(power) ; alpha = min(op*weight, 0.99)
            mul_unary_tile(DST_S6, NEG_HALF_BITS);
            exp_tile<false>(DST_S6);
            mul_unary_tile(DST_S6, opacity_bits);
            copy_tile_to_dst_init_short(CB_CONST_099);
            copy_tile(CB_CONST_099, 0, DST_S7);
            binary_min_tile(DST_S6, DST_S7, DST_S6);  // alpha in S6

            // contrib = alpha * T ; R/G/B += color * contrib
            mul_binary_tile(DST_S6, DST_T, DST_S4);   // contrib
            mul_unary_tile(DST_S4, color_r_bits);
            add_binary_tile(DST_R, DST_S4, DST_R);

            mul_binary_tile(DST_S6, DST_T, DST_S4);
            mul_unary_tile(DST_S4, color_g_bits);
            add_binary_tile(DST_G, DST_S4, DST_G);

            mul_binary_tile(DST_S6, DST_T, DST_S4);
            mul_unary_tile(DST_S4, color_b_bits);
            add_binary_tile(DST_B, DST_S4, DST_B);

            // T *= (1 - alpha)
            rsub_unary_tile(DST_S6, ONE_F_BITS);      // S6 = 1 - alpha
            mul_binary_tile(DST_T, DST_S6, DST_T);

            cb_pop_front(CB_SCALARS, 1);
        }

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(DST_R, CB_COLOR_OUT);
        pack_tile(DST_G, CB_COLOR_OUT);
        pack_tile(DST_B, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
