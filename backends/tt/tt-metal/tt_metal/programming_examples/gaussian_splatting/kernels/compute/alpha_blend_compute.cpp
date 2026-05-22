// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
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
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/binary_max_min.h"

// Alpha-blend compute kernel: 3D Gaussian Splatting forward rasterizer
// (front-to-back compositing) for a per-core slice of screen tiles.
//
// HIGH-LEVEL FLOW
// ----------------
// For each screen tile this core owns, with running per-pixel accumulator
// state (R/G/B color, transmittance T):
//
//   Init      : R = G = B = 0,  T = 1   (per pixel)
//   Per tile  : precompute px², py², px·py basis tiles once
//   For each Gaussian g in this tile, sorted front-to-back:
//     Stage F : every 16 g's (skip g=0): hard-zero T where T < 1e-4
//     Stage A : read 10 fp32 scalars (basis coeffs, color, opacity)
//     Stage B : q = A·px² + B·px·py + C·py² + D·px + E·py + F  (FPU bcast)
//     Stage C : weight  = exp(min(-0.5·q, 0)); alpha = min(opacity·weight, 0.99)
//     Stage D1: contrib = alpha · T_state
//     Stage D2: color_c_state += color_c · contrib
//     Stage E : T ← T · (1 - alpha)
//   Output    : pack R_state, G_state, B_state to CB_COLOR_OUT (3 tiles).
//
// RUNTIME ARGS
//   0: num_tiles  -- number of screen tiles this core processes
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX            = 0;
    constexpr uint32_t CB_PY            = 1;
    constexpr uint32_t CB_SCALARS       = 2;
    constexpr uint32_t CB_TILE_META     = 3;
    constexpr uint32_t CB_PX2           = 4;
    constexpr uint32_t CB_PY2           = 5;
    constexpr uint32_t CB_PXPY          = 6;
    constexpr uint32_t CB_COEFF         = 7;
    constexpr uint32_t CB_DXDY          = 8;
    constexpr uint32_t CB_Q             = 9;
    constexpr uint32_t CB_POWER         = 10;
    constexpr uint32_t CB_ALPHA         = 12;
    constexpr uint32_t CB_CONTRIB       = 13;
    constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
    constexpr uint32_t CB_T_TMP         = 15;
    constexpr uint32_t CB_COLOR_OUT     = 16;
    constexpr uint32_t CB_COLOR_R_STATE = 17;
    constexpr uint32_t CB_COLOR_G_STATE = 18;
    constexpr uint32_t CB_COLOR_B_STATE = 19;
    constexpr uint32_t CB_T_STATE       = 20;
    constexpr uint32_t CB_SAT_MASK      = 21;
    constexpr uint32_t CB_CONST_ZERO    = 22;
    constexpr uint32_t CB_CONST_099     = 23;

    // Stage D2 scratch (disjoint from per-tile basis CBs held at front).
    constexpr uint32_t CB_T_R = CB_DXDY;
    constexpr uint32_t CB_T_G = CB_Q;
    constexpr uint32_t CB_T_B = CB_POWER;

    constexpr uint32_t NEG_HALF_BITS  = 0xBF000000u;
    constexpr uint32_t ONE_F_BITS     = 0x3F800000u;
    constexpr uint32_t T_THRESH_BITS  = 0x38D1B717u;

    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    fill_tile_init();

    cb_reserve_back(CB_CONST_ZERO, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_ZERO);
    tile_regs_release();
    cb_push_back(CB_CONST_ZERO, 1);

    cb_reserve_back(CB_CONST_099, 1);
    tile_regs_acquire();
    fill_tile(0, 0.99f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_099);
    tile_regs_release();
    cb_push_back(CB_CONST_099, 1);

    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_reserve_back(CB_COLOR_R_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_R_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_R_STATE, 1);

        cb_reserve_back(CB_COLOR_G_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_G_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_G_STATE, 1);

        cb_reserve_back(CB_COLOR_B_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 0.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_COLOR_B_STATE);
        tile_regs_release();
        cb_push_back(CB_COLOR_B_STATE, 1);

        cb_reserve_back(CB_T_STATE, 1);
        tile_regs_acquire();
        fill_tile(0, 1.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_T_STATE);
        tile_regs_release();
        cb_push_back(CB_T_STATE, 1);

        cb_wait_front(CB_COLOR_R_STATE, 1);
        cb_wait_front(CB_COLOR_G_STATE, 1);
        cb_wait_front(CB_COLOR_B_STATE, 1);
        cb_wait_front(CB_T_STATE, 1);

        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, 0, 0);
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        // Precompute px², py², px·py once per tile (held for all Gaussians).
        tile_regs_acquire();
        mul_tiles_init(CB_PX, CB_PX);
        mul_tiles(CB_PX, CB_PX, 0, 0, 0);
        mul_tiles_init(CB_PY, CB_PY);
        mul_tiles(CB_PY, CB_PY, 0, 0, 1);
        mul_tiles_init(CB_PX, CB_PY);
        mul_tiles(CB_PX, CB_PY, 0, 0, 2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_PX2, 1);
        pack_tile(0, CB_PX2);
        cb_push_back(CB_PX2, 1);
        cb_reserve_back(CB_PY2, 1);
        pack_tile(1, CB_PY2);
        cb_push_back(CB_PY2, 1);
        cb_reserve_back(CB_PXPY, 1);
        pack_tile(2, CB_PXPY);
        cb_push_back(CB_PXPY, 1);
        tile_regs_release();
        cb_wait_front(CB_PX2, 1);
        cb_wait_front(CB_PY2, 1);
        cb_wait_front(CB_PXPY, 1);

        for (uint32_t g = 0; g < g_count; g++) {
            if (g > 0 && (g & 0xFu) == 0u) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(CB_T_STATE);
                copy_tile(CB_T_STATE, 0, 0);
                unary_ge_tile_init();
                unary_ge_tile(0, T_THRESH_BITS);
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(CB_T_TMP, 1);
                pack_tile(0, CB_T_TMP);
                cb_push_back(CB_T_TMP, 1);
                tile_regs_release();
                cb_wait_front(CB_T_TMP, 1);

                tile_regs_acquire();
                mul_tiles_init(CB_T_STATE, CB_T_TMP);
                mul_tiles(CB_T_STATE, CB_T_TMP, 0, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                cb_pop_front(CB_T_STATE, 1);
                cb_reserve_back(CB_T_STATE, 1);
                pack_tile(0, CB_T_STATE);
                cb_push_back(CB_T_STATE, 1);
                tile_regs_release();
                cb_wait_front(CB_T_STATE, 1);
                cb_pop_front(CB_T_TMP, 1);
            }

            cb_wait_front(CB_SCALARS, 1);

            uint32_t A_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            uint32_t B_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            uint32_t C_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            uint32_t D_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            uint32_t E_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            uint32_t F_bits       = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            uint32_t color_r_bits = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            uint32_t color_g_bits = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            uint32_t color_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 8);
            uint32_t opacity_bits = ckernel::read_tile_value(CB_SCALARS, 0, 9);

            // Pack six scalar-broadcast tiles for FPU mul_tiles_bcast_scalar.
            tile_regs_acquire();
            fill_tile(0, *reinterpret_cast<const float*>(&A_bits));
            fill_tile(1, *reinterpret_cast<const float*>(&B_bits));
            fill_tile(2, *reinterpret_cast<const float*>(&C_bits));
            fill_tile(3, *reinterpret_cast<const float*>(&D_bits));
            fill_tile(4, *reinterpret_cast<const float*>(&E_bits));
            fill_tile(5, *reinterpret_cast<const float*>(&F_bits));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_COEFF, 6);
            pack_tile(0, CB_COEFF);
            pack_tile(1, CB_COEFF);
            pack_tile(2, CB_COEFF);
            pack_tile(3, CB_COEFF);
            pack_tile(4, CB_COEFF);
            pack_tile(5, CB_COEFF);
            cb_push_back(CB_COEFF, 6);
            tile_regs_release();
            cb_wait_front(CB_COEFF, 6);

            // Stage B + C: FPU basis-form quadratic, then SFPU exp/alpha chain.
            tile_regs_acquire();
            mul_tiles_bcast_scalar_init_short(CB_PX2, CB_COEFF);
            mul_tiles_bcast_scalar(CB_PX2, CB_COEFF, 0, 0, 0);

            mul_tiles_bcast_scalar_init_short(CB_PXPY, CB_COEFF);
            mul_tiles_bcast_scalar(CB_PXPY, CB_COEFF, 0, 1, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);

            mul_tiles_bcast_scalar_init_short(CB_PY2, CB_COEFF);
            mul_tiles_bcast_scalar(CB_PY2, CB_COEFF, 0, 2, 1);
            add_binary_tile(0, 1, 0);

            mul_tiles_bcast_scalar_init_short(CB_PX, CB_COEFF);
            mul_tiles_bcast_scalar(CB_PX, CB_COEFF, 0, 3, 1);
            add_binary_tile(0, 1, 0);

            mul_tiles_bcast_scalar_init_short(CB_PY, CB_COEFF);
            mul_tiles_bcast_scalar(CB_PY, CB_COEFF, 0, 4, 1);
            add_binary_tile(0, 1, 0);

            copy_tile_to_dst_init_short(CB_COEFF);
            copy_tile(CB_COEFF, 5, 1);
            add_binary_tile(0, 1, 0);

            mul_unary_tile(0, NEG_HALF_BITS);

            copy_tile_to_dst_init_short(CB_CONST_ZERO);
            copy_tile(CB_CONST_ZERO, 0, 1);
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            exp_tile_init<true>();
            exp_tile<true>(0);

            mul_unary_tile(0, opacity_bits);

            copy_tile_to_dst_init_short(CB_CONST_099);
            copy_tile(CB_CONST_099, 0, 1);
            binary_min_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();

            cb_pop_front(CB_COEFF, 6);
            cb_wait_front(CB_ALPHA, 1);

            tile_regs_acquire();
            mul_tiles_init(CB_ALPHA, CB_T_STATE);
            mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_CONTRIB, 1);
            pack_tile(0, CB_CONTRIB);
            cb_push_back(CB_CONTRIB, 1);
            tile_regs_release();
            cb_wait_front(CB_CONTRIB, 1);

            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            copy_tile(CB_CONTRIB, 0, 1);
            copy_tile(CB_CONTRIB, 0, 2);
            mul_unary_tile(0, color_r_bits);
            mul_unary_tile(1, color_g_bits);
            mul_unary_tile(2, color_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_R, 1);
            pack_tile(0, CB_T_R);
            cb_push_back(CB_T_R, 1);
            cb_reserve_back(CB_T_G, 1);
            pack_tile(1, CB_T_G);
            cb_push_back(CB_T_G, 1);
            cb_reserve_back(CB_T_B, 1);
            pack_tile(2, CB_T_B);
            cb_push_back(CB_T_B, 1);
            tile_regs_release();
            cb_wait_front(CB_T_R, 1);
            cb_wait_front(CB_T_G, 1);
            cb_wait_front(CB_T_B, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_R_STATE, CB_T_R);
            add_tiles(CB_COLOR_R_STATE, CB_T_R, 0, 0, 0);
            add_tiles_init(CB_COLOR_G_STATE, CB_T_G);
            add_tiles(CB_COLOR_G_STATE, CB_T_G, 0, 0, 1);
            add_tiles_init(CB_COLOR_B_STATE, CB_T_B);
            add_tiles(CB_COLOR_B_STATE, CB_T_B, 0, 0, 2);
            sub_tiles_init(CB_T_STATE, CB_CONTRIB);
            sub_tiles(CB_T_STATE, CB_CONTRIB, 0, 0, 3);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_R_STATE, 1);
            cb_reserve_back(CB_COLOR_R_STATE, 1);
            pack_tile(0, CB_COLOR_R_STATE);
            cb_push_back(CB_COLOR_R_STATE, 1);
            cb_pop_front(CB_COLOR_G_STATE, 1);
            cb_reserve_back(CB_COLOR_G_STATE, 1);
            pack_tile(1, CB_COLOR_G_STATE);
            cb_push_back(CB_COLOR_G_STATE, 1);
            cb_pop_front(CB_COLOR_B_STATE, 1);
            cb_reserve_back(CB_COLOR_B_STATE, 1);
            pack_tile(2, CB_COLOR_B_STATE);
            cb_push_back(CB_COLOR_B_STATE, 1);
            cb_pop_front(CB_T_STATE, 1);
            cb_reserve_back(CB_T_STATE, 1);
            pack_tile(3, CB_T_STATE);
            cb_push_back(CB_T_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_R_STATE, 1);
            cb_wait_front(CB_COLOR_G_STATE, 1);
            cb_wait_front(CB_COLOR_B_STATE, 1);
            cb_wait_front(CB_T_STATE, 1);
            cb_pop_front(CB_T_R, 1);
            cb_pop_front(CB_T_G, 1);
            cb_pop_front(CB_T_B, 1);
            cb_pop_front(CB_CONTRIB, 1);

            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_COLOR_R_STATE);
        copy_tile(CB_COLOR_R_STATE, 0, 0);
        copy_tile_to_dst_init_short(CB_COLOR_G_STATE);
        copy_tile(CB_COLOR_G_STATE, 0, 1);
        copy_tile_to_dst_init_short(CB_COLOR_B_STATE);
        copy_tile(CB_COLOR_B_STATE, 0, 2);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        cb_pop_front(CB_COLOR_R_STATE, 1);
        cb_pop_front(CB_COLOR_G_STATE, 1);
        cb_pop_front(CB_COLOR_B_STATE, 1);
        cb_pop_front(CB_T_STATE, 1);
        cb_pop_front(CB_PX2, 1);
        cb_pop_front(CB_PY2, 1);
        cb_pop_front(CB_PXPY, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
