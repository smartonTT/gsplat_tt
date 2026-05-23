// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
#include "api/compute/copy_dest_values.h"

// Alpha-blend compute kernel — iter 070: Dst-resident R/G/B/T state.
//
// HIGH-LEVEL FLOW
// ----------------
// For each screen tile this core owns, with running per-pixel accumulator
// state (R/G/B color, transmittance T):
//
//   Init      : R = G = B = 0,  T = 1   (per pixel, loaded into Dst[0..3])
//   Precompute: lx² → CB_LX2, ly² → CB_LY2, lx·ly → CB_LXLY (once per tile)
//   For each Gaussian g in this tile, sorted front-to-back:
//     Stage B+C: power = A·lx² + B·lx·ly + C·ly² + D'·lx + E'·ly + F'
//                alpha = opacity · exp(power)
//     Stage D1: contrib = alpha · T_state           (Dst mul)
//     Stage E:  T_state -= contrib                  (Dst sub)
//     Stage D2: R/G/B_state += contrib · color_c    (Dst copy+mul+add ×3)
//     — all in ONE acquire/release per Gaussian, no state CB spills
//   Output    : pack Dst[0..2] → CB_COLOR_OUT (3 tiles)
//
// DST SLOT LAYOUT (iter 070, dst_full_sync_en=true → 8 tiles with fp32_dest_acc_en=true)
// ---------------------------------------------------------------------------------------
//   Dst[0] = R_state   \
//   Dst[1] = G_state    | persistent across all Gaussians for a tile;
//   Dst[2] = B_state    | not packed in inner loop → preserved by SyncFull mode
//   Dst[3] = T_state   /
//   Dst[4] = power accumulator → alpha (scratch, per-Gaussian)
//   Dst[5] = scratch A: B·lx·ly, D'·lx, E'·ly, contrib, color·contrib
//   Dst[6] = scratch B: C·ly², contrib copy for G/B channels
//   Dst[7] = scratch C: A·lx²
//
// EXECUTION MODEL
// ----------------
// - dst_full_sync_en=true: SyncFull mode. Pack only reads slots you call
//   pack_tile on; non-packed slots (0..3 state) survive across acquire/release.
// - fp32_dest_acc_en=true: 8 Dst tiles in SyncFull mode.
//
// RUNTIME ARGS
//   0: num_tiles  -- number of screen tiles this core processes
//
// CB INDICES — must match alpha_blend_host.h.
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX            = 0;
    constexpr uint32_t CB_PY            = 1;
    constexpr uint32_t CB_SCALARS       = 2;
    constexpr uint32_t CB_TILE_META     = 3;
    constexpr uint32_t CB_LX2           = 4;   // precomputed lx² per tile
    constexpr uint32_t CB_LY2           = 5;   // precomputed ly² per tile
    constexpr uint32_t CB_LXLY          = 9;   // precomputed lx·ly per tile (slot 9)
    constexpr uint32_t CB_COLOR_OUT     = 16;

    binary_op_init_common(CB_PX, CB_LX2, CB_COLOR_OUT);

    for (uint32_t t = 0; t < num_tiles; t++) {

        // =====================================================================
        // Per-tile state init: Dst[0..3] = R=0, G=0, B=0, T=1.
        // In SyncFull mode, Pack touches only explicitly packed slots.
        // We pack nothing here → state persists in Dst[0..3] for all Gaussians.
        // =====================================================================
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0f);   // Dst[0] = R_state = 0
        fill_tile(1, 0.0f);   // Dst[1] = G_state = 0
        fill_tile(2, 0.0f);   // Dst[2] = B_state = 0
        fill_tile(3, 1.0f);   // Dst[3] = T_state = 1
        tile_regs_commit();
        tile_regs_wait();
        // No pack_tile calls → Pack reads nothing → Dst[0..3] unchanged after release
        tile_regs_release();

        // =====================================================================
        // Per-tile precompute: lx², ly², lx·ly → CBs (reused every Gaussian).
        // Uses Dst[4..7] as scratch (state in Dst[0..3] is preserved).
        // =====================================================================
        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_PX);
        copy_tile(CB_PX, 0, 4);         // Dst[4] = lx
        copy_tile_to_dst_init_short(CB_PY);
        copy_tile(CB_PY, 0, 5);         // Dst[5] = ly
        mul_binary_tile_init();
        mul_binary_tile(4, 5, 6);       // Dst[6] = lx·ly
        mul_binary_tile(4, 4, 7);       // Dst[7] = lx²  (Dst[4]=lx still valid)
        mul_binary_tile(5, 5, 4);       // Dst[4] = ly²  (overwrites lx in slot 4)
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_LX2, 1);  pack_tile(7, CB_LX2);  cb_push_back(CB_LX2, 1);
        cb_reserve_back(CB_LY2, 1);  pack_tile(4, CB_LY2);  cb_push_back(CB_LY2, 1);
        cb_reserve_back(CB_LXLY, 1); pack_tile(6, CB_LXLY); cb_push_back(CB_LXLY, 1);
        tile_regs_release();

        cb_wait_front(CB_LX2, 1);
        cb_wait_front(CB_LY2, 1);
        cb_wait_front(CB_LXLY, 1);

        // Read per-tile Gaussian count.
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_TILE_META, 1);

        // =====================================================================
        // Per-Gaussian inner loop — single acquire/release per Gaussian.
        //
        // Dst[0..3] hold persistent state across iterations (SyncFull: Pack
        // only reads slots we explicitly call pack_tile on; state slots are
        // never packed here, so they survive each acquire/release cycle).
        // =====================================================================
        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);

            // Read 10 basis-form scalars: [A, B, C, D', E', F', R, G, B, opacity]
            uint32_t cov_a_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            uint32_t two_cov_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            uint32_t cov_c_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            uint32_t d_prime_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            uint32_t e_prime_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            uint32_t f_prime_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            uint32_t color_r_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            uint32_t color_g_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            uint32_t color_b_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 8);
            uint32_t opacity_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 9);

            // ----- Single acquire block: B+C, D1, E, D2 -----
            //
            // power = A·lx² + B·lx·ly + C·ly² + D'·lx + E'·ly + F'
            // alpha = opacity · exp(power)
            // contrib = alpha · T_state    [D1]
            // T_state -= contrib           [E]
            // R/G/B_state += contrib · c   [D2]
            //
            // Dst slot usage:
            //   [4] = power accumulator → alpha (overwrites scratch)
            //   [5] = scratch: B·lx·ly, D'·lx, E'·ly, contrib, color·contrib
            //   [6] = scratch: C·ly², then contrib copy for G/B
            //   [7] = scratch: A·lx²
            tile_regs_acquire();

            // Stage B+C quadratic terms → Dst[4] = power
            copy_tile_to_dst_init_short(CB_LX2);
            copy_tile(CB_LX2, 0, 7);              // Dst[7] = lx²
            mul_unary_tile(7, cov_a_bits);         // Dst[7] = A·lx²
            copy_tile_to_dst_init_short(CB_LXLY);
            copy_tile(CB_LXLY, 0, 5);             // Dst[5] = lx·ly
            mul_unary_tile(5, two_cov_b_bits);    // Dst[5] = B·lx·ly
            copy_tile_to_dst_init_short(CB_LY2);
            copy_tile(CB_LY2, 0, 6);              // Dst[6] = ly²
            mul_unary_tile(6, cov_c_bits);        // Dst[6] = C·ly²
            add_binary_tile_init();
            add_binary_tile(7, 5, 4);             // Dst[4] = A·lx² + B·lx·ly
            add_binary_tile(4, 6, 4);             // Dst[4] += C·ly²
            // Linear correction terms
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 5);               // Dst[5] = lx
            mul_unary_tile(5, d_prime_bits);       // Dst[5] = D'·lx
            add_binary_tile_init();
            add_binary_tile(4, 5, 4);             // Dst[4] += D'·lx
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 5);               // Dst[5] = ly
            mul_unary_tile(5, e_prime_bits);       // Dst[5] = E'·ly
            add_binary_tile_init();
            add_binary_tile(4, 5, 4);             // Dst[4] += E'·ly
            add_unary_tile(4, f_prime_bits);       // Dst[4] = power

            // Stage C: alpha = opacity · exp(power)
            exp_tile_init<true>();
            exp_tile<true>(4);                     // Dst[4] = exp(power)
            mul_unary_tile(4, opacity_bits);       // Dst[4] = alpha

            // Stage D1: contrib = alpha · T_state
            mul_binary_tile_init();
            mul_binary_tile(4, 3, 5);             // Dst[5] = alpha · Dst[3](T) = contrib

            // Stage E: T_state -= contrib  (transmittance front-to-back update)
            sub_binary_tile_init();
            sub_binary_tile(3, 5, 3);             // Dst[3] = T_state - contrib

            // Stage D2: R/G/B_state += contrib · color_c
            // contrib is in Dst[5]; copy to Dst[6] as a spare so we can multiply
            // Dst[5] in-place by color_R, then restore contrib from Dst[6] for G/B.
            copy_dest_values_init();
            copy_dest_values(5, 6);               // Dst[6] = contrib (backup)
            mul_unary_tile(5, color_r_bits);       // Dst[5] = contrib · R
            add_binary_tile_init();
            add_binary_tile(0, 5, 0);             // Dst[0] = R_state + contrib·R

            copy_dest_values_init();
            copy_dest_values(6, 5);               // Dst[5] = contrib (restore from Dst[6])
            mul_unary_tile(5, color_g_bits);       // Dst[5] = contrib · G
            add_binary_tile_init();
            add_binary_tile(1, 5, 1);             // Dst[1] = G_state + contrib·G

            copy_dest_values_init();
            copy_dest_values(6, 5);               // Dst[5] = contrib (again from Dst[6])
            mul_unary_tile(5, color_b_bits);       // Dst[5] = contrib · B
            add_binary_tile_init();
            add_binary_tile(2, 5, 2);             // Dst[2] = B_state + contrib·B

            // Commit + wait: Pack reads nothing → state in Dst[0..3] is preserved.
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();

            cb_pop_front(CB_SCALARS, 1);
        }

        // =====================================================================
        // Per-tile finalize: Dst[0..2] hold final R/G/B accumulators.
        // Pack them to CB_COLOR_OUT. Acquire a trivial block so Pack can read.
        // =====================================================================
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        // Drain per-tile CBs.
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
        cb_pop_front(CB_LX2, 1);
        cb_pop_front(CB_LY2, 1);
        cb_pop_front(CB_LXLY, 1);
    }
}
