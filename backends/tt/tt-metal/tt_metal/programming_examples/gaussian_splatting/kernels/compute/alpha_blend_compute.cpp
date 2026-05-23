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

// Alpha-blend compute kernel — iter 070c: single acquire per tile.
//
// KEY INSIGHT: In SyncFull mode, tile_regs_release() ALWAYS calls
// llk_pack_dest_section_done() which fires TTI_ZEROACC, clearing ALL Dst
// tiles — regardless of whether pack_tile was called.  Therefore, state
// CANNOT survive intermediate releases.
//
// FIX: Use a SINGLE acquire/release per screen tile.  State in Dst[0..3]
// persists through the entire per-Gaussian loop because there is no
// intermediate release (and therefore no ZEROACC) until the final pack.
//
// HIGH-LEVEL FLOW
// ----------------
//   tile_regs_acquire()
//   Init: R=G=B=0, T=1 in Dst[0..3]
//   For each Gaussian g (sorted front-to-back):
//     Compute lx², ly², lxly inline from CB_PX, CB_PY
//     power = A·lx² + B·lx·ly + C·ly² + D'·lx + E'·ly + F'
//     alpha = opacity · exp(power)
//     contrib = alpha · T
//     T -= contrib
//     R/G/B += contrib · color
//   tile_regs_commit/wait; pack Dst[0..2] → CB_COLOR_OUT
//   tile_regs_release()   ← ONLY release; ZEROACC fires here (done, OK)
//
// DST SLOT USAGE (8 tiles with dst_full_sync_en=true, fp32_dest_acc_en=true)
// --------------------------------------------------------------------------
//   Dst[0] = R_state  |
//   Dst[1] = G_state  | persistent throughout all Gaussians (no release)
//   Dst[2] = B_state  |
//   Dst[3] = T_state  |
//   Dst[4] = power accumulator / scratch
//   Dst[5] = lx, D'lx, ly², C·ly², contrib, contrib·color (reused)
//   Dst[6] = ly, E'ly, lx·ly, contrib backup (reused)
//   Dst[7] = lx·ly (temp), lx², A·lx², B·lx·ly (reused)
//
// No precompute block — lx², ly², lxly computed inline every Gaussian.
// CB_LX2, CB_LY2, CB_LXLY not used.
//
// RUNTIME ARGS
//   0: num_tiles
//
// CB INDICES — must match alpha_blend_host.h.
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_COLOR_OUT = 16;

    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    for (uint32_t t = 0; t < num_tiles; t++) {

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, 0, 0);
        cb_pop_front(CB_TILE_META, 1);

        // ================================================================
        // SINGLE acquire covers state init + ALL Gaussians + output pack.
        // No intermediate releases → no ZEROACC → Dst[0..3] persists.
        // ================================================================
        tile_regs_acquire();

        // State init: R=G=B=0, T=1
        fill_tile_init();
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        fill_tile(2, 0.0f);
        fill_tile(3, 1.0f);

        for (uint32_t g = 0; g < g_count; g++) {
            cb_wait_front(CB_SCALARS, 1);

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

            // --- Load lx, ly; compute lxly and lx² inline ---
            // Dst[5] = lx,  Dst[6] = ly
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 5);
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 6);

            mul_binary_tile_init();
            mul_binary_tile(5, 6, 7);   // Dst[7] = lxly    [Dst[5]=lx, Dst[6]=ly]
            mul_binary_tile(5, 5, 4);   // Dst[4] = lx²     [Dst[5]=lx still]

            // A·lx², B·lxly
            mul_unary_tile(4, cov_a_bits);       // Dst[4] = A·lx²
            mul_unary_tile(7, two_cov_b_bits);  // Dst[7] = B·lxly

            // D'·lx: overwrite Dst[5] (lx no longer needed after this)
            mul_unary_tile(5, d_prime_bits);     // Dst[5] = D'·lx
            add_binary_tile_init();
            add_binary_tile(4, 5, 4);            // Dst[4] = A·lx² + D'·lx

            // C·ly²: Dst[6]=ly still valid; put ly² in Dst[5]
            mul_binary_tile_init();
            mul_binary_tile(6, 6, 5);            // Dst[5] = ly²   [Dst[6]=ly]
            mul_unary_tile(5, cov_c_bits);        // Dst[5] = C·ly²

            // Accumulate B·lxly, C·ly²
            add_binary_tile_init();
            add_binary_tile(4, 7, 4);            // Dst[4] += B·lxly
            add_binary_tile(4, 5, 4);            // Dst[4] += C·ly²

            // E'·ly: overwrite Dst[6] (ly no longer needed)
            mul_unary_tile(6, e_prime_bits);     // Dst[6] = E'·ly
            add_binary_tile_init();
            add_binary_tile(4, 6, 4);            // Dst[4] += E'·ly
            add_unary_tile(4, f_prime_bits);      // Dst[4] = power

            // alpha = opacity · exp(power)
            exp_tile_init<true>();
            exp_tile<true>(4);
            mul_unary_tile(4, opacity_bits);

            // contrib = alpha · T_state  →  Dst[5]
            mul_binary_tile_init();
            mul_binary_tile(4, 3, 5);            // Dst[5] = contrib

            // T_state -= contrib
            sub_binary_tile_init();
            sub_binary_tile(3, 5, 3);            // Dst[3] = new T

            // R/G/B += contrib · color  (contrib backup in Dst[6])
            copy_dest_values_init();
            copy_dest_values(5, 6);              // Dst[6] = contrib backup
            mul_unary_tile(5, color_r_bits);
            add_binary_tile_init();
            add_binary_tile(0, 5, 0);            // Dst[0] += contrib·R

            copy_dest_values_init();
            copy_dest_values(6, 5);              // Dst[5] = contrib
            mul_unary_tile(5, color_g_bits);
            add_binary_tile_init();
            add_binary_tile(1, 5, 1);            // Dst[1] += contrib·G

            copy_dest_values_init();
            copy_dest_values(6, 5);              // Dst[5] = contrib
            mul_unary_tile(5, color_b_bits);
            add_binary_tile_init();
            add_binary_tile(2, 5, 2);            // Dst[2] += contrib·B

            cb_pop_front(CB_SCALARS, 1);
        }

        // Pack final R/G/B accumulators
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();   // ZEROACC fires here — we're done with this tile

        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
