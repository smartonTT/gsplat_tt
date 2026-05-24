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

// Alpha-blend compute kernel — iter 071: precompute lx²/ly²/lxly once per tile
// combined with single-acquire Dst-resident state (from iter 070c).
//
// INSIGHT: iter 070c recomputed lx²/ly²/lxly via SFPU mul_binary_tile for every
// Gaussian (expensive, ~3 SFPU binary ops per Gaussian).  Moving this to a
// SEPARATE precompute acquire — which runs BEFORE the state acquire and fires its
// own ZEROACC on release (fine, no state yet) — replaces those 3 SFPU binary ops
// with 3 FPU copy_tile reads from the precomputed L1 CBs.  FPU is ~5× faster.
//
// HIGH-LEVEL FLOW
// ---------------
//   [PRECOMPUTE BLOCK — separate acquire, no state to lose]
//   tile_regs_acquire()
//     Dst[2] = lx²  (from CB_PX × CB_PX)
//     Dst[3] = ly²  (from CB_PY × CB_PY)
//     Dst[4] = lxly (from CB_PX × CB_PY)
//   tile_regs_commit/wait
//   Pack Dst[2,3,4] → CB_DX2, CB_DY2, CB_DXDY (reusing old CB slots)
//   tile_regs_release()  ← ZEROACC here is OK (no state yet)
//
//   [STATE LOOP — single acquire, no intermediate release]
//   tile_regs_acquire()
//   Init Dst[0..3] = R=G=B=0, T=1
//   For each Gaussian g:
//     Read A,B,C,D',E',F',R,G,B,opacity from CB_SCALARS
//     FPU copy: lx²→Dst[7], lxly→Dst[6], ly²→Dst[5] from precomputed CBs
//     mul_unary × 3: A·lx², B·lxly, C·ly²  [SFPU scalar]
//     add_binary × 2: accumulate A·lx²+B·lxly+C·ly²  [SFPU]
//     FPU copy: lx→Dst[5] from CB_PX  → mul_unary D'·lx  [SFPU] → add to Dst[4]
//     FPU copy: ly→Dst[5] from CB_PY  → mul_unary E'·ly  [SFPU] → add to Dst[4]
//     add_unary F': Dst[4] = power
//     exp(power), ×opacity → alpha
//     contrib=alpha·T; T-=contrib; R/G/B+=contrib·color
//   tile_regs_commit/wait; pack Dst[0..2] → CB_COLOR_OUT
//   tile_regs_release()  ← ONLY release in state loop
//
// DST SLOT USAGE (8 tiles, dst_full_sync_en=true, fp32_dest_acc_en=true)
// -----------------------------------------------------------------------
//   Precompute block (slots 0..4 used temporarily, then packed out):
//     Dst[2]=lx², Dst[3]=ly², Dst[4]=lxly
//   State loop (8 slots):
//     Dst[0]=R, Dst[1]=G, Dst[2]=B, Dst[3]=T  (persistent)
//     Dst[4]=power/scratch, Dst[5]=scratch, Dst[6]=scratch, Dst[7]=lx²/scratch
//
// CB REUSE (precomputed tiles — depth=1, popped after all Gaussians):
//   CB_DX2  (6) ← lx²   (previously: dx²)
//   CB_DY2  (7) ← ly²   (previously: dy²)
//   CB_DXDY (8) ← lxly  (previously: dx·dy)
//
// RUNTIME ARGS
//   0: num_tiles
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;
    constexpr uint32_t CB_LX2       = 6;   // reusing CB_DX2 slot
    constexpr uint32_t CB_LY2       = 7;   // reusing CB_DY2 slot
    constexpr uint32_t CB_LXLY      = 8;   // reusing CB_DXDY slot
    constexpr uint32_t CB_COLOR_OUT = 16;

    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    for (uint32_t t = 0; t < num_tiles; t++) {

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, 0, 0);
        cb_pop_front(CB_TILE_META, 1);

        // ================================================================
        // PRECOMPUTE BLOCK — compute lx², ly², lxly once per screen tile.
        // Uses a separate acquire; ZEROACC on release is fine (no state yet).
        // ================================================================
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_PX);
        copy_tile(CB_PX, 0, 0);   // Dst[0] = lx
        copy_tile_to_dst_init_short(CB_PY);
        copy_tile(CB_PY, 0, 1);   // Dst[1] = ly

        mul_binary_tile_init();
        mul_binary_tile(0, 0, 2); // Dst[2] = lx²
        mul_binary_tile(1, 1, 3); // Dst[3] = ly²
        mul_binary_tile(0, 1, 4); // Dst[4] = lxly

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_LX2, 1);
        pack_tile(2, CB_LX2);
        cb_push_back(CB_LX2, 1);
        cb_reserve_back(CB_LY2, 1);
        pack_tile(3, CB_LY2);
        cb_push_back(CB_LY2, 1);
        cb_reserve_back(CB_LXLY, 1);
        pack_tile(4, CB_LXLY);
        cb_push_back(CB_LXLY, 1);
        tile_regs_release();   // ZEROACC fires — OK, state loop not started yet

        cb_wait_front(CB_LX2, 1);
        cb_wait_front(CB_LY2, 1);
        cb_wait_front(CB_LXLY, 1);

        // ================================================================
        // STATE LOOP — single acquire covers ALL Gaussians + output pack.
        // No intermediate releases → no ZEROACC → Dst[0..3] persists.
        // ================================================================
        tile_regs_acquire();

        fill_tile_init();
        fill_tile(0, 0.0f);   // R
        fill_tile(1, 0.0f);   // G
        fill_tile(2, 0.0f);   // B
        fill_tile(3, 1.0f);   // T

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

            // --- Load precomputed lx², ly², lxly (FPU, fast) ---
            copy_tile_to_dst_init_short(CB_LX2);
            copy_tile(CB_LX2, 0, 7);   // Dst[7] = lx²
            copy_tile_to_dst_init_short(CB_LXLY);
            copy_tile(CB_LXLY, 0, 6);  // Dst[6] = lxly
            copy_tile_to_dst_init_short(CB_LY2);
            copy_tile(CB_LY2, 0, 5);   // Dst[5] = ly²

            // --- A·lx², B·lxly, C·ly² (SFPU scalar) ---
            mul_unary_tile(7, cov_a_bits);       // Dst[7] = A·lx²
            mul_unary_tile(6, two_cov_b_bits);   // Dst[6] = B·lxly
            mul_unary_tile(5, cov_c_bits);       // Dst[5] = C·ly²

            // --- Accumulate into Dst[4] ---
            add_binary_tile_init();
            add_binary_tile(7, 6, 4);  // Dst[4] = A·lx² + B·lxly
            add_binary_tile(4, 5, 4);  // Dst[4] += C·ly²

            // --- D'·lx: load lx from CB_PX, multiply, add ---
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 5);    // Dst[5] = lx
            mul_unary_tile(5, d_prime_bits);  // Dst[5] = D'·lx
            add_binary_tile_init();
            add_binary_tile(4, 5, 4);  // Dst[4] += D'·lx

            // --- E'·ly: load ly from CB_PY, multiply, add ---
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 5);    // Dst[5] = ly
            mul_unary_tile(5, e_prime_bits);  // Dst[5] = E'·ly
            add_binary_tile_init();
            add_binary_tile(4, 5, 4);  // Dst[4] += E'·ly
            add_unary_tile(4, f_prime_bits);  // Dst[4] = power

            // --- alpha = opacity · exp(power) ---
            exp_tile_init<true>();
            exp_tile<true>(4);
            mul_unary_tile(4, opacity_bits);   // Dst[4] = alpha

            // --- contrib = alpha · T_state ---
            mul_binary_tile_init();
            mul_binary_tile(4, 3, 5);  // Dst[5] = contrib

            // --- T_state -= contrib ---
            sub_binary_tile_init();
            sub_binary_tile(3, 5, 3);  // Dst[3] = new T

            // --- R/G/B += contrib · color (contrib backup in Dst[6]) ---
            copy_dest_values_init();
            copy_dest_values(5, 6);    // Dst[6] = contrib backup
            mul_unary_tile(5, color_r_bits);
            add_binary_tile_init();
            add_binary_tile(0, 5, 0);  // Dst[0] += contrib·R

            copy_dest_values_init();
            copy_dest_values(6, 5);    // Dst[5] = contrib
            mul_unary_tile(5, color_g_bits);
            add_binary_tile_init();
            add_binary_tile(1, 5, 1);  // Dst[1] += contrib·G

            copy_dest_values_init();
            copy_dest_values(6, 5);    // Dst[5] = contrib
            mul_unary_tile(5, color_b_bits);
            add_binary_tile_init();
            add_binary_tile(2, 5, 2);  // Dst[2] += contrib·B

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
        tile_regs_release();   // ZEROACC fires — done with this tile

        cb_pop_front(CB_LX2, 1);
        cb_pop_front(CB_LY2, 1);
        cb_pop_front(CB_LXLY, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
