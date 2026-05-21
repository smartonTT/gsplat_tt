// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
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
//   For each Gaussian g in this tile, sorted front-to-back:
//     Stage F : every 16 g's (skip g=0): hard-zero T where T < 1e-4
//               -> "freeze" saturated pixels in T_state directly
//     Stage A : read 9 fp32 scalars (mean, cov_inv, color, opacity)
//     Stage B1: dx = px - mean_x,  dy = py - mean_y         (per-pixel offset)
//     Stage B2+B3a: weighted quadratic terms → CB_Q
//               [a·dx², c·dy², 2b·dx·dy] (no intermediate CBs)
//     Stage B3: Q = a·dx² + 2b·dx·dy + c·dy²                (Mahalanobis dist)
//               power = -0.5·Q                              (Gaussian exponent)
//     Stage C : weight  = exp(min(power, 0))                (Gaussian falloff)
//               alpha   = min(opacity · weight, 0.99)       (per-pixel opacity)
//     Stage D1: contrib = alpha · T_state                   (T pre-masked in Stage F)
//     Stage D2: for c in {R,G,B}: c_state += color_c · contrib
//     Stage E : T ← T · (1 - alpha)                         (transmittance update)
//   Output    : pack R_state, G_state, B_state to CB_COLOR_OUT (3 tiles).
//   Cleanup   : pop state CBs so the next tile's iteration starts fresh.
//
// EXECUTION MODEL
// ----------------
// - tt-metal compute kernel running on one Tensix core; reader (BRISC) and
//   writer (NCRISC) on the same core handle NoC traffic via circular buffers.
// - SFPU operates on tiles of 32x32 elements as 4 vector passes of 32 lanes
//   each. All math here is per-pixel, broadcast across the tile's pixels.
// - Dst (destination register file) holds fp32 working values during a single
//   acquire/commit/release block; we spill back to L1 CBs between Gaussians
//   because Dst is too small to hold all running state across the loop.
//
// RUNTIME ARGS
//   0: num_tiles  -- number of screen tiles this core processes
//
// CB INDICES — see alpha_blend_host.h for the canonical declaration. The
// constexprs below mirror those values (compute kernels can't include the
// host-side namespace, so we duplicate; keep them in sync).
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // CB indices — must match alpha_blend_host.h verbatim. Compute kernels
    // can't `#include` host-side headers, so we duplicate the constants here.
    constexpr uint32_t CB_PX            = 0;   // px tile (per-pixel x coord)
    constexpr uint32_t CB_PY            = 1;   // py tile (per-pixel y coord)
    constexpr uint32_t CB_SCALARS       = 2;   // 9 fp32 scalars per Gaussian (64-byte page)
    constexpr uint32_t CB_TILE_META     = 3;   // 1 uint32 (g_count) per screen tile
    constexpr uint32_t CB_DX            = 4;   // dx = px - mean_x
    constexpr uint32_t CB_DY            = 5;   // dy = py - mean_y
    constexpr uint32_t CB_DX2           = 6;   // dx²
    constexpr uint32_t CB_DY2           = 7;   // dy²
    constexpr uint32_t CB_DXDY          = 8;   // dx·dy
    constexpr uint32_t CB_Q             = 9;   // [a·dx², c·dy², 2b·dx·dy] (3 tiles)
    constexpr uint32_t CB_POWER         = 10;  // partial sum a·dx² + c·dy²
    // CB 11 reserved (was -88 clamp tile; now unused after exp_tile<approx=true>)
    constexpr uint32_t CB_ALPHA         = 12;  // per-pixel alpha
    constexpr uint32_t CB_CONTRIB       = 13;  // contrib = alpha · T_state
    constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
    constexpr uint32_t CB_T_TMP         = 15;  // generic intermediate
    constexpr uint32_t CB_COLOR_OUT     = 16;  // R, G, B output tiles per screen tile
    constexpr uint32_t CB_COLOR_R_STATE = 17;  // running R accumulator
    constexpr uint32_t CB_COLOR_G_STATE = 18;
    constexpr uint32_t CB_COLOR_B_STATE = 19;
    constexpr uint32_t CB_T_STATE       = 20;  // running transmittance per pixel
    constexpr uint32_t CB_SAT_MASK      = 21;  // unused (host still allocates)
    constexpr uint32_t CB_CONST_ZERO    = 22;  // constant 0.0 tile
    constexpr uint32_t CB_CONST_099     = 23;  // constant 0.99 tile

    // Bit-pattern fp32 constants for SFPU scalar-unary ops (mul_unary_tile,
    // sub_unary_tile, rsub_unary_tile, etc.) which take their immediate as a
    // uint32 bit-cast of the float they want.
    constexpr uint32_t NEG_HALF_BITS  = 0xBF000000u;  // fp32(-0.5)
    constexpr uint32_t ONE_F_BITS     = 0x3F800000u;  // fp32( 1.0)
    constexpr uint32_t T_THRESH_BITS  = 0x38D1B717u;  // fp32(1e-4) — T saturation threshold

    // Foundational SFPU/FPU init: configures unpack and pack hardware for
    // binary tile ops on this core. Must come before any tile op.
    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    // ----- Pre-fill the two constant tiles used by the inner loop. -----
    // These are pushed once per kernel invocation, never popped; the kernel
    // copies them into Dst whenever it needs a 0.0 or 0.99 operand for
    // binary_min_tile.
    fill_tile_init();

    // CB_CONST_ZERO ← tile of 0.0
    cb_reserve_back(CB_CONST_ZERO, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_ZERO);
    tile_regs_release();
    cb_push_back(CB_CONST_ZERO, 1);

    // CB_CONST_099 ← tile of 0.99 (the alpha clamp ceiling)
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
        // =====================================================================
        // Per-tile state CB init: zero the color accumulators, set transmittance
        // to 1.0. Each state CB lives
        // at depth 1; we push its initial value here and pop it at the end of
        // this iteration. Mid-loop the kernel "spills" updated values back to
        // these CBs by pop_front + reserve_back + pack + push_back (since the
        // Dst register file is too small to hold per-pixel state across
        // many Gaussian iterations).
        // =====================================================================

        // R_state, G_state, B_state = 0.0
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

        // T_state = 1.0
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

        // Read the per-tile Gaussian count the reader wrote into CB_TILE_META.
        // This tells us how many entries from CB_SCALARS we'll consume in the
        // inner loop below. Pop it immediately so the slot is free for the
        // reader to fill ahead with the *next* tile's count while we work.
        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_TILE_META, 1);

        // PX/PY tiles: the reader pushes one of each per screen tile (they
        // hold per-pixel global screen coords). We hold them at front for
        // the duration of this tile's Gaussian loop and pop at the end.
        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        // =====================================================================
        // Per-Gaussian inner loop. The reader has pushed g_count fp32 packs
        // into CB_SCALARS; we consume one pack per iteration in this strict
        // front-to-back order (already sorted on the host).
        // =====================================================================
        for (uint32_t g = 0; g < g_count; g++) {
            // ----- Stage F: hard-zero T_state where T < 1e-4 (every 16 g's, skip g=0) -----
            // Saturated pixels get T=0 in CB_T_STATE; downstream D1/E muls by T
            // then naturally produce zero contribution. g=0 skipped (T freshly 1).
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

            // ----- Stage A: read this Gaussian's 9 fp32 attributes from CB_SCALARS.
            // Layout (set on the host in prepare_kernel_inputs):
            //   [mean_x, mean_y, cov_a, 2·cov_b, cov_c, R, G, B, opacity]
            // We keep them as bit-pattern uint32 because that's the form
            // the SFPU scalar-unary ops (mul_unary_tile, sub_unary_tile, etc.)
            // expect their immediate in. The bytes themselves are valid fp32.
            uint32_t mean_x_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            uint32_t mean_y_bits    = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            uint32_t cov_a_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            uint32_t two_cov_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            uint32_t cov_c_bits     = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            uint32_t color_r_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            uint32_t color_g_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            uint32_t color_b_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            uint32_t opacity_bits   = ckernel::read_tile_value(CB_SCALARS, 0, 8);

            // ----- Stage B1: per-pixel offsets from the Gaussian center.
            // Loads the px/py tiles into Dst slots 0/1 and subtracts the
            // Gaussian center scalar (mean_x, mean_y) lane-wise. After this:
            //   Dst[0][i] = px[i] - mean_x
            //   Dst[1][i] = py[i] - mean_y
            // Pack each back to its own CB so subsequent stages can read them
            // as binary operands (mul_tiles, etc., need CB inputs).
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 0);
            sub_unary_tile(0, mean_x_bits);
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 1);
            sub_unary_tile(1, mean_y_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DX, 1);
            pack_tile(0, CB_DX);
            cb_push_back(CB_DX, 1);
            cb_reserve_back(CB_DY, 1);
            pack_tile(1, CB_DY);
            cb_push_back(CB_DY, 1);
            tile_regs_release();
            cb_wait_front(CB_DX, 1);
            cb_wait_front(CB_DY, 1);

            // ----- Stage B2+B3a: dx²·a, dy²·c, dx·dy·2b → CB_Q (fused).
            // One acquire block: mul products in Dst[0..2], scale by covariance
            // scalars, pack directly to CB_Q — no CB_DX2/DY2/DXDY round-trips.
            //   CB_Q[0] = a · dx²
            //   CB_Q[1] = c · dy²
            //   CB_Q[2] = 2b · dx·dy
            tile_regs_acquire();
            mul_tiles_init(CB_DX, CB_DX);
            mul_tiles(CB_DX, CB_DX, 0, 0, 0);
            mul_unary_tile(0, cov_a_bits);

            mul_tiles_init(CB_DY, CB_DY);
            mul_tiles(CB_DY, CB_DY, 0, 0, 1);
            mul_unary_tile(1, cov_c_bits);

            mul_tiles_init(CB_DX, CB_DY);
            mul_tiles(CB_DX, CB_DY, 0, 0, 2);
            mul_unary_tile(2, two_cov_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_Q, 3);
            pack_tile(0, CB_Q);
            pack_tile(1, CB_Q);
            pack_tile(2, CB_Q);
            cb_push_back(CB_Q, 3);
            tile_regs_release();
            cb_wait_front(CB_Q, 3);

            // ----- Stage B3b1: partial sum a·dx² + c·dy² → CB_POWER.
            // Just adding CB_Q[0] + CB_Q[1]. The third term (2b·dx·dy)
            // is folded in within the next acquire block (B3b2+C) below.
            tile_regs_acquire();
            add_tiles_init(CB_Q, CB_Q);
            add_tiles(CB_Q, CB_Q, 0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_POWER, 1);
            pack_tile(0, CB_POWER);
            cb_push_back(CB_POWER, 1);
            tile_regs_release();
            cb_wait_front(CB_POWER, 1);

            // ----- Stage B3b2 + C: finish Q, compute power, exp, and alpha.
            // This is the longest single Dst block in the kernel — it folds
            // the last add into Q, then runs the entire exp / clamp / opacity
            // chain to produce the per-pixel alpha. Doing it as one block
            // avoids extra spill/reload to L1 between sub-stages.
            //
            //   alpha = min( opacity · exp(min(-0.5·Q, 0)),  0.99 )
            //
            tile_regs_acquire();
            add_tiles_init(CB_POWER, CB_Q);
            add_tiles(CB_POWER, CB_Q, 0, 2, 0);  // dst[0] = (a·dx² + c·dy²) + 2b·dx·dy = Q

            // power = -0.5 · Q
            mul_unary_tile(0, NEG_HALF_BITS);

            // Clamp power to ≤ 0. For valid PSD covariance Q ≥ 0 always, so
            // this is mostly defensive (handles fp rounding edge cases).
            copy_tile_to_dst_init_short(CB_CONST_ZERO);
            copy_tile(CB_CONST_ZERO, 0, 1);
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            // weight = exp(power). Approximate-mode polynomial (~100 cycles
            // vs ~800 for the accurate path) with built-in ClampToNegative
            // (input clamped to ≥ -88.5 before evaluating). Accuracy ~3-4
            // decimal digits — fine for alpha ∈ [0, 0.99]; PSNR drops ~11 dB
            // vs accurate but still has 16 dB headroom over the 35 dB floor.
            exp_tile_init<true>();
            exp_tile<true>(0);

            // dst[0] *= opacity
            mul_unary_tile(0, opacity_bits);

            // alpha = min(opacity · weight, 0.99). Cap at 0.99 (instead of
            // 1.0) so transmittance T can never reach exactly 0 — keeps
            // (1-alpha) > 0 in Stage E and avoids degenerate compositing.
            copy_tile_to_dst_init_short(CB_CONST_099);
            copy_tile(CB_CONST_099, 0, 1);
            binary_min_tile(0, 1, 0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();

            // Cleanup intermediates from B/C stages.
            cb_pop_front(CB_POWER, 1);
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1);
            cb_pop_front(CB_DY, 1);

            cb_wait_front(CB_ALPHA, 1);

            // ----- Stage D1: contrib = alpha · T_state (T pre-masked in Stage F).
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

            // ----- Stage D2: per-channel color accumulator update.
            //   color_c_state ← color_c_state + color_c · contrib
            // Done independently for R, G, B (3 channels × 2 acquire blocks
            // each = 6 acquire blocks total — repetitive but each is small).
            // The repeated pattern per channel is:
            //   1) load contrib into Dst, multiply by the channel's color
            //      scalar, pack into the scratch tile CB_T_TMP.
            //   2) add CB_T_TMP into the channel's persistent state CB,
            //      spilling the new sum back via pop+reserve+pack+push (the
            //      "in-place state replace" pattern for depth-1 state CBs).

            // R channel
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_r_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_R_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_R_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_R_STATE, 1);
            cb_reserve_back(CB_COLOR_R_STATE, 1);
            pack_tile(0, CB_COLOR_R_STATE);
            cb_push_back(CB_COLOR_R_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_R_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            // G channel
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_g_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_G_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_G_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_G_STATE, 1);
            cb_reserve_back(CB_COLOR_G_STATE, 1);
            pack_tile(0, CB_COLOR_G_STATE);
            cb_push_back(CB_COLOR_G_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_G_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            // B channel
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_b_bits);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_T_TMP, 1);
            pack_tile(0, CB_T_TMP);
            cb_push_back(CB_T_TMP, 1);
            tile_regs_release();
            cb_wait_front(CB_T_TMP, 1);

            tile_regs_acquire();
            add_tiles_init(CB_COLOR_B_STATE, CB_T_TMP);
            add_tiles(CB_COLOR_B_STATE, CB_T_TMP, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_B_STATE, 1);
            cb_reserve_back(CB_COLOR_B_STATE, 1);
            pack_tile(0, CB_COLOR_B_STATE);
            cb_push_back(CB_COLOR_B_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_B_STATE, 1);
            cb_pop_front(CB_T_TMP, 1);

            cb_pop_front(CB_CONTRIB, 1);

            // ----- Stage E: front-to-back transmittance update.
            //   T_state ← T_state · (1 - alpha)
            //
            // Step 1: one_minus_alpha ← rsub(alpha, 1.0) = 1.0 - alpha
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_ALPHA);
            copy_tile(CB_ALPHA, 0, 0);
            rsub_unary_tile(0, ONE_F_BITS);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ONE_MINUS_ALPHA, 1);
            pack_tile(0, CB_ONE_MINUS_ALPHA);
            cb_push_back(CB_ONE_MINUS_ALPHA, 1);
            tile_regs_release();
            cb_wait_front(CB_ONE_MINUS_ALPHA, 1);

            // Step 2: T_state ← T_state · (1 - alpha)  (spill)
            tile_regs_acquire();
            mul_tiles_init(CB_T_STATE, CB_ONE_MINUS_ALPHA);
            mul_tiles(CB_T_STATE, CB_ONE_MINUS_ALPHA, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_T_STATE, 1);
            cb_reserve_back(CB_T_STATE, 1);
            pack_tile(0, CB_T_STATE);
            cb_push_back(CB_T_STATE, 1);
            tile_regs_release();
            cb_pop_front(CB_ONE_MINUS_ALPHA, 1);
            cb_wait_front(CB_T_STATE, 1);

            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // ----- Per-tile finalize: pack the running R/G/B accumulators to
        // CB_COLOR_OUT in a single 3-tile push. The writer kernel waits on
        // 3-at-a-time pushes here and DMAs them out to DRAM at the correct
        // global tile offset. After this, drain all the per-tile state
        // CBs so the next iteration starts with empty slots ready for re-init.
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

        // ===== Drain state CBs and per-tile inputs =====
        cb_pop_front(CB_COLOR_R_STATE, 1);
        cb_pop_front(CB_COLOR_G_STATE, 1);
        cb_pop_front(CB_COLOR_B_STATE, 1);
        cb_pop_front(CB_T_STATE, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
