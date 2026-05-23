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
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"

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
//     Stage F2: SUM-reduce sat mask; if sum==0 set block_saturated and skip
//               Stages A–E for remaining g's (still pop CB_SCALARS)
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
    // CB 24 = CB_READER_SCRATCH (reader-only, not visible to compute)
    constexpr uint32_t CB_CONST_ONE     = 25;  // constant 1.0 tile (bf16, reduce scaler)
    constexpr uint32_t CB_T_MAX         = 26;  // MAX-reduce output scratch (fp32, depth 1)
    constexpr uint32_t CB_BCAST_A       = 27;  // cov_a' row-bcast tile (bf16)
    constexpr uint32_t CB_BCAST_B       = 28;  // 2·cov_b' row-bcast tile (bf16)
    constexpr uint32_t CB_BCAST_C       = 29;  // cov_c' row-bcast tile (bf16)

    // Scratch CBs for Stage D2 producer batching (alias unused-since-iter-003 CBs).
    constexpr uint32_t CB_T_R = CB_DX2;
    constexpr uint32_t CB_T_G = CB_DY2;
    constexpr uint32_t CB_T_B = CB_DXDY;

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

    // CB_CONST_ONE ← tile of 1.0 (MAX-reduce scaler; ×1 leaves max unchanged)
    cb_reserve_back(CB_CONST_ONE, 1);
    tile_regs_acquire();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_ONE);
    tile_regs_release();
    cb_push_back(CB_CONST_ONE, 1);

    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);
    cb_wait_front(CB_CONST_ONE, 1);

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
            // Iter 038b: Stage F (per-pixel T saturation reset every 16 g's)
            // removed. With iter 035's cull eps=5e-2, contribution per
            // Gaussian is bounded; T decays smoothly to ~0 without ever
            // going negative or causing cumulative drift large enough to
            // visibly affect the output. PSNR check confirmed clean-keep
            // gate held after removal.

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

            // ----- Stage B1+B2 (iter 064): merged single acquire block.
            // Compute dx/dy via SFPU sub_unary, then fold dx·dy, dx², dy² all
            // in-Dst using SFPU mul_binary_tile (Dst→Dst, no CB round-trip for
            // dx/dy).  Saves one acquire/release pair vs the prior two-block
            // approach.
            //
            // Order: compute dx·dy BEFORE squaring so all three Dst values
            // fit in 3 slots (fp32 half-Dst = 4 slots; using 3 is safe):
            //   Dst[0] = dx, Dst[1] = dy
            //   → Dst[2] = dx·dy  (mul_binary_tile(0,1,2))
            //   → Dst[0] = dx²   (in-place: mul_binary_tile(0,0,0))
            //   → Dst[1] = dy²   (in-place: mul_binary_tile(1,1,1))
            // CB_DX2 ← pack(0), CB_DY2 ← pack(1), CB_DXDY ← pack(2).
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_PX);
            copy_tile(CB_PX, 0, 0);
            sub_unary_tile(0, mean_x_bits);   // Dst[0] = dx
            copy_tile_to_dst_init_short(CB_PY);
            copy_tile(CB_PY, 0, 1);
            sub_unary_tile(1, mean_y_bits);   // Dst[1] = dy
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 2);         // Dst[2] = dx·dy
            mul_binary_tile(0, 0, 0);         // Dst[0] = dx²  (in-place)
            mul_binary_tile(1, 1, 1);         // Dst[1] = dy²  (in-place)
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_DX2, 1);
            pack_tile(0, CB_DX2);
            cb_push_back(CB_DX2, 1);
            cb_reserve_back(CB_DY2, 1);
            pack_tile(1, CB_DY2);
            cb_push_back(CB_DY2, 1);
            cb_reserve_back(CB_DXDY, 1);
            pack_tile(2, CB_DXDY);
            cb_push_back(CB_DXDY, 1);
            tile_regs_release();
            cb_wait_front(CB_DX2, 1);
            cb_wait_front(CB_DY2, 1);
            cb_wait_front(CB_DXDY, 1);
            cb_wait_front(CB_BCAST_A, 1);
            cb_wait_front(CB_BCAST_B, 1);
            cb_wait_front(CB_BCAST_C, 1);

            // ----- Stage B2+B3+C (Iter 061 / re-applied iter 049d): host
            // pre-folds -0.5 into cov scalars; reader pushes bf16 row-bcast
            // tiles so FPU mul_tiles_bcast_rows replaces 3× SFPU mul_unary.
            //
            //   power = (-c/2det)·dx² + (b/det)·dx·dy + (-a/2det)·dy² = -Q/2
            //   alpha = opacity · exp(power)
            tile_regs_acquire();
            mul_bcast_rows_init_short(CB_DX2, CB_BCAST_A);
            mul_tiles_bcast_rows(CB_DX2, CB_BCAST_A, 0, 0, 0);
            mul_bcast_rows_init_short(CB_DY2, CB_BCAST_C);
            mul_tiles_bcast_rows(CB_DY2, CB_BCAST_C, 0, 0, 1);
            mul_bcast_rows_init_short(CB_DXDY, CB_BCAST_B);
            mul_tiles_bcast_rows(CB_DXDY, CB_BCAST_B, 0, 0, 2);

            // dst[0] = power = sum of the three pre-halved signed products.
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);

            // weight = exp(power)
            exp_tile_init<true>();
            exp_tile<true>(0);

            // alpha = opacity · weight. Iter 041: opacity is host-clamped at
            // 0.99, so alpha ≤ 0.99 always. No per-pixel min needed.
            mul_unary_tile(0, opacity_bits);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_ALPHA, 1);
            pack_tile(0, CB_ALPHA);
            cb_push_back(CB_ALPHA, 1);
            tile_regs_release();

            cb_pop_front(CB_DX2, 1);
            cb_pop_front(CB_DY2, 1);
            cb_pop_front(CB_DXDY, 1);
            cb_pop_front(CB_BCAST_A, 1);
            cb_pop_front(CB_BCAST_B, 1);
            cb_pop_front(CB_BCAST_C, 1);

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
            // One batched producer (contrib·color_r/g/b → CB_T_R/G/B) plus one
            // batched FPU adder block (state += scratch), in-place spill to 3 CBs.

            // Batched producer: dst[0..2] = contrib · color_r/g/b
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

            // Batched adder + Stage E: dst[0..2] = R/G/B_state + T_R/G/B,
            // dst[3] = T_state - CB_CONTRIB (front-to-back transmittance update)
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
