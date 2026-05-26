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
#include "api/compute/eltwise_binary_sfpu.h"

// Alpha-blend compute kernel: 3D Gaussian Splatting forward rasterizer
// (front-to-back compositing) for a per-core slice of screen tiles.
//
// M1 CHANGES (iter-003)
// ----------------------
// Q(x,y) is now assembled in basis form:
//   Q(x,y) = A*x^2 + B*x*y + C*y^2 + D*x + E*y + F
// where (A,B,C,D,E,F) are pre-computed host-side per-Gaussian and sent in
// CB_SCALARS pack[0..5]. x,y are tile-local pixel coords in [0,32).
// Six precomputed basis tiles (x², xy, y², x, y, 1) are generated once at
// kernel startup from fill_tile ops and held in CB_BASIS_X2..CB_BASIS_ONE.
//
// This replaces the per-Gaussian dx/dy/dx²/dy²/dxdy chain (stages B1/B2/B3)
// with a 4-acquire FPU-copy + SFPU-scale + add tree:
//   Block 1 (6 mul): copy each basis tile to DST, scale by coefficient, pack → CB_Q (depth 6)
//   Block 2 (3 add): sum 3 pairs from CB_Q → pack 3 → CB_POWER (depth 3)
//   Block 3 (1 add): sum first 2 CB_POWER entries → CB_T_TMP
//   Block 4 (Q+alpha chain): add CB_T_TMP + CB_POWER[2] = Q, then -0.5*Q, clamp, exp, opacity, min 0.99 → alpha
//
// HIGH-LEVEL FLOW (unchanged from iter-0 except Q computation)
// ----------------
// For each screen tile:
//   Init      : R = G = B = 0,  T = 1,  sat_mask = 1   (per pixel)
//   For each Gaussian g in this tile, sorted front-to-back:
//     Stage F : every 16 g's (skip g=0): sat_mask = (T >= 1e-4)
//     Stage A : read 10 fp32 scalars (A,B,C,D,E,F, R,G,B, opacity)
//     Stage B : build Q in basis form (4 acquire blocks)
//     Stage C : alpha = min(opacity * exp(min(-0.5*Q, 0)), 0.99)
//     Stage D1: contrib = alpha * T * sat_mask
//     Stage D2: for c in {R,G,B}: c_state += color_c * contrib
//     Stage E : T <- T * (1-alpha) * sat_mask
//   Output    : pack R_state, G_state, B_state to CB_COLOR_OUT.
//
// RUNTIME ARGS
//   0: num_tiles  -- number of screen tiles this core processes
//
// CB INDICES — see alpha_blend_host.h for the canonical declaration.
void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // CB indices — must match alpha_blend_host.h verbatim.
    constexpr uint32_t CB_PX            = 0;
    constexpr uint32_t CB_PY            = 1;   // unused in M1 but kept for symmetry
    constexpr uint32_t CB_SCALARS       = 2;   // 10 fp32 per Gaussian (64-byte page)
    constexpr uint32_t CB_TILE_META     = 3;
    // CB 4..8 formerly dx/dy/dx²/dy²/dxdy — unused in M1, allocated but not consumed.
    constexpr uint32_t CB_Q             = 9;   // Q term accumulator (6 tiles during Q build)
    constexpr uint32_t CB_POWER         = 10;  // 3 pair-sums during Q build; also power(-0.5*Q)
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
    // M1 basis tiles
    constexpr uint32_t CB_BASIS_X2      = 24;  // x² = (j+0.5)²
    constexpr uint32_t CB_BASIS_XY      = 25;  // x*y = (j+0.5)*(i+0.5)
    constexpr uint32_t CB_BASIS_Y2      = 26;  // y² = (i+0.5)²
    constexpr uint32_t CB_BASIS_X       = 27;  // x  = j+0.5
    constexpr uint32_t CB_BASIS_Y       = 28;  // y  = i+0.5
    constexpr uint32_t CB_BASIS_ONE     = 29;  // 1.0

    // Bit-pattern fp32 constants for SFPU scalar ops.
    constexpr uint32_t NEG_HALF_BITS  = 0xBF000000u;  // fp32(-0.5)
    constexpr uint32_t ONE_F_BITS     = 0x3F800000u;  // fp32( 1.0)
    constexpr uint32_t T_THRESH_BITS  = 0x38D1B717u;  // fp32(1e-4)

    // Foundational SFPU/FPU init.
    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    // ----- Pre-fill constant and basis tiles (once per kernel invocation) -----
    // All tiles pushed here are never popped; the kernel copies from them when needed.

    fill_tile_init();

    // CB_CONST_ZERO <- 0.0
    cb_reserve_back(CB_CONST_ZERO, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_ZERO);
    tile_regs_release();
    cb_push_back(CB_CONST_ZERO, 1);

    // CB_CONST_099 <- 0.99
    cb_reserve_back(CB_CONST_099, 1);
    tile_regs_acquire();
    fill_tile(0, 0.99f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_CONST_099);
    tile_regs_release();
    cb_push_back(CB_CONST_099, 1);

    // ----- Basis tiles for M1 Q assembly -----
    // Tile-local coords: x = j + 0.5 for column j in [0,31],
    //                    y = i + 0.5 for row    i in [0,31].
    // The fill_tile kernel op can only fill a constant value; we need varying
    // per-element values. Use fill_tile(0.5f) as a base and then build x/y
    // via SFPU scalar ops.
    //
    // Strategy: fill x_tile with 0.5, then add column index j.
    // Since we only have fill_tile (constant), we build x and y from
    // CB_PX and CB_PY which are tile-local pixel coord tiles already
    // available for the first tile (pushed by reader per tile).
    // BUT: basis tiles must be INDEPENDENT of the tile being processed —
    // they are the SAME for every screen tile (tile-local coords [0,31]).
    //
    // The host code in alpha_blend.cpp sends CB_PX and CB_PY as tile-local
    // coords (verified: _get_px_py_grids returns px[i,j]=j+0.5, py[i,j]=i+0.5
    // for EVERY tile due to the tile-local convention). So the FIRST tile's
    // CB_PX/PY values serve as the basis tiles for ALL tiles.
    //
    // We use a two-pass approach:
    //   1. Wait for the first PX/PY tiles (reader pushes them per tile).
    //   2. Compute the 6 basis tiles from PX/PY using mul_tiles and pack to their CBs.
    //   3. Pop PX/PY (the outer tile loop will re-push them for the actual first tile).
    //
    // NOTE: This requires the reader to have pushed at least 1 PX/PY pair.
    // We handle this by waiting for CB_TILE_META first (which gives us g_count
    // for the first tile) so the reader is ahead. But actually we should NOT
    // pre-consume the first tile's CB_TILE_META/CB_PX/CB_PY here because the
    // outer loop expects them.
    //
    // SIMPLER ALTERNATIVE: Generate basis tiles purely from fill_tile by
    // computing x = col + 0.5 and y = row + 0.5 using 32 separate fill ops
    // or by using the existing CB_CONST_ZERO as a base and SFPU ops on it.
    //
    // ACTUAL APPROACH: Use a loop over 32 rows/cols to fill the tile.
    // In tt-metal kernel code, we can use fill_tile with a constant and
    // use sfpu sub_unary_tile/add_unary_tile on specific rows... but
    // tile operations always apply to all 32x32 elements, not individual rows.
    //
    // FINAL DECISION: Piggyback on CB_PX/CB_PY. The reader always pushes
    // PX/PY for every screen tile in the same pixel-local coord system
    // (px[i,j] = j+0.5, py[i,j] = i+0.5). We capture these from the FIRST
    // tile's PX/PY push, compute the 6 basis tiles from them, and pack to
    // the 6 permanent basis CBs. Then we pop PX/PY to allow the outer loop
    // to see them again.
    //
    // Since the outer tile loop also calls cb_wait_front(CB_PX/CB_PY), and
    // we consume one pair here BEFORE the loop, the reader must push at
    // least 1 pair before we start. This is guaranteed because the reader
    // always pushes TILE_META + PX + PY BEFORE the Gaussian scalars for
    // each tile, so by the time kernel_main reaches here, the reader has
    // at least started pushing the first tile's data.
    //
    // To avoid a deadlock with CB_TILE_META (depth 2), we use CB_PX/CB_PY
    // (depth 2 each) to capture basis tiles before entering the main loop.
    // We then pop CB_PX/CB_PY to make the slot available for the main loop.

    // Wait for first tile's PX/PY to arrive (reader pushes them with the first tile).
    // These are tile-local pixel coordinates:
    //   px[i,j] = j + 0.5  (x coordinate in [0.5, 31.5])
    //   py[i,j] = i + 0.5  (y coordinate in [0.5, 31.5])
    // Since the kernel uses tile-local coordinates (same for every tile),
    // the first tile's PX/PY serve as the template for all 6 basis tiles.
    // IMPORTANT: Do NOT pop CB_PX/CB_PY here — the outer tile loop will
    // consume them normally for tile 0's Gaussian processing.
    cb_wait_front(CB_PX, 1);
    cb_wait_front(CB_PY, 1);

    // CB_BASIS_X2 <- x^2 = px * px
    cb_reserve_back(CB_BASIS_X2, 1);
    tile_regs_acquire();
    mul_tiles_init(CB_PX, CB_PX);
    mul_tiles(CB_PX, CB_PX, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_X2);
    tile_regs_release();
    cb_push_back(CB_BASIS_X2, 1);

    // CB_BASIS_XY <- x*y = px * py
    cb_reserve_back(CB_BASIS_XY, 1);
    tile_regs_acquire();
    mul_tiles_init(CB_PX, CB_PY);
    mul_tiles(CB_PX, CB_PY, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_XY);
    tile_regs_release();
    cb_push_back(CB_BASIS_XY, 1);

    // CB_BASIS_Y2 <- y^2 = py * py
    cb_reserve_back(CB_BASIS_Y2, 1);
    tile_regs_acquire();
    mul_tiles_init(CB_PY, CB_PY);
    mul_tiles(CB_PY, CB_PY, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_Y2);
    tile_regs_release();
    cb_push_back(CB_BASIS_Y2, 1);

    // CB_BASIS_X <- x = px
    cb_reserve_back(CB_BASIS_X, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_PX);
    copy_tile(CB_PX, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_X);
    tile_regs_release();
    cb_push_back(CB_BASIS_X, 1);

    // CB_BASIS_Y <- y = py
    cb_reserve_back(CB_BASIS_Y, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_PY);
    copy_tile(CB_PY, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_Y);
    tile_regs_release();
    cb_push_back(CB_BASIS_Y, 1);

    // CB_BASIS_ONE <- 1.0
    cb_reserve_back(CB_BASIS_ONE, 1);
    tile_regs_acquire();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, CB_BASIS_ONE);
    tile_regs_release();
    cb_push_back(CB_BASIS_ONE, 1);

    // Now all constant and basis CBs are ready. Wait on them.
    cb_wait_front(CB_CONST_ZERO, 1);
    cb_wait_front(CB_CONST_099, 1);
    cb_wait_front(CB_BASIS_X2, 1);
    cb_wait_front(CB_BASIS_XY, 1);
    cb_wait_front(CB_BASIS_Y2, 1);
    cb_wait_front(CB_BASIS_X,  1);
    cb_wait_front(CB_BASIS_Y,  1);
    cb_wait_front(CB_BASIS_ONE, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        // =====================================================================
        // Per-tile state CB init.
        // =====================================================================

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

        cb_reserve_back(CB_SAT_MASK, 1);
        tile_regs_acquire();
        fill_tile(0, 1.0f);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SAT_MASK);
        tile_regs_release();
        cb_push_back(CB_SAT_MASK, 1);

        cb_wait_front(CB_COLOR_R_STATE, 1);
        cb_wait_front(CB_COLOR_G_STATE, 1);
        cb_wait_front(CB_COLOR_B_STATE, 1);
        cb_wait_front(CB_T_STATE, 1);
        cb_wait_front(CB_SAT_MASK, 1);

        cb_wait_front(CB_TILE_META, 1);
        uint32_t g_count = ckernel::read_tile_value(CB_TILE_META, /*tile_index=*/0, /*element_offset=*/0);
        cb_pop_front(CB_TILE_META, 1);

        cb_wait_front(CB_PX, 1);
        cb_wait_front(CB_PY, 1);

        // =====================================================================
        // Per-Gaussian inner loop.
        // =====================================================================
        for (uint32_t g = 0; g < g_count; g++) {
            // ----- Stage F: sat_mask refresh (every 16 Gaussians, skip g=0) -----
            if (g > 0 && (g & 0xFu) == 0u) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(CB_T_STATE);
                copy_tile(CB_T_STATE, 0, 0);
                unary_ge_tile_init();
                unary_ge_tile(0, T_THRESH_BITS);
                tile_regs_commit();
                tile_regs_wait();
                cb_pop_front(CB_SAT_MASK, 1);
                cb_reserve_back(CB_SAT_MASK, 1);
                pack_tile(0, CB_SAT_MASK);
                cb_push_back(CB_SAT_MASK, 1);
                tile_regs_release();
                cb_wait_front(CB_SAT_MASK, 1);
            }

            cb_wait_front(CB_SCALARS, 1);

            // ----- Stage A: read 10 fp32 attributes from CB_SCALARS.
            // M1 layout: [A, B, C, D, E, F, R, G, B_col, opacity]
            // where Q(x,y) = A*x^2 + B*xy + C*y^2 + D*x + E*y + F
            uint32_t coeff_A_bits = ckernel::read_tile_value(CB_SCALARS, 0, 0);
            uint32_t coeff_B_bits = ckernel::read_tile_value(CB_SCALARS, 0, 1);
            uint32_t coeff_C_bits = ckernel::read_tile_value(CB_SCALARS, 0, 2);
            uint32_t coeff_D_bits = ckernel::read_tile_value(CB_SCALARS, 0, 3);
            uint32_t coeff_E_bits = ckernel::read_tile_value(CB_SCALARS, 0, 4);
            uint32_t coeff_F_bits = ckernel::read_tile_value(CB_SCALARS, 0, 5);
            uint32_t color_r_bits = ckernel::read_tile_value(CB_SCALARS, 0, 6);
            uint32_t color_g_bits = ckernel::read_tile_value(CB_SCALARS, 0, 7);
            uint32_t color_b_bits = ckernel::read_tile_value(CB_SCALARS, 0, 8);
            uint32_t opacity_bits = ckernel::read_tile_value(CB_SCALARS, 0, 9);

            // ----- Stage B+C: Build Q in basis form + exp + alpha clamp -----
            //
            // SINGLE acquire block. Q stays in fp32 DST throughout; no
            // intermediate packs to bf16. This is critical because the 6
            // basis terms (A·x², B·xy, C·y², D·x, E·y, F) have magnitudes
            // ~O(qxx · 1000) ≈ O(10) that cancel to a small Q (~0.01 near
            // the Gaussian center). Packing each term to bf16 would lose
            // ~0.04 per term; 6 such errors compound to swamp the true Q
            // signal. Keep everything in the fp32 DST register until exp().
            tile_regs_acquire();

            // dst[0] = A·x²
            copy_tile_to_dst_init_short(CB_BASIS_X2);
            copy_tile(CB_BASIS_X2, 0, 0);
            mul_unary_tile(0, coeff_A_bits);

            // dst[1] = B·xy; dst[0] += dst[1]
            copy_tile_to_dst_init_short(CB_BASIS_XY);
            copy_tile(CB_BASIS_XY, 0, 1);
            mul_unary_tile(1, coeff_B_bits);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);

            // dst[1] = C·y²; dst[0] += dst[1]
            copy_tile_to_dst_init_short(CB_BASIS_Y2);
            copy_tile(CB_BASIS_Y2, 0, 1);
            mul_unary_tile(1, coeff_C_bits);
            add_binary_tile(0, 1, 0);

            // dst[1] = D·x; dst[0] += dst[1]
            copy_tile_to_dst_init_short(CB_BASIS_X);
            copy_tile(CB_BASIS_X, 0, 1);
            mul_unary_tile(1, coeff_D_bits);
            add_binary_tile(0, 1, 0);

            // dst[1] = E·y; dst[0] += dst[1]
            copy_tile_to_dst_init_short(CB_BASIS_Y);
            copy_tile(CB_BASIS_Y, 0, 1);
            mul_unary_tile(1, coeff_E_bits);
            add_binary_tile(0, 1, 0);

            // dst[0] += F  (scalar add, no tile load)
            add_unary_tile(0, coeff_F_bits);
            // dst[0] is now Q.

            // power = -0.5 * Q
            mul_unary_tile(0, NEG_HALF_BITS);

            // Clamp power to <= 0 (defensive: Q >= 0 for valid PSD covariance).
            copy_tile_to_dst_init_short(CB_CONST_ZERO);
            copy_tile(CB_CONST_ZERO, 0, 1);
            binary_min_tile_init();
            binary_min_tile(0, 1, 0);

            // weight = exp(power). Approximate mode.
            exp_tile_init<true>();
            exp_tile<true>(0);

            // alpha = min(opacity * weight, 0.99)
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

            cb_wait_front(CB_ALPHA, 1);

            // ----- Stage D1: contrib = alpha * T_state * sat_mask (FUSED, was 2 acquires) -----
            // FPU mul_tiles(alpha, T_state) -> dst[0]
            // SFPU copy(sat_mask) -> dst[1]; mul_binary_tile dst[0]*=dst[1]
            tile_regs_acquire();
            mul_tiles_init(CB_ALPHA, CB_T_STATE);
            mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0);
            copy_tile_to_dst_init_short(CB_SAT_MASK);
            copy_tile(CB_SAT_MASK, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_CONTRIB, 1);
            pack_tile(0, CB_CONTRIB);
            cb_push_back(CB_CONTRIB, 1);
            tile_regs_release();
            cb_wait_front(CB_CONTRIB, 1);

            // ----- Stage D2: color accumulator update (FUSED, each channel was 2 acquires) -----
            // R channel: dst[0] = R_state, dst[1] = contrib*color_r, dst[0] += dst[1]
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_COLOR_R_STATE);
            copy_tile(CB_COLOR_R_STATE, 0, 0);
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 1);
            mul_unary_tile(1, color_r_bits);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_R_STATE, 1);
            cb_reserve_back(CB_COLOR_R_STATE, 1);
            pack_tile(0, CB_COLOR_R_STATE);
            cb_push_back(CB_COLOR_R_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_R_STATE, 1);

            // G channel
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_COLOR_G_STATE);
            copy_tile(CB_COLOR_G_STATE, 0, 0);
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 1);
            mul_unary_tile(1, color_g_bits);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_G_STATE, 1);
            cb_reserve_back(CB_COLOR_G_STATE, 1);
            pack_tile(0, CB_COLOR_G_STATE);
            cb_push_back(CB_COLOR_G_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_G_STATE, 1);

            // B channel
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_COLOR_B_STATE);
            copy_tile(CB_COLOR_B_STATE, 0, 0);
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 1);
            mul_unary_tile(1, color_b_bits);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_COLOR_B_STATE, 1);
            cb_reserve_back(CB_COLOR_B_STATE, 1);
            pack_tile(0, CB_COLOR_B_STATE);
            cb_push_back(CB_COLOR_B_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_COLOR_B_STATE, 1);

            cb_pop_front(CB_CONTRIB, 1);

            // ----- Stage E: T_state = T_state * (1 - alpha) * sat_mask (FUSED, was 3 acquires) -----
            // dst[0] = T_state, dst[1] = 1-alpha, mul; dst[1] = sat_mask, mul
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_T_STATE);
            copy_tile(CB_T_STATE, 0, 0);
            copy_tile_to_dst_init_short(CB_ALPHA);
            copy_tile(CB_ALPHA, 0, 1);
            rsub_unary_tile(1, ONE_F_BITS);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_tile_to_dst_init_short(CB_SAT_MASK);
            copy_tile(CB_SAT_MASK, 0, 1);
            mul_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_T_STATE, 1);
            cb_reserve_back(CB_T_STATE, 1);
            pack_tile(0, CB_T_STATE);
            cb_push_back(CB_T_STATE, 1);
            tile_regs_release();
            cb_wait_front(CB_T_STATE, 1);

            cb_pop_front(CB_ALPHA, 1);
            cb_pop_front(CB_SCALARS, 1);
        }

        // ----- Per-tile finalize -----
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

        // Drain state CBs and per-tile inputs
        cb_pop_front(CB_COLOR_R_STATE, 1);
        cb_pop_front(CB_COLOR_G_STATE, 1);
        cb_pop_front(CB_COLOR_B_STATE, 1);
        cb_pop_front(CB_T_STATE, 1);
        cb_pop_front(CB_SAT_MASK, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
