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

#include "tools/profiler/kernel_profiler.hpp"

// Alpha-blend compute kernel: 3D Gaussian Splatting forward rasterizer
// (front-to-back compositing) for a per-core slice of screen tiles.
//
// HIGH-LEVEL FLOW
// ----------------
// For each screen tile this core owns, with running per-pixel accumulator
// state (R/G/B color, transmittance T, saturation sentinel mask):
//
//   Init      : R = G = B = 0,  T = 1,  sat_mask = 1   (per pixel)
//   For each Gaussian g in this tile, sorted front-to-back:
//     Stage F : every 16 g's (skip g=0): sat_mask = (T >= 1e-4)
//               -> "freeze" pixels whose contribution would round to <1/255
//     Stage A : read 9 fp32 scalars (mean, cov_inv, color, opacity)
//     Stage B1: dx = px - mean_x,  dy = py - mean_y         (per-pixel offset)
//     Stage B2: dx² , dy² , dx·dy                           (squared offsets)
//     Stage B3: Q = a·dx² + 2b·dx·dy + c·dy²                (Mahalanobis dist)
//               power = -0.5·Q                              (Gaussian exponent)
//     Stage C : weight  = exp(min(power, 0))                (Gaussian falloff)
//               alpha   = min(opacity · weight, 0.99)       (per-pixel opacity)
//     Stage D1: contrib = alpha · T · sat_mask              (effective contribution)
//     Stage D2: for c in {R,G,B}: c_state += color_c · contrib
//     Stage E : T ← T · (1 - alpha) · sat_mask              (transmittance update)
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
    constexpr uint32_t CB_CONTRIB       = 13;  // contrib = alpha · T · sat_mask
    constexpr uint32_t CB_ONE_MINUS_ALPHA = 14;
    constexpr uint32_t CB_T_TMP         = 15;  // generic intermediate
    constexpr uint32_t CB_COLOR_OUT     = 16;  // R, G, B output tiles per screen tile
    constexpr uint32_t CB_COLOR_R_STATE = 17;  // running R accumulator
    constexpr uint32_t CB_COLOR_G_STATE = 18;
    constexpr uint32_t CB_COLOR_B_STATE = 19;
    constexpr uint32_t CB_T_STATE       = 20;  // running transmittance per pixel
    constexpr uint32_t CB_SAT_MASK      = 21;  // 1.0 if T>=1e-4 else 0.0
    constexpr uint32_t CB_CONST_ZERO    = 22;  // constant 0.0 tile
    constexpr uint32_t CB_CONST_099     = 23;  // constant 0.99 tile

    // Bit-pattern fp32 constants for SFPU scalar-unary ops (mul_unary_tile,
    // sub_unary_tile, rsub_unary_tile, etc.) which take their immediate as a
    // uint32 bit-cast of the float they want.
    constexpr uint32_t NEG_HALF_BITS  = 0xBF000000u;  // fp32(-0.5)
    constexpr uint32_t ONE_F_BITS     = 0x3F800000u;  // fp32( 1.0)
    constexpr uint32_t T_THRESH_BITS  = 0x38D1B717u;  // fp32(1e-4) — T threshold for sat_mask

    // Foundational SFPU/FPU init: configures unpack and pack hardware for
    // binary tile ops on this core. Must come before any tile op.
    binary_op_init_common(CB_PX, CB_PY, CB_COLOR_OUT);

    // iter-054: CB_CONST_ZERO/CB_CONST_099 prologue dropped. The min(power,0)
    // clamp (iter-050) and min(alpha,0.99) cap (iter-051) were the only consumers;
    // both removed. Slots 22/23 reserved but no longer allocated/filled.

    for (uint32_t t = 0; t < num_tiles; t++) {
        // Per-tile profiling zone. In non-profile builds DeviceZoneScopedN is
        // a no-op macro (see tt-metal kernel_profiler.hpp), so this costs
        // nothing unless --profile is on.
        DeviceZoneScopedN("Z_C_tile");

        // =====================================================================
        // Per-tile state CB init: zero the color accumulators, set transmittance
        // to 1.0, and start with all pixels unsaturated. Each state CB lives
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

        // sat_mask = 1.0 (all pixels active)
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
            // Per-Gaussian accumulator zone (SumN1 → slot 0 in sums[]).
            // Aggregates cycles across all g iterations on this RISC so the
            // device-profile CSV reports a single SUM row per kernel — the
            // headline "compute spent N cycles in per-Gaussian work."
            DeviceZoneScopedSumN1("Z_C_g");

            // ----- Stage F: sat_mask refresh (every 16 Gaussians, skip g=0) -----
            // Recompute which pixels are still "active" (T >= 1e-4). For pixels
            // whose transmittance has saturated below 1e-4, sat_mask becomes 0
            // and zeroes their contribution in stages D1/E going forward —
            // effectively a per-pixel early termination without breaking the
            // SFPU's vector lock-step (we can't actually skip lanes, but multiplying
            // by 0 does the same job at the same op cost). g=0 is skipped because
            // T is freshly initialized to 1 above.
            if (g > 0 && (g & 0xFu) == 0u) {
                tile_regs_acquire();
                copy_tile_to_dst_init_short(CB_T_STATE);
                copy_tile(CB_T_STATE, 0, 0);
                unary_ge_tile_init();
                unary_ge_tile(0, T_THRESH_BITS);
                tile_regs_commit();
                tile_regs_wait();
                // Spill: replace existing sat_mask tile.
                cb_pop_front(CB_SAT_MASK, 1);
                cb_reserve_back(CB_SAT_MASK, 1);
                pack_tile(0, CB_SAT_MASK);
                cb_push_back(CB_SAT_MASK, 1);
                tile_regs_release();
                cb_wait_front(CB_SAT_MASK, 1);
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

            // ----- Stage B2+B3a (fused): three acquire blocks, each computing
            // one Q term directly:  mul_tiles → mul_unary_tile in same acquire.
            // Eliminates CB_DX2/CB_DY2/CB_DXDY scratch round-trips. Single
            // mul_tiles_init per block + SFPU mul_unary — safe (NOT the
            // iter-039 multi-init footgun: that hung from chaining ≥2
            // mul_tiles_init in one acquire; here it's one FPU init followed
            // by SFPU scalar mul, which is the standard pipeline).
            //   CB_Q[0] = a · dx²
            //   CB_Q[1] = c · dy²
            //   CB_Q[2] = 2b · dx·dy

            // FUSED (iter-049): All 3 B2 ops in ONE acquire using dst slots
            // 0/1/2. ONE mul_tiles_init shared by 3 mul_tiles calls with
            // different (but same-format bf16) CB pairs. Extends iter-048's
            // 2-way pattern. Third mul_tiles uses (CB_DX, CB_DY) — different
            // pair from init's (CB_DX, CB_DX), testing whether unpack
            // handles cross-CB pairing without re-init. Saves 2 acquires
            // per Gaussian vs the unfused 3-acquire pattern.
            //   dst[0] = a · dx²
            //   dst[1] = c · dy²
            //   dst[2] = 2b · dx·dy
            tile_regs_acquire();
            mul_tiles_init(CB_DX, CB_DX);
            mul_tiles(CB_DX, CB_DX, 0, 0, 0);
            mul_tiles(CB_DY, CB_DY, 0, 0, 1);
            mul_tiles(CB_DX, CB_DY, 0, 0, 2);
            mul_unary_tile(0, cov_a_bits);
            mul_unary_tile(1, cov_c_bits);
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

            // ----- Stage B3b + C + D1 (FUSED iter-052):
            // full Q sum → power → exp → alpha → contrib = α·T·sat, all in
            // ONE acquire. Eliminates the CB_ALPHA bf16 round-trip — alpha
            // never leaves fp32 dst between C and D1.
            //
            // Why this matters (per project-tile-structure-regression-iter-010):
            // CB_ALPHA was a bf16 CB that round-tripped α between C and D1.
            // The pack→bf16→unpack cycle quantized α uniformly per tile,
            // and that per-tile quantization correlates pixel error within
            // the tile — visible as tile-grid quilting in diff10 PNGs.
            // Keeping α in fp32 dst across the C→D1 boundary preserves the
            // higher-precision exp output through the T·sat multiplication.
            //
            // Same template as iter-042's E sub-identity-fuse: extend the
            // dst-resident computation past a bf16 CB boundary by chaining
            // binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA> directly on the
            // dst slot that holds α.
            //
            // Final identity in this acquire:
            //   contrib = (opacity · exp(-0.5·Q)) · T_state · sat_mask
            //
            tile_regs_acquire();
            add_tiles_init(CB_Q, CB_Q);
            add_tiles(CB_Q, CB_Q, 0, 1, 0);  // dst[0] = a·dx² + c·dy²
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_Q);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_Q, 2, 0);  // dst[0] = Q

            // power = -0.5 · Q (iter-050: dropped defensive min(power,0))
            mul_unary_tile(0, NEG_HALF_BITS);

            // weight = exp(power). Approximate-mode polynomial.
            exp_tile_init<true>();
            exp_tile<true>(0);

            // alpha = opacity · weight (iter-051: dropped 0.99 cap)
            mul_unary_tile(0, opacity_bits);

            // D1 fused in (iter-052): contrib = alpha · T_state · sat_mask
            // Skip CB_ALPHA bf16 round-trip — α stays in fp32 dst.
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_T_STATE);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_T_STATE, 0, 0);  // dst[0] = α·T
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SAT_MASK);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_SAT_MASK, 0, 0);  // dst[0] = contrib

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_CONTRIB, 1);
            pack_tile(0, CB_CONTRIB);
            cb_push_back(CB_CONTRIB, 1);
            tile_regs_release();

            // Cleanup B/C inputs (CB_ALPHA is no longer used).
            cb_pop_front(CB_Q, 3);
            cb_pop_front(CB_DX, 1);
            cb_pop_front(CB_DY, 1);

            cb_wait_front(CB_CONTRIB, 1);

            // ----- Stage D2: per-channel color accumulator update (FUSED iter-038).
            //   color_c_state ← color_c_state + color_c · contrib
            // Done independently for R, G, B (3 channels × 1 fused acquire each).
            //
            // FUSED (iter-038): per channel one acquire
            //   dst[0] = contrib · color_c
            //   binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(state_c) → dst[0] += state_c
            //   pack back to state_c (with pop_front → reserve_back → push_back)
            // Saves 1 acquire + the CB_T_TMP roundtrip per channel — same pattern as
            // iter-007 Stage D1 and iter-010 Stage E.

            // D2+E mega-fuse (iter-045): merge D2's R/G/B 3-channel update
            // with Stage E's T_new = T·sat - contrib into ONE acquire using
            // dst slots 0/1/2/3. Algebra unchanged from iter-042+043:
            //   slot 0: R_state += contrib · color_r
            //   slot 1: G_state += contrib · color_g
            //   slot 2: B_state += contrib · color_b
            //   slot 3: T_new = T·sat - contrib       (= T·(1-α)·sat)
            // The acquire has ONE mul_tiles_init (for E's T·sat) following
            // a sequence of binary_dest_reuse_tiles_init calls (for D2's
            // ELWADD per channel). iter-039 hang was MULTIPLE mul_tiles_init;
            // here it's a single mul_tiles_init mid-acquire — testing this
            // novel pattern. Saves 1 acquire per Gaussian.
            tile_regs_acquire();

            // D2-R: dst[0] = contrib · color_r + R_state
            // iter-055: single copy_tile_to_dst_init_short(CB_CONTRIB) covers
            // all 3 channels — same CB, same format, init configures format
            // not CB binding (same lesson as iter-048+049 mul_tiles_init).
            copy_tile_to_dst_init_short(CB_CONTRIB);
            copy_tile(CB_CONTRIB, 0, 0);
            mul_unary_tile(0, color_r_bits);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_R_STATE);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_R_STATE, 0, 0);

            // D2-G: dst[1] = contrib · color_g + G_state
            copy_tile(CB_CONTRIB, 0, 1);
            mul_unary_tile(1, color_g_bits);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_G_STATE);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_G_STATE, 0, 1);

            // D2-B: dst[2] = contrib · color_b + B_state
            copy_tile(CB_CONTRIB, 0, 2);
            mul_unary_tile(2, color_b_bits);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_B_STATE);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_COLOR_B_STATE, 0, 2);

            // E: dst[3] = T·sat - contrib = T_new
            mul_tiles_init(CB_T_STATE, CB_SAT_MASK);
            mul_tiles(CB_T_STATE, CB_SAT_MASK, 0, 0, 3);  // dst[3] = T·sat
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONTRIB);
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWSUB, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(CB_CONTRIB, 0, 3);  // dst[3] -= contrib

            tile_regs_commit();
            tile_regs_wait();

            cb_pop_front(CB_COLOR_R_STATE, 1);
            cb_pop_front(CB_COLOR_G_STATE, 1);
            cb_pop_front(CB_COLOR_B_STATE, 1);
            cb_pop_front(CB_T_STATE, 1);
            cb_reserve_back(CB_COLOR_R_STATE, 1);
            cb_reserve_back(CB_COLOR_G_STATE, 1);
            cb_reserve_back(CB_COLOR_B_STATE, 1);
            cb_reserve_back(CB_T_STATE, 1);
            pack_tile(0, CB_COLOR_R_STATE);
            pack_tile(1, CB_COLOR_G_STATE);
            pack_tile(2, CB_COLOR_B_STATE);
            pack_tile(3, CB_T_STATE);
            cb_push_back(CB_COLOR_R_STATE, 1);
            cb_push_back(CB_COLOR_G_STATE, 1);
            cb_push_back(CB_COLOR_B_STATE, 1);
            cb_push_back(CB_T_STATE, 1);
            tile_regs_release();

            cb_wait_front(CB_COLOR_R_STATE, 1);
            cb_wait_front(CB_COLOR_G_STATE, 1);
            cb_wait_front(CB_COLOR_B_STATE, 1);
            cb_wait_front(CB_T_STATE, 1);

            cb_pop_front(CB_CONTRIB, 1);
            // CB_ALPHA no longer used (iter-052 fused B3b/C+D1).
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
        cb_pop_front(CB_SAT_MASK, 1);
        cb_pop_front(CB_PX, 1);
        cb_pop_front(CB_PY, 1);
    }
}
