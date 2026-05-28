// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc (project_full_with_cov3d) device compute kernel — amendment-002 tt-008c.
//
// This is the "full pfwc on device" revision: the kernel now also computes
// cov2d and radii on-device. cov_cam is computed but kept purely as scratch.
// Outputs are 8 SoA streams the downstream device tile_assign (tt-006) reads
// directly via NoC:
//   * mean_2d.x, mean_2d.y        (perspective)
//   * depth (= tz)
//   * cov2d_a, cov2d_b, cov2d_c   (J · cov_cam · J^T + 0.3·I, 3 unique)
//   * radii.x, radii.y            (ceil(k · sqrt(a)) / ceil(k · sqrt(c)),
//                                  k = 3.0 — matches the project_full_fused
//                                  k_cap=3.0 default; PSNR-safe)
//
// PRECISION: same playbook as tt-007. UnpackToDestFp32 on every input/scratch
// CB, all arithmetic via SFPU (mul_unary, add_unary, mul_binary, add_binary)
// on FP32 DEST slots. recip/sqrt/ceil all run through SFPU.
//
// RUNTIME ARGS (uint32_t, layout matches pfwc_device.cpp):
//   0           : num_chunks
//   1..9        : R bits (unused here — host pre-baked into cov_cam scales)
//   10..12      : t0, t1, t2 (translation)
//   13..16      : fx, fy, cx, cy
//   17..52      : 36 cov_cam scales (6 entries × 6 scales)
//   53          : k_bits (fp32 of k = 3.0 default)
//   54          : neg_fx_bits (fp32 of -fx)
//   55          : neg_fy_bits (fp32 of -fy)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/rounding.h"

namespace {

// CB layout — must match pfwc_device.cpp.
constexpr uint32_t CB_MCX = 0;
constexpr uint32_t CB_MCY = 1;
constexpr uint32_t CB_MCZ = 2;
constexpr uint32_t CB_C00 = 3;
constexpr uint32_t CB_C01 = 4;
constexpr uint32_t CB_C02 = 5;
constexpr uint32_t CB_C11 = 6;
constexpr uint32_t CB_C12 = 7;
constexpr uint32_t CB_C22 = 8;

// Outputs — 8 SoA streams.
constexpr uint32_t CB_M2X = 9;
constexpr uint32_t CB_M2Y = 10;
constexpr uint32_t CB_DEP = 11;
constexpr uint32_t CB_A   = 12;
constexpr uint32_t CB_B   = 13;
constexpr uint32_t CB_C   = 14;
constexpr uint32_t CB_RX  = 15;
constexpr uint32_t CB_RY  = 16;

// Scratch (per-chunk intermediates).
constexpr uint32_t CB_TMP_TX     = 17;
constexpr uint32_t CB_TMP_TY     = 18;
constexpr uint32_t CB_TMP_TZ     = 19;
constexpr uint32_t CB_TMP_INV_TZ = 20;
constexpr uint32_t CB_TMP_CC00   = 21;
constexpr uint32_t CB_TMP_CC01   = 22;
constexpr uint32_t CB_TMP_CC02   = 23;
constexpr uint32_t CB_TMP_CC11   = 24;
constexpr uint32_t CB_TMP_CC12   = 25;
constexpr uint32_t CB_TMP_CC22   = 26;

constexpr uint32_t COV3D_CB[6] = {CB_C00, CB_C11, CB_C22, CB_C01, CB_C02, CB_C12};
constexpr uint32_t CC_SCRATCH[6] = {
    CB_TMP_CC00, CB_TMP_CC01, CB_TMP_CC02,
    CB_TMP_CC11, CB_TMP_CC12, CB_TMP_CC22};

inline void emit_dst(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// Pack tile from DEST slot 0 to scratch CB (no cb_pop_front; callers pop later).
inline void emit_scratch(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// dst[idst] = sum_k scales[k] * COV3D_CB[k]  (cov_cam entry expansion).
// scales must already be loaded via get_arg_val.
inline void compute_cc_entry_to_scratch(uint32_t base_arg, uint32_t cb_out_scratch) {
    const uint32_t s0 = get_arg_val<uint32_t>(base_arg + 0);
    const uint32_t s1 = get_arg_val<uint32_t>(base_arg + 1);
    const uint32_t s2 = get_arg_val<uint32_t>(base_arg + 2);
    const uint32_t s3 = get_arg_val<uint32_t>(base_arg + 3);
    const uint32_t s4 = get_arg_val<uint32_t>(base_arg + 4);
    const uint32_t s5 = get_arg_val<uint32_t>(base_arg + 5);
    const uint32_t scales[6] = {s0, s1, s2, s3, s4, s5};

    tile_regs_acquire();
    copy_tile_to_dst_init_short(COV3D_CB[0]);
    copy_tile(COV3D_CB[0], 0, 0);
    mul_unary_tile(0, scales[0]);
    for (uint32_t k = 1; k < 6; k++) {
        copy_tile_to_dst_init_short(COV3D_CB[k]);
        copy_tile(COV3D_CB[k], 0, 1);
        mul_unary_tile(1, scales[k]);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    emit_scratch(0, cb_out_scratch);
    tile_regs_release();
}

inline void translate_and_pack(uint32_t cb_in, uint32_t t_bits, uint32_t cb_out) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    add_unary_tile(0, t_bits);
    tile_regs_commit();
    tile_regs_wait();
    emit_scratch(0, cb_out);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(10);
    const uint32_t t1 = get_arg_val<uint32_t>(11);
    const uint32_t t2 = get_arg_val<uint32_t>(12);

    const uint32_t fx       = get_arg_val<uint32_t>(13);
    const uint32_t fy       = get_arg_val<uint32_t>(14);
    const uint32_t cx       = get_arg_val<uint32_t>(15);
    const uint32_t cy       = get_arg_val<uint32_t>(16);
    const uint32_t k_bits   = get_arg_val<uint32_t>(53);  // radii scale k = 3.0
    const uint32_t neg_fx_bits = get_arg_val<uint32_t>(54);
    const uint32_t neg_fy_bits = get_arg_val<uint32_t>(55);
    constexpr uint32_t two_fp32_bits = 0x40000000U;
    constexpr uint32_t pt3_fp32_bits = 0x3E99999AU;  // 0.3f

    init_sfpu(CB_MCX, CB_M2X);
    add_binary_tile_init();
    mul_binary_tile_init();
    recip_tile_init();
    sqrt_tile_init();
    relu_tile_init();
    rounding_op_tile_init();

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        cb_wait_front(CB_MCX, 1);
        cb_wait_front(CB_MCY, 1);
        cb_wait_front(CB_MCZ, 1);
        cb_wait_front(CB_C00, 1);
        cb_wait_front(CB_C01, 1);
        cb_wait_front(CB_C02, 1);
        cb_wait_front(CB_C11, 1);
        cb_wait_front(CB_C12, 1);
        cb_wait_front(CB_C22, 1);

        // ── 1. Translation → tx, ty, tz scratch
        translate_and_pack(CB_MCX, t0, CB_TMP_TX);
        translate_and_pack(CB_MCY, t1, CB_TMP_TY);
        translate_and_pack(CB_MCZ, t2, CB_TMP_TZ);

        // ── 2. inv_tz = 1/tz → scratch
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            recip_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_INV_TZ);
            tile_regs_release();
        }

        // ── 3. depth = tz → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_DEP);
            tile_regs_release();
        }

        // ── 4. mean_x = fx · tx · inv_tz + cx → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 0);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_binary_tile(0, 1, 0);   // dst[0] = tx · inv_tz
            mul_unary_tile(0, fx);
            add_unary_tile(0, cx);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_M2X);
            tile_regs_release();
        }

        // ── 5. mean_y = fy · ty · inv_tz + cy → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 0);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_binary_tile(0, 1, 0);
            mul_unary_tile(0, fy);
            add_unary_tile(0, cy);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_M2Y);
            tile_regs_release();
        }

        // ── 6. cov_cam (6 unique entries) → scratch CBs
        // CC_SCRATCH order matches host scale layout (cc00, cc01, cc02, cc11, cc12, cc22).
        for (uint32_t e = 0; e < 6; e++) {
            compute_cc_entry_to_scratch(17 + e * 6, CC_SCRATCH[e]);
        }

        // Drain cov3d inputs — done feeding cov_cam.
        cb_pop_front(CB_C00, 1);
        cb_pop_front(CB_C01, 1);
        cb_pop_front(CB_C02, 1);
        cb_pop_front(CB_C11, 1);
        cb_pop_front(CB_C12, 1);
        cb_pop_front(CB_C22, 1);

        // ── 7. cov2d_a = j00² · cc00 + 2·j00·j02 · cc02 + j02² · cc22 + 0.3
        // j00 = fx · inv_tz, j02 = -fx · tx · inv_tz² = -fx · tx · inv_tz · inv_tz
        // Compute j00 once into dst[1], j02 once into dst[2], reuse for all 3
        // cov2d outputs to avoid recomputing.
        //
        // Hold layout: dst[1] = j00, dst[2] = j02, dst[3] = j11, dst[4] = j12
        // dst[0] = accumulator, dst[5..6] = scratch
        {
            tile_regs_acquire();

            // dst[1] = j00 = fx · inv_tz
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);

            // dst[2] = j02 = -fx · tx · inv_tz²
            //        = (-fx · tx) · inv_tz · inv_tz
            // Use: dst[2] = tx; mul_unary -fx; then mul_binary with inv_tz twice.
            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);    // dst[5] = inv_tz scratch
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz²

            // dst[3] = j11 = fy · inv_tz
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);

            // dst[4] = j12 = -fy · ty · inv_tz²
            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            mul_binary_tile(4, 5, 4);          // dst[4] = -fy·ty·inv_tz
            mul_binary_tile(4, 5, 4);          // dst[4] = -fy·ty·inv_tz²

            // ── compute cov2d_a = j00²·cc00 + 2·j00·j02·cc02 + j02²·cc22 + 0.3 ──
            // accumulator init — load cc00 into dst[0], then mul by j00²
            copy_tile_to_dst_init_short(CB_TMP_CC00);
            copy_tile(CB_TMP_CC00, 0, 0);    // dst[0] = cc00
            // dst[5] = j00 (copy from dst[1] via scratch since DEST→DEST not direct)
            // Approach: just remul. We need cc00 · j00². Compute step by step.
            //   dst[0] = cc00 · j00 (via mul_binary)
            mul_binary_tile(0, 1, 0);        // dst[0] = cc00 · j00
            mul_binary_tile(0, 1, 0);        // dst[0] = cc00 · j00²

            // Add 2·j00·j02·cc02 in dst[5]
            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);    // dst[5] = cc02
            mul_binary_tile(5, 1, 5);        // dst[5] = cc02 · j00
            mul_binary_tile(5, 2, 5);        // dst[5] = cc02 · j00 · j02
            mul_unary_tile(5, two_fp32_bits);   // dst[5] = 2·cc02·j00·j02
            add_binary_tile(0, 5, 0);        // dst[0] += dst[5]

            // Add j02²·cc22
            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);    // dst[5] = cc22
            mul_binary_tile(5, 2, 5);        // dst[5] = cc22 · j02
            mul_binary_tile(5, 2, 5);        // dst[5] = cc22 · j02²
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);   // dst[0] = a

            // emit a → output
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_A);
            tile_regs_release();
        }

        // ── 8. cov2d_b = j00·j11·cc01 + j00·j12·cc02 + j02·j11·cc12 + j02·j12·cc22
        {
            tile_regs_acquire();

            // Recompute j00, j02, j11, j12 (DEST released between blocks)
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);    // dst[5] = inv_tz
            mul_binary_tile(2, 5, 2);
            mul_binary_tile(2, 5, 2);          // dst[2] = j02

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            // dst[0] = j00·j11·cc01
            copy_tile_to_dst_init_short(CB_TMP_CC01);
            copy_tile(CB_TMP_CC01, 0, 0);      // dst[0] = cc01
            mul_binary_tile(0, 1, 0);          // dst[0] *= j00
            mul_binary_tile(0, 3, 0);          // dst[0] *= j11

            // dst[5] = j00·j12·cc02
            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);          // dst[5] = cc02·j00
            mul_binary_tile(5, 4, 5);          // dst[5] *= j12
            add_binary_tile(0, 5, 0);

            // dst[5] = j02·j11·cc12
            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 3, 5);          // *= j11
            add_binary_tile(0, 5, 0);

            // dst[5] = j02·j12·cc22
            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 4, 5);          // *= j12
            add_binary_tile(0, 5, 0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_B);
            tile_regs_release();
        }

        // ── 9. cov2d_c = j11²·cc11 + 2·j11·j12·cc12 + j12²·cc22 + 0.3
        {
            tile_regs_acquire();

            // dst[3] = j11 = fy · inv_tz
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);

            // dst[4] = j12
            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);

            // dst[0] = j11²·cc11
            copy_tile_to_dst_init_short(CB_TMP_CC11);
            copy_tile(CB_TMP_CC11, 0, 0);
            mul_binary_tile(0, 3, 0);
            mul_binary_tile(0, 3, 0);

            // dst[5] = 2·j11·j12·cc12
            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 3, 5);
            mul_binary_tile(5, 4, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);

            // dst[5] = j12²·cc22
            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 4, 5);
            mul_binary_tile(5, 4, 5);
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_C);
            tile_regs_release();
        }

        // ── 10. radii.x = ceil(k · sqrt(max(a, 0)))
        // The CB_A buffer was just produced — re-read it for radii.
        // Need to wait until we've finished writing then read back. Since
        // we used cb_push_back(CB_A, 1) above and don't pop it (the writer
        // kernel drains), we cannot read it from inside this compute kernel.
        //
        // Workaround: recompute a from the same scratch values. Then radii.x.
        {
            tile_regs_acquire();
            // Recompute Jacobian j00, j02
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(2, 5, 2);
            mul_binary_tile(2, 5, 2);          // dst[2] = j02

            // Recompute a (same as step 7).
            copy_tile_to_dst_init_short(CB_TMP_CC00);
            copy_tile(CB_TMP_CC00, 0, 0);
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(0, 1, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);
            mul_binary_tile(5, 2, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);
            mul_binary_tile(5, 2, 5);
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);  // dst[0] = a

            // radii_x = ceil(k · sqrt(max(a, 0)))
            relu_tile(0);                       // dst[0] = max(a, 0)
            sqrt_tile(0);
            mul_unary_tile(0, k_bits);
            ceil_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_RX);
            tile_regs_release();
        }

        // ── 11. radii.y = ceil(k · sqrt(max(c, 0))) — recompute c too.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC11);
            copy_tile(CB_TMP_CC11, 0, 0);
            mul_binary_tile(0, 3, 0);
            mul_binary_tile(0, 3, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 3, 5);
            mul_binary_tile(5, 4, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 4, 5);
            mul_binary_tile(5, 4, 5);
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);

            relu_tile(0);
            sqrt_tile(0);
            mul_unary_tile(0, k_bits);
            ceil_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_RY);
            tile_regs_release();
        }

        // ── 12. Drain scratch CBs (consumed by all compute above).
        cb_pop_front(CB_TMP_TX, 1);
        cb_pop_front(CB_TMP_TY, 1);
        cb_pop_front(CB_TMP_TZ, 1);
        cb_pop_front(CB_TMP_INV_TZ, 1);
        cb_pop_front(CB_TMP_CC00, 1);
        cb_pop_front(CB_TMP_CC01, 1);
        cb_pop_front(CB_TMP_CC02, 1);
        cb_pop_front(CB_TMP_CC11, 1);
        cb_pop_front(CB_TMP_CC12, 1);
        cb_pop_front(CB_TMP_CC22, 1);

        // ── 13. Drain remaining inputs (means_cam).
        cb_pop_front(CB_MCX, 1);
        cb_pop_front(CB_MCY, 1);
        cb_pop_front(CB_MCZ, 1);
    }
}
