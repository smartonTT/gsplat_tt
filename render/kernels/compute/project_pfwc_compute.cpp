// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED project(means_cam) + pfwc compute kernel — iter-133 (program fusion #1).
//
// This kernel merges the former standalone `project_means_cam` program and the
// `pfwc` program into ONE device program, eliminating one per-program launch's
// firmware (BRISC-FW long pole) and the fp32 means_cam DRAM store+reload between
// the two stages. The means_cam handoff is now a CORE-LOCAL L1 pass: the
// world→camera transform writes its result straight into the per-chunk L1 DEST
// and folds the pfwc translation in, emitting tx/ty/tz directly — no NoC / DRAM
// round-trip.
//
// BIT-IDENTICAL: the former means_cam stored mc_j = R·m to fp32 DRAM (lossless)
// and pfwc reloaded it then added t_j (tx = mc_j + t_j). Here mc_j stays in an
// fp32 DEST slot and add_unary(t_j) is applied to the same fp32 value, so the
// emitted tx/ty/tz are bit-for-bit the values the two-program path produced.
// Everything downstream (steps 2..11) is the verbatim pfwc body on the same
// inputs.
//
// LAYOUT
//   Inputs  CB_MX/MY/MZ (0,1,2)  — fp32 world-space means SoA (reader-filled).
//           CB_C00..CB_C22 (3..8) — fp32 cov3d unique SoA (reader-filled).
//   Outputs CB_M2X..CB_RY (9..16) — the 8 pfwc outputs (writer-drained).
//   Scratch CB 17..29             — tx/ty/tz/inv_tz, cc00..cc22, conic a/b/c.
//   (33→30 CBs: the old means_cam MCX/MCY/MCZ bridge is gone — tx/ty/tz IS the
//   L1 bridge now.)
//
// RUNTIME ARGS (uint32_t, layout identical to the former pfwc_device.cpp):
//   0           : num_chunks
//   1..9        : R bits (row-major r00..r22) — USED by the fused transform
//   10..12      : t0, t1, t2 (translation)    — USED by the fused transform
//   13..16      : fx, fy, cx, cy
//   17..52      : 36 cov_cam scales (6 entries × 6 scales)
//   53          : k_bits (fp32 of k = 3.0 default)
//   54          : neg_fx_bits (fp32 of -fx)
//   55          : neg_fy_bits (fp32 of -fy)

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
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

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

namespace {

// Means inputs (CBs 0,1,2) — reader fills with world-space means SoA. These are
// the same physical CBs the standalone means_cam read; the fused transform turns
// them into camera-space tx/ty/tz in L1.
constexpr uint32_t CB_MX = 0;
constexpr uint32_t CB_MY = 1;
constexpr uint32_t CB_MZ = 2;
// cov3d unique inputs.
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

// Scratch (per-chunk intermediates). CB_TMP_TX/TY/TZ double as the means_cam→pfwc
// L1 bridge: the fused transform emits camera-space tx/ty/tz (= R·m + t) here.
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
constexpr uint32_t CB_TMP_A      = 27;
constexpr uint32_t CB_TMP_B      = 28;
constexpr uint32_t CB_TMP_C      = 29;

constexpr uint32_t COV3D_CB[6] = {CB_C00, CB_C11, CB_C22, CB_C01, CB_C02, CB_C12};
constexpr uint32_t CC_SCRATCH[6] = {
    CB_TMP_CC00, CB_TMP_CC01, CB_TMP_CC02,
    CB_TMP_CC11, CB_TMP_CC12, CB_TMP_CC22};

inline void emit_dst(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// Pack tile from DEST slot to scratch CB (no cb_pop_front; callers pop later).
inline void emit_scratch(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// dst[idst] = sum_k scales[k] * COV3D_CB[k]  (cov_cam entry expansion).
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

#ifdef TRISC_MATH
// A1 conic fold for ONE 32-lane vector V of the chunk tile. DEST tile 0 holds
// cov2d_a, tile 1 cov2d_b, tile 2 cov2d_c. Fold a,b,c -> A,B,C in place. Lifted
// verbatim from alpha_blend_compute_mb.cpp so the emitted A,B,C are bit-identical
// to what the blend kernel would have computed.
template <uint32_t V>
__attribute__((noinline, noipa)) void pfwc_conic_one() {
    using namespace sfpi;
    vFloat cov_a = dst_reg[0 * 32 + V];
    vFloat cov_b = dst_reg[1 * 32 + V];
    vFloat cov_c = dst_reg[2 * 32 + V];
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    vFloat det_floor = 1e-6f;
    vec_min_max(det_floor, det);          // det = max(det, 1e-6)
    vFloat inv = approx_recip(det);
    inv = inv * (vFloat(2.0f) - det * inv);
    inv = inv * (vFloat(2.0f) - det * inv);
    dst_reg[0 * 32 + V] = vFloat(-0.5f) * (cov_c * inv);  // A
    dst_reg[1 * 32 + V] = cov_b * inv;                    // B
    dst_reg[2 * 32 + V] = vFloat(-0.5f) * (cov_a * inv);  // C
}
#endif

template <uint32_t V>
inline void pfwc_conic_unroll() {
    if constexpr (V < 32) {
        MATH((pfwc_conic_one<V>()));
        pfwc_conic_unroll<V + 1>();
    }
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("pfwc");  // Tracy stage label (fused project+pfwc compute)
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    // R (row-major) + t for the fused world→camera transform.
    uint32_t r_bits[9];
    for (uint32_t k = 0; k < 9; k++) r_bits[k] = get_arg_val<uint32_t>(1 + k);
    const uint32_t t0 = get_arg_val<uint32_t>(10);
    const uint32_t t1 = get_arg_val<uint32_t>(11);
    const uint32_t t2 = get_arg_val<uint32_t>(12);
    const uint32_t t_bits[3] = {t0, t1, t2};

    const uint32_t fx       = get_arg_val<uint32_t>(13);
    const uint32_t fy       = get_arg_val<uint32_t>(14);
    const uint32_t cx       = get_arg_val<uint32_t>(15);
    const uint32_t cy       = get_arg_val<uint32_t>(16);
    const uint32_t k_bits   = get_arg_val<uint32_t>(53);  // radii scale k = 3.0
    const uint32_t neg_fx_bits = get_arg_val<uint32_t>(54);
    const uint32_t neg_fy_bits = get_arg_val<uint32_t>(55);
    constexpr uint32_t two_fp32_bits = 0x40000000U;
    constexpr uint32_t pt3_fp32_bits = 0x3E99999AU;  // 0.3f

    init_sfpu(CB_MX, CB_M2X);
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
        cb_wait_front(CB_MX, 1);
        cb_wait_front(CB_MY, 1);
        cb_wait_front(CB_MZ, 1);
        cb_wait_front(CB_C00, 1);
        cb_wait_front(CB_C01, 1);
        cb_wait_front(CB_C02, 1);
        cb_wait_front(CB_C11, 1);
        cb_wait_front(CB_C12, 1);
        cb_wait_front(CB_C22, 1);

        // ── 1. FUSED transform: tx/ty/tz = R · means + t  (means_cam folded
        //      with the pfwc translation; L1 bridge into the pfwc body below).
        //      Per j the sum mx*r_j0 + my*r_j1 + mz*r_j2 is computed by the SAME
        //      SFPU op sequence the standalone means_cam used, in an fp32 DEST
        //      slot, then add_unary(t_j) — bit-identical to the old fp32-DRAM
        //      handoff + pfwc translate.
        for (uint32_t j = 0; j < 3; j++) {
            const uint32_t r_j0 = r_bits[j * 3 + 0];
            const uint32_t r_j1 = r_bits[j * 3 + 1];
            const uint32_t r_j2 = r_bits[j * 3 + 2];
            const uint32_t cb_out = CB_TMP_TX + j;

            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MX);
            copy_tile(CB_MX, 0, 0);
            copy_tile(CB_MY, 0, 1);
            copy_tile(CB_MZ, 0, 2);
            mul_unary_tile(0, r_j0);
            mul_unary_tile(1, r_j1);
            mul_unary_tile(2, r_j2);
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);
            add_unary_tile(0, t_bits[j]);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, cb_out);
            tile_regs_release();
        }

        // Means consumed — drain so the reader can refill for the next chunk.
        cb_pop_front(CB_MX, 1);
        cb_pop_front(CB_MY, 1);
        cb_pop_front(CB_MZ, 1);

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

        // ── 7. cov2d_a = j00²·cc00 + 2·j00·j02·cc02 + j02²·cc22 + 0.3
        {
            tile_regs_acquire();

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00 = fx · inv_tz

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);    // dst[5] = inv_tz scratch
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz² = j02

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11 = fy · inv_tz

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC00);
            copy_tile(CB_TMP_CC00, 0, 0);      // dst[0] = cc00
            mul_binary_tile(0, 1, 0);          // dst[0] = cc00 · j00
            mul_binary_tile(0, 1, 0);          // dst[0] = cc00 · j00²

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);          // dst[5] = cc02 · j00
            mul_binary_tile(5, 2, 5);          // dst[5] = cc02 · j00 · j02
            mul_unary_tile(5, two_fp32_bits);  // dst[5] = 2·cc02·j00·j02
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);          // dst[5] = cc22 · j02
            mul_binary_tile(5, 2, 5);          // dst[5] = cc22 · j02²
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);  // dst[0] = a

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_A);
            tile_regs_release();
        }

        // ── 8. cov2d_b = j00·j11·cc01 + j00·j12·cc02 + j02·j11·cc12 + j02·j12·cc22
        {
            tile_regs_acquire();

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

            copy_tile_to_dst_init_short(CB_TMP_CC01);
            copy_tile(CB_TMP_CC01, 0, 0);      // dst[0] = cc01
            mul_binary_tile(0, 1, 0);          // dst[0] *= j00
            mul_binary_tile(0, 3, 0);          // dst[0] *= j11

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);          // dst[5] = cc02·j00
            mul_binary_tile(5, 4, 5);          // dst[5] *= j12
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 3, 5);          // *= j11
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 4, 5);          // *= j12
            add_binary_tile(0, 5, 0);

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_B);
            tile_regs_release();
        }

        // ── 9. cov2d_c = j11²·cc11 + 2·j11·j12·cc12 + j12²·cc22 + 0.3
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
            mul_binary_tile(0, 3, 0);          // dst[0] = j11²·cc11

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 3, 5);
            mul_binary_tile(5, 4, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);          // dst[0] += 2·j11·j12·cc12

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 4, 5);
            mul_binary_tile(5, 4, 5);
            add_binary_tile(0, 5, 0);          // dst[0] += j12²·cc22

            add_unary_tile(0, pt3_fp32_bits);

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_C);
            tile_regs_release();
        }

        // ── 9.5 (A1). Conic fold: A,B,C from scratch a,b,c → cov2d outputs.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_A);
            copy_tile(CB_TMP_A, 0, 0);
            copy_tile_to_dst_init_short(CB_TMP_B);
            copy_tile(CB_TMP_B, 0, 1);
            copy_tile_to_dst_init_short(CB_TMP_C);
            copy_tile(CB_TMP_C, 0, 2);

            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            pfwc_conic_unroll<0>();
            MATH((_llk_math_eltwise_unary_sfpu_done_()));

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_A);
            emit_dst(1, CB_B);
            emit_dst(2, CB_C);
            tile_regs_release();
        }

        // ── 10. radii.x = ceil(k · sqrt(max(a, 0))) — recompute a.
        {
            tile_regs_acquire();
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

            relu_tile(0);                       // dst[0] = max(a, 0)
            sqrt_tile(0);
            mul_unary_tile(0, k_bits);
            ceil_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_RX);
            tile_regs_release();
        }

        // ── 11. radii.y = ceil(k · sqrt(max(c, 0))) — recompute c.
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

        // ── 12. Drain scratch CBs.
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
        cb_pop_front(CB_TMP_A, 1);
        cb_pop_front(CB_TMP_B, 1);
        cb_pop_front(CB_TMP_C, 1);
    }
}
