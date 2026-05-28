// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc (project_full_with_cov3d) device compute kernel — amendment-002 tt-008a.
//
// Replaces the heaviest sub-steps of gsplat_cpu::project_full_fused
// per-Gaussian inner loop:
//   * translation (mc + t)
//   * perspective division   -> means_2d, depths
//   * cov_cam = R · cov3d · R^T  -> 6 unique entries (cc00..cc22)
//
// Leaves on the host:
//   * J = Jacobian (4 entries) — needs tx,ty,tz which we don't ship back
//     (only mean_x,mean_y,depth=tz). cov2d uses J, so cov2d stays on host.
//   * cov2d, radii, valid_mask — small compute relative to cov_cam.
//
// The big win: cov_cam is the dominant pfwc cost (45 mul + 30 add per
// Gaussian, ~70% of pfwc compute) and is now on-device. Outputs are
// downloaded once per frame; downstream stages still consume host arrays.
//
// INPUTS  (CBs 0..8, fp32, all UnpackToDestFp32):
//   CB 0..2  : mcx, mcy, mcz       (means_cam — device-resident from tt-005b)
//   CB 3..8  : c00 c01 c02 c11 c12 c22  (cov3d unique entries, symmetric)
//
// OUTPUTS (CBs 9..17, fp32):
//   CB 9..10 : mean_x_2d, mean_y_2d
//   CB 11    : depth (= tz)
//   CB 12..17: cc00, cc01, cc02, cc11, cc12, cc22  (cov_cam unique)
//
// SCRATCH (CBs 18..20):
//   CB 18..20: tx, ty, tz (after translation; recomputed inv_tz inline)
//
// PRECISION — same playbook as tt-007: UnpackToDestFp32 + SFPU add_binary_tile
// (DEST-slot adder, fp32-clean) + scalar muls via mul_unary_tile.
//
// RUNTIME ARGS (uint32_t, layout chosen to match host driver):
//   0           : num_chunks
//   1..9        : R matrix bits, row-major (r00 r01 r02 r10 r11 r12 r20 r21 r22)
//   10..12      : t0, t1, t2          (translation)
//   13..16      : fx, fy, cx, cy
//   17..52      : 6 cov_cam entries × 6 scales, packed as
//                 (entry_idx=0..5) × (s00 s11 s22 s01 s02 s12)
//                 host pre-computes these per view from R: see pfwc_device.cpp.
//                 Entry order: cc00, cc01, cc02, cc11, cc12, cc22.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"

namespace {

constexpr uint32_t CB_MCX = 0;
constexpr uint32_t CB_MCY = 1;
constexpr uint32_t CB_MCZ = 2;
constexpr uint32_t CB_C00 = 3;
constexpr uint32_t CB_C01 = 4;
constexpr uint32_t CB_C02 = 5;
constexpr uint32_t CB_C11 = 6;
constexpr uint32_t CB_C12 = 7;
constexpr uint32_t CB_C22 = 8;

constexpr uint32_t CB_M2X  = 9;
constexpr uint32_t CB_M2Y  = 10;
constexpr uint32_t CB_DEP  = 11;
constexpr uint32_t CB_CC00 = 12;
constexpr uint32_t CB_CC01 = 13;
constexpr uint32_t CB_CC02 = 14;
constexpr uint32_t CB_CC11 = 15;
constexpr uint32_t CB_CC12 = 16;
constexpr uint32_t CB_CC22 = 17;

constexpr uint32_t CB_TMP_TX  = 18;
constexpr uint32_t CB_TMP_TY  = 19;
constexpr uint32_t CB_TMP_TZ  = 20;

constexpr uint32_t COV3D_CB[6] = {CB_C00, CB_C11, CB_C22, CB_C01, CB_C02, CB_C12};

inline void emit_dst(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// dst[0] = c_k * scale_k summed over k=0..5 where c_k follows COV3D_CB order
// (c00, c11, c22, c01, c02, c12) and scale_k is the matching runtime arg.
inline void compute_cc_entry(uint32_t base_arg, uint32_t cb_out) {
    const uint32_t s0 = get_arg_val<uint32_t>(base_arg + 0);  // x c00
    const uint32_t s1 = get_arg_val<uint32_t>(base_arg + 1);  // x c11
    const uint32_t s2 = get_arg_val<uint32_t>(base_arg + 2);  // x c22
    const uint32_t s3 = get_arg_val<uint32_t>(base_arg + 3);  // x c01
    const uint32_t s4 = get_arg_val<uint32_t>(base_arg + 4);  // x c02
    const uint32_t s5 = get_arg_val<uint32_t>(base_arg + 5);  // x c12
    const uint32_t scales[6] = {s0, s1, s2, s3, s4, s5};

    tile_regs_acquire();
    // dst[0] = COV3D_CB[0] * scales[0]
    copy_tile_to_dst_init_short(COV3D_CB[0]);
    copy_tile(COV3D_CB[0], 0, 0);
    mul_unary_tile(0, scales[0]);
    // dst[0] += COV3D_CB[k] * scales[k] for k=1..5
    for (uint32_t k = 1; k < 6; k++) {
        copy_tile_to_dst_init_short(COV3D_CB[k]);
        copy_tile(COV3D_CB[k], 0, 1);
        mul_unary_tile(1, scales[k]);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    emit_dst(0, cb_out);
    tile_regs_release();
}

// dst[0] = (cb_in + t) packed to cb_out.
inline void translate_and_pack(uint32_t cb_in, uint32_t t_bits, uint32_t cb_out) {
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    add_unary_tile(0, t_bits);
    tile_regs_commit();
    tile_regs_wait();
    emit_dst(0, cb_out);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);
    // R matrix consumed implicitly via the 36 cov_cam scales the host packs
    // (s_ij_k = combination of R entries — see pfwc_device.cpp). We still
    // expose R bits 1..9 for reader/writer share, but this kernel only uses
    // the precomputed scales.

    const uint32_t t0 = get_arg_val<uint32_t>(10);
    const uint32_t t1 = get_arg_val<uint32_t>(11);
    const uint32_t t2 = get_arg_val<uint32_t>(12);

    const uint32_t fx = get_arg_val<uint32_t>(13);
    const uint32_t fy = get_arg_val<uint32_t>(14);
    const uint32_t cx = get_arg_val<uint32_t>(15);
    const uint32_t cy = get_arg_val<uint32_t>(16);

    init_sfpu(CB_MCX, CB_M2X);
    add_binary_tile_init();
    mul_binary_tile_init();
    recip_tile_init();

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

        // ── 1. Translation: tx=mcx+t0, ty=mcy+t1, tz=mcz+t2 → scratch CBs
        translate_and_pack(CB_MCX, t0, CB_TMP_TX);
        translate_and_pack(CB_MCY, t1, CB_TMP_TY);
        translate_and_pack(CB_MCZ, t2, CB_TMP_TZ);

        // ── 2. depth = tz → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_DEP);
            tile_regs_release();
        }

        // ── 3. mean_x = fx * tx * inv_tz + cx  (inv_tz computed in-place)
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 0);  // dst[0] = tx
            copy_tile(CB_TMP_TZ, 0, 1);  // dst[1] = tz
            recip_tile(1);                // dst[1] = 1/tz
            mul_binary_tile(0, 1, 0);     // dst[0] = tx * inv_tz
            mul_unary_tile(0, fx);
            add_unary_tile(0, cx);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_M2X);
            tile_regs_release();
        }

        // ── 4. mean_y = fy * ty * inv_tz + cy  (recompute inv_tz)
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 0);
            copy_tile(CB_TMP_TZ, 0, 1);
            recip_tile(1);
            mul_binary_tile(0, 1, 0);
            mul_unary_tile(0, fy);
            add_unary_tile(0, cy);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_M2Y);
            tile_regs_release();
        }

        // tx/ty/tz scratch CBs are consumed — drain them.
        cb_pop_front(CB_TMP_TX, 1);
        cb_pop_front(CB_TMP_TY, 1);
        cb_pop_front(CB_TMP_TZ, 1);

        // ── 6. cov_cam (6 unique entries) ─────────────────────────────────
        // Each entry: dst[0] = sum_k c_k * s_k where s_k are precomputed
        // scales (functions of R). Coefficients laid out in runtime args
        // starting at offset 17 (6 entries × 6 scales = 36 args).
        constexpr uint32_t CC_OUT[6] = {
            CB_CC00, CB_CC01, CB_CC02, CB_CC11, CB_CC12, CB_CC22};
        for (uint32_t e = 0; e < 6; e++) {
            compute_cc_entry(17 + e * 6, CC_OUT[e]);
        }

        // ── 7. Drain inputs ───────────────────────────────────────────────
        cb_pop_front(CB_MCX, 1);
        cb_pop_front(CB_MCY, 1);
        cb_pop_front(CB_MCZ, 1);
        cb_pop_front(CB_C00, 1);
        cb_pop_front(CB_C01, 1);
        cb_pop_front(CB_C02, 1);
        cb_pop_front(CB_C11, 1);
        cb_pop_front(CB_C12, 1);
        cb_pop_front(CB_C22, 1);
    }
}
