// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// project_means_cam compute kernel — amendment-002 tt-005 / tt-007.
//
// Computes means_cam[i] = R @ means[i] for N Gaussians (translation +t added
// later in the per-Gaussian inner loop, on CPU until tt-008 ports pfwc).
//
// LAYOUT
//   Inputs  CB_MX, CB_MY, CB_MZ   — fp32, one 32x32 tile = 1024 Gaussians.
//   Outputs CB_MCX, CB_MCY, CB_MCZ — fp32 SoA, one tile per chunk.
//
// MATH (per output j in {0,1,2}):
//   mc_j = mx*r_j0 + my*r_j1 + mz*r_j2
//
// PRECISION (tt-007)
// ------------------
// Earlier revisions used FPU `add_tiles` to combine partial products via CB
// scratch. add_tiles routes through SrcA/SrcB which truncate fp32 inputs to
// bf16 on Blackhole — that is the source of the 34.5 dB hero PSNR regression
// observed at iter-06e (≈0.001 relative error/Gaussian, amplified by blend).
//
// This revision keeps everything in DEST:
//   * UnpackToDestFp32 mode is set for the input CBs in
//     project_device.cpp ComputeConfig.unpack_to_dest_mode — copy_tile lands
//     fp32 directly in DEST, bypassing SrcA/SrcB.
//   * Per-coord scalar multiply uses SFPU `mul_unary_tile`
//     (vFloat * Converter::as_float(uint32) — fp32-clean).
//   * Three partials are summed via SFPU `add_binary_tile(i, j, k)` which
//     reads/writes DEST slots directly. No FPU add_tiles, no L1 round-trip.
//
// RUNTIME ARGS
//   0:    num_chunks  (tiles this core processes; 1024 Gaussians per tile)
//   1..9: R matrix entries as fp32 bits (r00..r02 r10..r12 r20..r22)

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    DeviceZoneScopedN("mcam");  // Tracy device-timeline stage label (project means-cam compute)
    uint32_t num_chunks = get_arg_val<uint32_t>(0);
    uint32_t r_bits[9];
    for (uint32_t k = 0; k < 9; k++) r_bits[k] = get_arg_val<uint32_t>(1 + k);

    constexpr uint32_t CB_MX  = 0;
    constexpr uint32_t CB_MY  = 1;
    constexpr uint32_t CB_MZ  = 2;
    constexpr uint32_t CB_MCX = 3;
    constexpr uint32_t CB_MCY = 4;
    constexpr uint32_t CB_MCZ = 5;

    // One-time SFPU + datacopy init. mul_unary_tile / add_binary_tile share
    // the same SFPU and only need add_binary_tile_init for the binary op
    // descriptor.
    init_sfpu(CB_MX, CB_MCX);
    add_binary_tile_init();

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        cb_wait_front(CB_MX, 1);
        cb_wait_front(CB_MY, 1);
        cb_wait_front(CB_MZ, 1);

        for (uint32_t j = 0; j < 3; j++) {
            const uint32_t r_j0 = r_bits[j * 3 + 0];
            const uint32_t r_j1 = r_bits[j * 3 + 1];
            const uint32_t r_j2 = r_bits[j * 3 + 2];
            const uint32_t cb_out = CB_MCX + j;

            tile_regs_acquire();

            // Reload fresh inputs each j (mul_unary clobbers DEST slots).
            copy_tile_to_dst_init_short(CB_MX);
            copy_tile(CB_MX, 0, 0);
            copy_tile(CB_MY, 0, 1);
            copy_tile(CB_MZ, 0, 2);

            mul_unary_tile(0, r_j0);
            mul_unary_tile(1, r_j1);
            mul_unary_tile(2, r_j2);

            // dest[0] = dest[0] + dest[1] + dest[2]   (SFPU, fp32 throughout)
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }

        cb_pop_front(CB_MX, 1);
        cb_pop_front(CB_MY, 1);
        cb_pop_front(CB_MZ, 1);
    }
}
