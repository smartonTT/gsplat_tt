// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// project_means_cam compute kernel — amendment-002 tt-005 (bounded hotspot
// port). Computes means_cam[i] = R @ means[i] + t for N Gaussians.
//
// LAYOUT
// ------
// Inputs (3 SoA streams, fp32, one 32x32 tile = 1024 Gaussians per chunk):
//   CB_MX  - x coordinates of means (world space)
//   CB_MY  - y coordinates
//   CB_MZ  - z coordinates
// Outputs (3 SoA streams, fp32):
//   CB_MCX - x coordinates of means_cam (camera space)
//   CB_MCY - y coordinates
//   CB_MCZ - z coordinates
//
// MATH (per output coordinate j in {0,1,2}):
//   mc_j = mx*r_j0 + my*r_j1 + mz*r_j2
//
// Matches the CPU `means_cam` semantics (translation +t is added in the
// per-Gaussian inner loop of project_full_fused — see project.cpp:546-549).
//
// SCHEDULE (per chunk of 1024 Gaussians):
//   For j in {0,1,2}:
//     prod_x  = mx * r_j0          (copy_tile + mul_unary_tile)
//     prod_y  = my * r_j1
//     sum_xy  = prod_x + prod_y    (add_tiles)
//     prod_z  = mz * r_j2
//     mc_j    = sum_xy + prod_z    (add_tiles)
//
// RUNTIME ARGS
//   0:    num_chunks  (tiles this core processes; 1024 Gaussians per tile)
//   1..9: R matrix entries as fp32 bits (r00, r01, r02, r10, r11, r12, r20, r21, r22)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

void kernel_main() {
    uint32_t num_chunks = get_arg_val<uint32_t>(0);
    uint32_t r_bits[9];
    for (uint32_t k = 0; k < 9; k++) r_bits[k] = get_arg_val<uint32_t>(1 + k);

    constexpr uint32_t CB_MX    = 0;
    constexpr uint32_t CB_MY    = 1;
    constexpr uint32_t CB_MZ    = 2;
    constexpr uint32_t CB_MCX   = 3;
    constexpr uint32_t CB_MCY   = 4;
    constexpr uint32_t CB_MCZ   = 5;
    constexpr uint32_t CB_TMP_A = 6;
    constexpr uint32_t CB_TMP_B = 7;

    binary_op_init_common(CB_MX, CB_MY, CB_TMP_A);

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

            // prod_x = mx * r_j0 -> CB_TMP_A
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MX);
            copy_tile(CB_MX, 0, 0);
            mul_unary_tile(0, r_j0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_TMP_A, 1);
            pack_tile(0, CB_TMP_A);
            cb_push_back(CB_TMP_A, 1);
            tile_regs_release();

            // prod_y = my * r_j1 -> CB_TMP_B
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MY);
            copy_tile(CB_MY, 0, 0);
            mul_unary_tile(0, r_j1);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_TMP_B, 1);
            pack_tile(0, CB_TMP_B);
            cb_push_back(CB_TMP_B, 1);
            tile_regs_release();

            // sum_xy = prod_x + prod_y -> CB_TMP_A (re-use)
            cb_wait_front(CB_TMP_A, 1);
            cb_wait_front(CB_TMP_B, 1);
            tile_regs_acquire();
            add_tiles_init(CB_TMP_A, CB_TMP_B);
            add_tiles(CB_TMP_A, CB_TMP_B, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_TMP_A, 1);
            cb_pop_front(CB_TMP_B, 1);
            cb_reserve_back(CB_TMP_A, 1);
            pack_tile(0, CB_TMP_A);
            cb_push_back(CB_TMP_A, 1);
            tile_regs_release();

            // prod_z = mz * r_j2 -> CB_TMP_B (re-use)
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MZ);
            copy_tile(CB_MZ, 0, 0);
            mul_unary_tile(0, r_j2);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_TMP_B, 1);
            pack_tile(0, CB_TMP_B);
            cb_push_back(CB_TMP_B, 1);
            tile_regs_release();

            // mc_j = sum_xy + prod_z -> CB_MC{X,Y,Z}
            cb_wait_front(CB_TMP_A, 1);
            cb_wait_front(CB_TMP_B, 1);
            tile_regs_acquire();
            add_tiles_init(CB_TMP_A, CB_TMP_B);
            add_tiles(CB_TMP_A, CB_TMP_B, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(CB_TMP_A, 1);
            cb_pop_front(CB_TMP_B, 1);
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
