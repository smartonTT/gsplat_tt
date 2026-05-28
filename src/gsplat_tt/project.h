// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gsplat_tt project port — amendment-002 tt-005.
//
// Initial bounded hotspot port: device transform_means_cam (world→camera
// matrix-vec). This is the matrix-vec stage of project that runs once per
// frame, has a clean per-Gaussian data-parallel structure, and maps directly
// to SFPU vectorization (32 Gaussians per SFPU vector pass, 1024 per tile).
//
// On linux without Accelerate, the CPU path is a single-threaded scalar loop
// (see src/gsplat_cpu/project.cpp:528-535) that costs ~30-50 ms / frame at
// N=6.13M Gaussians. Device target: ~5-10 ms with SFPU vectorization + 140
// Tensix cores.
//
// PSNR-correctness: math is identical to CPU; only execution location moves.
// Bit-identical fp32 output up to scheduler-induced FMA-ordering rounding
// (which is negligible — this transform is a pure mat-vec, no big sums).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_tt {

// Device transform_means_cam: means_cam[i] = R(extr) @ means[i] + t(extr).
//
//   means        N * 3 fp32, world-space means (row-major)
//   extrinsics   16 fp32, row-major 4x4 view matrix. R is rows 0..2 col 0..2;
//                t is column 3 (i.e. extrinsics[3], extrinsics[7], extrinsics[11]).
//   means_cam_out N * 3 fp32 output (caller allocates).
//
// Returns the wall-clock kernel execution time in ms (device kernel only —
// excludes host-side pack/upload/download overhead).
//
// Falls back to a CPU implementation if device init fails, returning -1.0 ms
// to signal fallback to the caller.
double transform_means_cam_tt(
    const float* means,
    const float* extrinsics,
    std::size_t N,
    float* means_cam_out);

// Returns true if the TT device project path is initialized and operational.
bool project_device_ready();

// Optional explicit shutdown for the project device context. Idempotent.
void project_device_shutdown();

}  // namespace gsplat_tt
