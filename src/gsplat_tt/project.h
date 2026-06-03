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

// Per-call sub-timing breakdown for transform_means_cam_tt. All times in ms.
// Cache-hit path skips pack/upload — those fields will be 0.0 in that case.
// tt-005c instrumentation: helps tease apart kernel vs DMA vs host work.
struct ProjectCallTimings {
    double pack_ms = 0.0;      // host SoA pack (means -> 3 column vectors)
    double upload_ms = 0.0;    // 3 x EnqueueWriteMeshBuffer (means H2D)
    double launch_ms = 0.0;    // SetRuntimeArgs + EnqueueMeshWorkload
    double compute_ms = 0.0;   // Finish() — actual device kernel wall time
    double download_ms = 0.0;  // 3 x EnqueueReadMeshBuffer (means_cam D2H)
    double unpack_ms = 0.0;    // SoA -> AoS host repack into means_cam_out
    bool cache_hit = false;    // true if pack+upload skipped this call
};

// Device transform_means_cam: means_cam[i] = R(extr) @ means[i] + t(extr).
//
//   means        N * 3 fp32, world-space means (row-major)
//   extrinsics   16 fp32, row-major 4x4 view matrix. R is rows 0..2 col 0..2;
//                t is column 3 (i.e. extrinsics[3], extrinsics[7], extrinsics[11]).
//   means_cam_out N * 3 fp32 output (caller allocates).
//   timings_out  optional, filled with per-phase timings if non-null.
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
    float* means_cam_out,
    ProjectCallTimings* timings_out = nullptr);

// tt-008b: skip-download variant. means_cam_out is allowed to be nullptr;
// then the D2H readback of means_cam_{x,y,z} is omitted entirely. The
// device buffers are still registered with device_state (means_cam_x/y/z)
// so downstream device-resident stages (pfwc_tt) can read them via NoC.
// Saves the 25-30 ms/view wasted on the means_cam download whenever the
// downstream stage is on-device too.
double transform_means_cam_tt_no_download(
    const float* means,
    const float* extrinsics,
    std::size_t N,
    ProjectCallTimings* timings_out = nullptr);

// Returns true if the TT device project path is initialized and operational.
bool project_device_ready();

// Optional explicit shutdown for the project device context. Idempotent.
void project_device_shutdown();

}  // namespace gsplat_tt
