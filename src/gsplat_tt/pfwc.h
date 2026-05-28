// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Public C++ API for the amendment-002 tt-008 pfwc port.
//
// pfwc_tt produces:
//   * mean_2d (N, 2) — fx*tx/tz + cx, fy*ty/tz + cy
//   * depth   (N,)
//   * cov_cam_unique (N, 6) — cc00, cc01, cc02, cc11, cc12, cc22
//
// These three outputs replace the perspective + cov_cam parts of
// gsplat_cpu::project_full_fused — the heaviest 70% of pfwc's per-Gaussian
// inner loop. cov2d / radii / valid_mask remain on the host (downstream of
// this call) and consume the cov_cam_unique we computed.

#pragma once

#include <cstddef>

namespace gsplat_tt {

struct PfwcCallTimings {
    double pack_ms     = 0.0;   // host-side SoA pack of cov3d (cached → 0 after first)
    double upload_ms   = 0.0;   // H2D for cov3d (cached → 0 after first)
    double launch_ms   = 0.0;
    double compute_ms  = 0.0;
    double download_ms = 0.0;
    double unpack_ms   = 0.0;
    bool   cache_hit   = false;
};

// Compute mean_2d, depth, cov_cam_unique for N Gaussians using device-resident
// means_cam (already on device from the prior project kernel call).
//
//   cov3d_unique : (N, 6) row-major, order [c00, c01, c02, c11, c12, c22].
//                  Cached across calls when the pointer/N are unchanged.
//   extrinsics   : 4×4 row-major (only R[3×3] + t[3] used).
//   intrinsics   : 3×3 row-major (only fx, fy, cx, cy used).
//   mean_2d_out  : (N, 2) row-major output.
//   depth_out    : (N,) output.
//   cov_cam_out  : (N, 6) row-major output, same order as cov3d_unique.
//
// Returns total wall-time of the call in ms (compute + DMA), or -1.0 if
// the device path is unavailable (caller must fall back to CPU pfwc).
double pfwc_tt(
    const float* cov3d_unique,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    float* mean_2d_out,
    float* depth_out,
    float* cov_cam_out,
    PfwcCallTimings* timings_out = nullptr);

// True if the kernel is built and the shared MeshDevice is alive. Calling
// pfwc_tt without checking this just returns -1.0 on failure — but ready()
// lets the Python backend avoid the per-frame marshaling work when the path
// is dead.
bool pfwc_device_ready();

void pfwc_device_shutdown();

}  // namespace gsplat_tt
