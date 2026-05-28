// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Public C++ API for the amendment-002 tt-008 pfwc port.
//
// tt-008c (this revision): the device kernel now also computes cov2d + radii,
// so pfwc_tt produces the full set of per-Gaussian outputs the downstream
// stages need:
//   * mean_2d (N, 2) — fx*tx/tz + cx, fy*ty/tz + cy
//   * depth   (N,)
//   * cov2d   (N, 3) — a, b, c (canonical cov2d unique entries; the host
//                      finisher will expand to [a, b, b, c] before tile_assign)
//   * radii   (N, 2) — ceil(k · sqrt(max(a, 0))), ceil(k · sqrt(max(c, 0)))
//                      with k = 3.0 (matches project_full_fused k_cap=3.0 default).
//
// cov_cam_unique is computed in scratch CBs and not exposed — host no longer
// needs it now that cov2d/radii are device-resident.

#pragma once

#include <cstddef>

namespace gsplat_tt {

struct PfwcCallTimings {
    double pack_ms     = 0.0;
    double upload_ms   = 0.0;
    double launch_ms   = 0.0;
    double compute_ms  = 0.0;
    double download_ms = 0.0;
    double unpack_ms   = 0.0;
    bool   cache_hit   = false;
};

// Compute mean_2d, depth, cov2d, radii for N Gaussians using device-resident
// means_cam (already on device from the prior project kernel call).
//
//   cov3d_unique : (N, 6) row-major, order [c00, c01, c02, c11, c12, c22].
//   extrinsics   : 4×4 row-major (only R[3×3] + t[3] used).
//   intrinsics   : 3×3 row-major (only fx, fy, cx, cy used).
//   mean_2d_out  : (N, 2) row-major output. Set to nullptr to skip D2H+unpack
//                  of this stream (output stays purely device-resident,
//                  registered under "pfwc_m2x"/"pfwc_m2y" in device_state).
//   depth_out    : (N,) output. nullptr → skip D2H, registered "pfwc_depth".
//   cov2d_out    : (N, 3) row-major output, order [a, b, c]. nullptr → skip
//                  D2H, registered "pfwc_a", "pfwc_b", "pfwc_c".
//   radii_out    : (N, 2) row-major output, order [rx, ry]. nullptr → skip
//                  D2H, registered "pfwc_rx", "pfwc_ry".
//
// Returns total wall-time of the call in ms, or -1.0 on failure.
double pfwc_tt(
    const float* cov3d_unique,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    float* mean_2d_out,
    float* depth_out,
    float* cov2d_out,
    float* radii_out,
    PfwcCallTimings* timings_out = nullptr);

bool pfwc_device_ready();
void pfwc_device_shutdown();

}  // namespace gsplat_tt
