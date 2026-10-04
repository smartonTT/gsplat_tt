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
#include <cstdint>

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

// Lever 2 (task #99, GSPLAT_TT_SFPU_VIS): parameters of the visibility
// predicate (gather_visible) and the tile rectangle (tile_assign K1). When
// pfwc_tt gets them it runs the PFWC_VIS program, which also reads the resident
// scene opacities ("scene_opacities", uploaded by gather_visible_upload_scene)
// and registers "pfwc_tpg", "pfwc_aabb" (word tiles, vis_tile.h),
// "pfwc_vis_mask" (128 B per tile) and "pfwc_tile_counts" ([visible, pairs]
// per tile, 1 KB pages).
struct PfwcVisParams {
    float k_near = 0.2f;
    float min_opacity = 0.0f;
    float image_width = 0.0f;
    float image_height = 0.0f;
    float max_radius = 0.0f;  // effective (gather_visible_effective_max_radius)
    int tile_size = 32;
    int tiles_x = 1;
    int tiles_y = 1;
    // Lanes whose fl(m + r) / tile_size is within edge_tau of an integer are
    // re-evaluated exactly by the writer (SFPMAD rounding insurance; 0 = off).
    float edge_tau = 1.0f / 4096.0f;
    // Lever B (task #125, GSPLAT_TT_PFWC_FUSE=1): run the fused writer, which
    // writes the compact proj_m_depth / blendrec / offs / aabb segments and the
    // per-core counts table ("pfwc_fuse_counts") instead of the pfwc tiles.
    // Only for the resident chain (gather downstream_resident, no verify).
    bool fuse = false;
    // Lever C (task #140, GSPLAT_TT_PRECULL=1 or 2): the microblock band cull's
    // contribution floor; > 0 shrinks the rectangle to the opacity-aware
    // extent (project_pfwc_compute.cpp step 11.6). 0 = off for this view (band
    // cull disabled), the PFWC_PRECULL program then keeps the 3-sigma radii.
    float precull_floor = 0.0f;
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
// iter-133 fusion: pfwc_tt now also runs the world→camera means transform
// in-kernel (the former standalone project_means_cam program is fused in), so
// it takes `means` directly and no longer depends on a prior project_tt call /
// device-resident means_cam buffers.
//   means : (N, 3) row-major world-space means.
double pfwc_tt(
    const float* means,
    const float* cov3d_unique,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    float* mean_2d_out,
    float* depth_out,
    float* cov2d_out,
    float* radii_out,
    PfwcCallTimings* timings_out = nullptr,
    const PfwcVisParams* vis = nullptr);

// True when the last pfwc_tt call ran the PFWC_VIS program (its word tiles,
// mask and counts are current); gather_visible only takes the SFPU path then.
bool pfwc_ran_vis();

// Lever B: true when the last pfwc_tt call ran the fused writer. Then the
// compact segments and the counts table are current, the pfwc tiles are NOT,
// and gather_visible hands over to tile_assign_fused_k2.
struct PfwcFuseInfo {
    uint32_t num_cores = 0;  // pfwc cores = segments
    uint32_t num_tiles = 0;
    uint32_t tiles_x = 0;
};
bool pfwc_ran_fused(PfwcFuseInfo* info = nullptr);

bool pfwc_device_ready();
void pfwc_device_shutdown();

}  // namespace gsplat_tt
