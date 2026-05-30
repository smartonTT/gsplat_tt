// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gsplat_tt gather_visible — residency pass R2 (amendment-003 R2a/R2b).
//
// Consumes the N-indexed device-resident pfwc_* buffers (pfwc_m2x / pfwc_m2y /
// pfwc_depth / pfwc_a / pfwc_b / pfwc_c / pfwc_rx / pfwc_ry, published by
// pfwc_device.cpp) plus the scene colors/opacities (uploaded + registered as
// scene_col_r/g/b and scene_opacities), applies the SAME valid_mask predicate
// as gsplat_cpu::project_finish_with_cov2d_radii (k_near=0.2, min_opacity,
// AABB-vs-image, rx/ry>0, rx/ry<=max_radius), and gathers the M visible
// Gaussians into M-compact device-resident DRAM buffers:
//
//   proj_m_px      (M)    fp32 SoA   <- pfwc_m2x[i]
//   proj_m_py      (M)    fp32 SoA   <- pfwc_m2y[i]
//   proj_m_rx      (M)    fp32 SoA   <- pfwc_rx[i]
//   proj_m_ry      (M)    fp32 SoA   <- pfwc_ry[i]
//   proj_m_a       (M)    fp32 SoA   <- pfwc_a[i]
//   proj_m_b       (M)    fp32 SoA   <- pfwc_b[i]
//   proj_m_c       (M)    fp32 SoA   <- pfwc_c[i]
//   proj_m_depth   (M)    fp32 SoA   <- pfwc_depth[i]
//   proj_m_colors  (M*3)  fp32 AoS   <- scene colors[i]
//   proj_m_opacity (M)    fp32 SoA   <- scene opacities[i]
//   proj_M         (1)    uint32     <- visible count M
//
// Output order is the host finisher's increasing-source-index order so the
// compact index `m` is consistent with the CPU ProjectResult. The covariance
// is kept as a/b/c SoA (NOT expanded to [a,b,b,c]); the returned host
// ProjectResult expands to [a,b,b,c] for the unchanged downstream pipeline.
//
// Two implementations behind GSPLAT_TT_RESIDENT_GATHER=1:
//   * R2a (GSPLAT_TT_GATHER_HOST=1): host gather. Reads the resident pfwc_*
//     back to host, runs project_finish_with_cov2d_radii + color/opacity
//     compaction, and writes the compact results into the proj_m_* buffers
//     via EnqueueWriteMeshBuffer. Zero numerical risk; validates plumbing.
//   * R2b (default when gate on): on-device single-core gather_visible kernel
//     scatters directly into proj_m_*. Host only reads the scalar M back and
//     reassembles the ProjectResult from the M-compact readback.

#pragma once

#include <cstddef>

#include "gsplat_cpu/project.h"
#include "gsplat_cpu/thread_pool.h"

namespace gsplat_tt {

struct GatherCallTimings {
    double upload_ms  = 0.0;   // scene colors/opacities H2D (cache-miss only)
    double kernel_ms  = 0.0;   // device scatter kernel wall time (R2b)
    double readback_ms = 0.0;  // D2H of proj_m_* to build the ProjectResult
    double host_ms    = 0.0;   // host finisher + pack (R2a) or assemble (R2b)
    double verify_ms  = 0.0;   // self-check vs CPU reference (gate-only)
    bool   cache_hit  = false; // scene upload skipped
    bool   host_path  = false; // true if R2a host gather was used
};

// Gather the visible (M-compact) project outputs into device-resident DRAM
// from the resident N-indexed pfwc_* buffers + scene colors/opacities.
//
//   scene_colors    : (N, 3) fp32 row-major host pointer (uploaded + cached).
//   scene_opacities : (N,)   fp32 host pointer (uploaded + cached).
//   N               : Gaussian count.
//   image_height/width, min_opacity, max_radius_param : predicate params,
//                     identical semantics to project_finish_with_cov2d_radii.
//   host_gather     : R2a host path when true, R2b device kernel when false.
//   verify          : when true, read proj_m_* back and assert equality with
//                     the CPU project_finish + compaction reference.
//
// Returns a layout-identical gsplat_cpu::ProjectResult (means_2d, covs_2d
// expanded to [a,b,b,c], depths, radii, colors M*3, opacities M). On device
// failure sets *device_ok=false and returns an empty result.
gsplat_cpu::ProjectResult gather_visible_tt(
    const float* scene_colors,
    const float* scene_opacities,
    std::size_t N,
    int image_height,
    int image_width,
    float min_opacity,
    int max_radius_param,
    bool host_gather,
    bool verify,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    GatherCallTimings* timings = nullptr,
    // PERF: when true (set by the caller iff the FULL on-device resident
    // downstream chain — device tile_assign/sort + resident-pairs + resident
    // device blend — will consume the resident proj_m_* directly over NoC),
    // the gather skips the wasted bulk proj_m_* D2H and returns an M-only
    // ProjectResult (depths sized M + compact opacities for the TA host cull).
    // MUST be false whenever the host path still reads proj.means_2d/covs_2d/
    // colors (e.g. the cpu_cpp_mb reference render's CPU cull_and_blend).
    bool downstream_resident = false);

// R1 fallback helper: read the resident N-indexed pfwc_* buffers back to host
// AoS arrays so the legacy host finisher can run when GSPLAT_TT_RESIDENT_PROJECT
// skipped the in-pfwc D2H but the gather path is off/failed.
//   mean_2d : (N, 2)  depth : (N,)  cov2d : (N, 3) [a,b,c]  radii : (N, 2)
// Returns false if the pfwc_* buffers are not registered.
bool readback_pfwc_resident(
    std::size_t N, float* mean_2d, float* depth, float* cov2d, float* radii);

bool gather_visible_device_ready();
void gather_visible_device_shutdown();

}  // namespace gsplat_tt
