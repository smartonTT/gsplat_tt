// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gsplat_tt tile_assign (binning) port — amendment-002 tt-006.
//
// Moves the host CPU tile_assign stage (src/gsplat_cpu/tile_assign.cpp,
// ~87 ms/frame on the hero scene) onto the Tenstorrent device. The device
// path is a behaviour-preserving drop-in for gsplat_cpu::tile_assign: it
// returns the IDENTICAL (gaussian_id, tile_id) pair SET (and gaussian-major
// pair order) so the downstream sort_and_bin + blend are unaffected.
//
// Algorithm mirrors the CPU phases (split over GAUSSIANS, never tiles):
//   K1  per-gaussian AABB -> tile_min_x/min_y/width + tiles_per_gaussian
//   H1  host exclusive prefix-sum of tiles_per_gaussian -> offs[M+1], P
//   K2  pair-centric scatter -> gaussian_ids[P], tile_ids[P]
//   K3  per-gaussian m2_thresh / opacity-floor precompute (host bridge)
//   K4  per-pair constrained-min Mahalanobis cull -> keep_mask[P]
//   H2  host chunked compaction -> kept_gids/kept_tids (P')
//
// Host bridges retained for now (perf comes later via residency): the prefix
// sum (H1), the m2_thresh precompute (tiny, M logs), and the final
// compaction + D2H of the pairs.

#pragma once

#include <cstddef>

#include "gsplat_cpu/tile_assign.h"

namespace gsplat_tt {

// Per-call sub-timing breakdown (ms). Helps separate device kernel time from
// the host prefix-sum / compaction / DMA bridges.
struct TileAssignCallTimings {
    double k1_ms = 0.0;        // bbox kernel (device)
    double d2h_tpg_ms = 0.0;   // D2H of tiles_per_gaussian (host bridge)
    double prefix_ms = 0.0;    // host exclusive prefix-sum (H1)
    double h2d_offs_ms = 0.0;  // H2D of offs[] (host bridge)
    double k2_ms = 0.0;        // scatter kernel (device)
    double k3_ms = 0.0;        // m2_thresh/cov precompute + H2D (host bridge)
    double k3_compute_ms = 0.0;// m2_thresh/opacok host loop (CPU)
    double k3_h2d_ms = 0.0;    // H2D of m2thr/opacok (+cov when !resident)
    double k4_ms = 0.0;        // per-pair cull kernel (device)
    double cull_ms = 0.0;      // K3 + K4 combined (back-compat)
    double publish_ms = 0.0;   // resident-pairs publish (write P + Finish)
    double compact_ms = 0.0;   // host compaction (H2)
    double d2h_ms = 0.0;       // device->host readbacks (pairs/keep)
    double total_ms = 0.0;     // wall clock of the whole call
};

// Device tile_assign. Same signature shape as gsplat_cpu::tile_assign. When
// covs_2d && opacities are non-null the per-pair Mahalanobis cull runs (full
// path); otherwise only the AABB overlap is produced (cull_disabled path).
//
// On success sets *device_ok = true and returns the result. On any device
// failure / unsupported state sets *device_ok = false and returns an empty
// result so the caller can transparently fall back to the CPU path.
gsplat_cpu::TileAssignResult tile_assign_tt(
    const float* means_2d,   // M * 2
    const float* radii,      // M * 2 (rx, ry per row)
    std::size_t M,
    int image_height,
    int image_width,
    int tile_size,
    const float* covs_2d,    // nullable; M * 4 (a, b, b, c)
    const float* opacities,  // nullable; M
    float contrib_floor,
    bool* device_ok,
    TileAssignCallTimings* timings = nullptr);

// Lazily initializes the device tile_assign context. Returns true if the
// device path is operational.
bool tile_assign_device_ready();

// Idempotent shutdown of the tile_assign device context. Does NOT close the
// shared MeshDevice (device_state owns that).
void tile_assign_device_shutdown();

}  // namespace gsplat_tt
