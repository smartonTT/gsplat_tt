// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gsplat_tt sort (per-tile depth sort) port — amendment-002 tt-003.
//
// Behaviour-preserving drop-in for gsplat_cpu::sort_and_bin. Returns the
// IDENTICAL (sorted_gaussian_ids, tile_ranges) the CPU produces so the
// downstream cull_and_blend / render_blend_tt is unaffected. The returned
// SortResult keeps the CPU output TYPE (int64 ids + int64 ranges).
//
// Two staged paths, gated by env in pybind render_full_py:
//   GSPLAT_TT_DEVICE_SORT>=1  enable this path at all.
//   GSPLAT_TT_SORT_STAGE=0    S0 — run the CPU sort_and_bin on host, but
//                             publish its outputs into DRAM buffers registered
//                             in device_state ("sort_sorted_ids",
//                             "sort_tile_ranges"). Zero PSNR risk; makes the
//                             Stage-3 outputs device-resident.
//   GSPLAT_TT_SORT_STAGE>=1   S1 (default when STAGE unset) — host binning
//                             (Pass1+Pass2, identical to CPU) builds
//                             packed_keys/packed_ids in a page-aligned DRAM
//                             layout; the per-tile STABLE LSD radix sort
//                             (Pass3) runs as a DEVICE kernel; host compacts
//                             the per-tile aligned segments back into the
//                             CPU-contiguous order (Pass4).
//   GSPLAT_TT_SORT_VERIFY=1   run gsplat_cpu::sort_and_bin in parallel and
//                             assert byte-identical sorted_gaussian_ids +
//                             tile_ranges (prints a SORT line; aborts on
//                             mismatch).

#pragma once

#include <cstddef>
#include <cstdint>

#include "gsplat_cpu/sort.h"

namespace gsplat_cpu {
class ThreadPool;
}

namespace gsplat_tt {

// Per-call sub-timing breakdown (ms). Separates the device radix kernel time
// from the host binning / compaction / DMA bridges.
struct SortCallTimings {
    double bin_ms = 0.0;       // host Pass1+Pass2 binning into aligned layout
    double upload_ms = 0.0;    // H2D of packed keys/ids
    double kernel_ms = 0.0;    // device per-tile radix kernel
    double d2h_ms = 0.0;       // device->host readback of sorted ids
    double compact_ms = 0.0;   // host Pass4 aligned->contiguous compaction
    double publish_ms = 0.0;   // H2D of the resident contiguous outputs
    double total_ms = 0.0;     // wall clock of the whole call
    int stage = -1;            // which staged path ran (0 = S0, 1 = S1)
};

// Device sort. Same signature shape as gsplat_cpu::sort_and_bin. On success
// sets *device_ok = true and returns a SortResult identical to the CPU's. On
// any device failure / unsupported state sets *device_ok = false and returns
// an empty result so the caller falls back to gsplat_cpu::sort_and_bin.
gsplat_cpu::SortResult sort_and_bin_tt(
    const int64_t* gaussian_ids,  // P
    const int64_t* tile_ids,      // P
    const float* depths,          // M (indexed by gaussian_ids[i])
    std::size_t P,
    std::size_t M,
    int tiles_x,
    int tiles_y,
    gsplat_cpu::ThreadPool* pool,
    bool* device_ok,
    SortCallTimings* timings = nullptr);

// Lazily initializes the device sort context (programs + CBs). Returns true
// if the device path is operational.
bool sort_device_ready();

// Idempotent shutdown of the sort device context. Does NOT close the shared
// MeshDevice (device_state owns that).
void sort_device_shutdown();

}  // namespace gsplat_tt
