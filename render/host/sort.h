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
#include <vector>

#include "gsplat_cpu/sort.h"

namespace gsplat_cpu {
class ThreadPool;
}

namespace gsplat_tt {

// Per-call sub-timing breakdown (ms). Separates the device radix kernel time
// from the host binning / compaction / DMA bridges.
// Optional in-sort resident blend (GSPLAT_TT_SORT_BLEND_PIPE). When set and the
// device sort succeeds, cull+blend run before returning to render_full_py so
// frame-1 does not pay a separate blend cold-start / host gap after sort.
struct SortBlendContinuation {
    uint8_t* image_out = nullptr;  // final u8 RGB image, H*W*3
    int image_height = 0;
    int image_width = 0;
    float mb_contrib_floor = 0.0f;
    // Saturation epsilon forwarded to the blend compute kernel (viewer
    // "Transmittance threshold" slider). 0 => kernel keeps its compile-time
    // default (iter-107 baseline). NOTE: this struct is ODR-shared with
    // src/gsplat_tt/sort.h — keep the two layouts byte-identical.
    float transmittance_threshold = 0.0f;
    bool cull_disabled = false;
    bool* blend_ok = nullptr;
    bool invoked = false;
    // De-lumped device-stage timings (ms), filled by the continuation so
    // render_full_py can report SORT / CULL / BLEND separately instead of
    // lumping cull+blend under the sort host-timer. Measurement-only.
    double cull_ms = 0.0;
    double blend_ms = 0.0;
};

// On the resident-pairs path (the only one render_clean runs) every "bin" step
// is a DEVICE kernel except the per-tile layout: bin_ms = bin_count_ms +
// bin_hist_d2h_ms + bin_layout_ms + bin_emit_ms. The leaf fields below are
// disjoint wall-clock spans of sort_and_bin_tt (the fused cull/blend
// continuation excluded); render.cpp books them as stage_sort_* buckets.
struct SortCallTimings {
    double bin_ms = 0.0;       // count + hist D2H + host layout + emit (aggregate)
    double upload_ms = 0.0;    // H2D enqueue of the layout outputs + metadata
    double kernel_ms = 0.0;    // device per-tile radix kernel (enqueue only; drained in publish)
    double d2h_ms = 0.0;       // device->host readback of sorted ids
    double compact_ms = 0.0;   // host Pass4 aligned->contiguous compaction
    double publish_ms = 0.0;   // publish_host_ms + publish_wait_ms
    double materialize_ms = 0.0;  // post-radix PACK2 subchunk materialize (step A)
    double total_ms = 0.0;     // wall clock of the whole call
    int stage = -1;            // which staged path ran (0 = S0, 1 = S1)
    // Leaf split (task #18).
    double pread_ms = 0.0;         // blocking D2H of tile_assign's P control page
    double bin_count_ms = 0.0;     // Pass A device histogram kernel: launch + Finish
    double bin_hist_d2h_ms = 0.0;  // blocking D2H of the per-(core,tile) histogram
    double bin_layout_ms = 0.0;    // host_bin_layout_from_hist (host CPU only)
    double bin_emit_ms = 0.0;      // Pass B device scatter/emit kernel: launch + Finish
    double publish_host_ms = 0.0;  // publish: host prep + enqueues before the drain
    double publish_wait_ms = 0.0;  // publish: Finish draining radix+publish+directory
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
    SortCallTimings* timings = nullptr,
    // When true, D2H sort_sorted_ids into SortResult even if RESIDENT_BLEND
    // skips it (required for CPU blend_mode=0 in the same process as TT env).
    bool need_host_sorted_ids = false,
    SortBlendContinuation* sort_blend = nullptr);

// Task #270: after a failed sort_and_bin_tt, the largest (padded) record count
// of a tile over the bucket capacity, else 0; and that capacity (32768, or
// GSPLAT_TT_TEST_TILE_CAP). render_view retries such a view at a coarser floor.
uint32_t sort_last_tile_overflow();
uint32_t sort_tile_capacity();

// Task #198 (GSPLAT_TT_SORT_OL_EARLY): enqueue the one-launch sort on the
// device's CQ0 right behind the fold K2, before the host knows P. The kernel
// reads P from ta_pairs_P and takes each mover's K2 page range from the running
// mover-speed sums acc (2 * cores + 1, as the K2 got them), so the fold holds
// by construction. The pair buffers are passed in: tile_assign registers them
// only later. Returns false, with nothing enqueued, when the one-launch config
// does not hold; the sort then runs at its usual place.
bool sort_onelaunch_enqueue_early(uint32_t num_tiles, uint32_t tiles_x, uint32_t row_pages,
                                  uint32_t rows_addr, uint32_t gids_addr, uint32_t tids_addr,
                                  uint32_t keep_addr, uint32_t pairs_P_addr,
                                  const std::vector<uint64_t>& acc);

// Lazily initializes the device sort context (programs + CBs). Returns true
// if the device path is operational.
bool sort_device_ready();

// Idempotent shutdown of the sort device context. Does NOT close the shared
// MeshDevice (device_state owns that).
void sort_device_shutdown();

}  // namespace gsplat_tt
