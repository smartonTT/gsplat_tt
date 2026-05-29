// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microblock-major (4x8) alpha-blend READER — amendment-003 step 3.
//
// Feeds alpha_blend_compute_mb.cpp. For each screen tile assigned to this core
// (LPT slice, same as the legacy reader) it pushes, in order:
//
//   CB_XRAMP    (0): one fp32 32x32 tile, tile-local x = col + 0.5  (shared)
//   CB_YRAMP    (1): one fp32 32x32 tile, tile-local y = row + 0.5  (shared)
//   CB_MB_COUNTS(3): 32 uint32 (one per microblock) gaussian counts
//   CB_MB_COEFF (2): one 48B coeff row per (microblock, gaussian), microblock-
//                    major, depth-sorted within a microblock. Pre-gathered host
//                    side (de-referenced through mb_stream) so the reader streams
//                    the tile's slice linearly.
//
// The host builds a per-tile microblock-major coeff stream + per-tile row
// offsets (coeff_off[t], coeff_off[t+1]) + per-tile 32-count pages. xramp/yramp
// are a single shared tile each (identical for every screen tile).
//
// RUNTIME ARGS
//   0: counts_addr        DRAM base of per-tile count pages (128B = 32 uint32)
//   1: coeff_stream_addr  DRAM base of mb-major coeff rows (48B each)
//   2: coeff_off_addr     DRAM base of per-tile row offsets (uint32)
//   3: xramp_addr         DRAM base of the shared xramp tile (one 4KB fp32 tile)
//   4: yramp_addr         DRAM base of the shared yramp tile
//   5: tile_ids_addr      DRAM base of concatenated tile-id list (uint32 each)
//   6: tile_ids_start     this core's element offset into that list
//   7: tile_ids_count     number of tile IDs this core handles
//   8: num_tiles          total screen tiles (unused; kept for parity)
//
// COMPILE-TIME ARGS: 6 TensorAccessorArgs: counts, coeff_stream, coeff_off,
// xramp, yramp, tile_ids. All DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(MB_DEBUG_PAGEDBG)
#include "api/debug/dprint.h"
#endif

constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
constexpr uint32_t NUM_MB = 32;
constexpr uint32_t COUNTS_PAGE_BYTES = 128;  // 32 uint32
// 10 real fp32 lanes padded to a 64B DRAM-aligned page (see blend_device.cpp:
// an unaligned 48B page corrupts interleaved per-bank addressing).
constexpr uint32_t COEFF_ROW_BYTES = 64;
// Gaussian rows are streamed in large interleaved pages of COEFF_PAGE_ROWS rows
// (must match COEFF_PAGE_ROWS in blend_device.cpp / CHUNK_ROWS in the compute
// kernel). One noc_async_read_tile per page == one big DRAM transaction.
constexpr uint32_t COEFF_PAGE_ROWS  = 64;
constexpr uint32_t COEFF_PAGE_BYTES = COEFF_PAGE_ROWS * COEFF_ROW_BYTES;  // 4096
// Pipeline depth for coeff page reads. The blend is DRAM-LATENCY bound: a
// barrier after every page read serializes one outstanding read at a time
// (~720ms even after coalescing to 4KB pages). Issuing COEFF_READ_BATCH reads
// before a single barrier lets the DRAM banks service them in parallel, hiding
// latency. COEFF_CB_DEPTH MUST match CB_MB_COEFF depth in blend_device.cpp.
constexpr uint32_t COEFF_CB_DEPTH   = 16;
constexpr uint32_t COEFF_READ_BATCH = 8;
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;  // fp32 32x32 tile

constexpr uint32_t CB_XRAMP      = 0;
constexpr uint32_t CB_YRAMP      = 1;
constexpr uint32_t CB_MB_COEFF   = 2;
constexpr uint32_t CB_MB_COUNTS  = 3;
// Reader-private scratch (64B). NEVER pushed/popped, so compute never sees it
// and its write pointer is stable. Used for the small tile-id + page-offset
// DRAM reads. MUST NOT reuse CB_MB_COUNTS for this: when the reader runs ahead
// of compute (big-page makes the reader fast), writing scratch into a COUNTS
// slot clobbers the next tile's gaussian count before compute reads it, so
// compute reads a page offset as num_g and deadlocks on cb_wait_front.
constexpr uint32_t CB_RD_SCRATCH = 4;

void kernel_main() {
    uint32_t counts_addr       = get_arg_val<uint32_t>(0);
    uint32_t coeff_stream_addr = get_arg_val<uint32_t>(1);
    uint32_t coeff_off_addr    = get_arg_val<uint32_t>(2);
    uint32_t xramp_addr        = get_arg_val<uint32_t>(3);
    uint32_t yramp_addr        = get_arg_val<uint32_t>(4);
    uint32_t tile_ids_addr     = get_arg_val<uint32_t>(5);
    uint32_t tile_ids_start    = get_arg_val<uint32_t>(6);
    uint32_t tile_ids_count    = get_arg_val<uint32_t>(7);

    constexpr auto counts_args  = TensorAccessorArgs<0>();
    constexpr auto coeff_args    = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto coeff_off_args = TensorAccessorArgs<coeff_args.next_compile_time_args_offset()>();
    constexpr auto xramp_args    = TensorAccessorArgs<coeff_off_args.next_compile_time_args_offset()>();
    constexpr auto yramp_args    = TensorAccessorArgs<xramp_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args = TensorAccessorArgs<yramp_args.next_compile_time_args_offset()>();

    const auto counts_acc    = TensorAccessor(counts_args,    counts_addr,       COUNTS_PAGE_BYTES);
    const auto coeff_acc     = TensorAccessor(coeff_args,     coeff_stream_addr, COEFF_PAGE_BYTES);
    const auto coeff_off_acc = TensorAccessor(coeff_off_args, coeff_off_addr,    /*page=*/4);
    const auto xramp_acc     = TensorAccessor(xramp_args,     xramp_addr,        RAMP_TILE_BYTES);
    const auto yramp_acc     = TensorAccessor(yramp_args,     yramp_addr,        RAMP_TILE_BYTES);
    const auto tile_ids_acc  = TensorAccessor(tile_ids_args,  tile_ids_addr,     64);

    if (tile_ids_count == 0) {
        return;
    }

    // Cache this core's tile-ID slice in L1 (reader-private scratch CB).
    const uint32_t scratch_addr = get_write_ptr(CB_RD_SCRATCH);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t ids_per_page = 64 / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            uint64_t page_noc = get_noc_addr(page_idx, tile_ids_acc);
            noc_async_read(page_noc, scratch_addr, 64);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            }
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    // Logical write slot of CB_MB_COEFF (this core is its sole producer). Used to
    // cap each read batch so it never wraps past the CB's physical end.
    uint32_t coeff_back_slot = 0;

    for (uint32_t t = 0; t < tile_ids_count; t++) {
        uint32_t tile_id = tile_ids[t];

        // (1) Shared coordinate ramps (page 0 of each ramp buffer).
        cb_reserve_back(CB_XRAMP, 1);
        noc_async_read_tile(0, xramp_acc, get_write_ptr(CB_XRAMP));
        cb_reserve_back(CB_YRAMP, 1);
        noc_async_read_tile(0, yramp_acc, get_write_ptr(CB_YRAMP));
        noc_async_read_barrier();
        cb_push_back(CB_XRAMP, 1);
        cb_push_back(CB_YRAMP, 1);

        // (2) Per-microblock counts page for this tile.
        cb_reserve_back(CB_MB_COUNTS, 1);
        uint32_t counts_wp = get_write_ptr(CB_MB_COUNTS);
        noc_async_read_tile(tile_id, counts_acc, counts_wp);
        noc_async_read_barrier();
        cb_push_back(CB_MB_COUNTS, 1);

        // (3) Coeff PAGE slice [page_off[t], page_off[t+1]). Each page holds up to
        // COEFF_PAGE_ROWS gaussian rows, packed contiguously host-side.
        uint32_t page_start, page_end;
        {
            uint64_t off_noc = get_noc_addr(tile_id, coeff_off_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
            page_start = scratch_ptr[0];
            off_noc = get_noc_addr(tile_id + 1, coeff_off_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
            page_end = scratch_ptr[0];
        }
#if defined(MB_DEBUG_PAGEDBG)
        DPRINT << "R" << t << "t" << tile_id << "p" << page_start << "-" << page_end << ENDL();
#endif

        // Stream the tile's pages in batches: issue up to COEFF_READ_BATCH page
        // reads, then ONE barrier, so the DRAM banks service them in parallel
        // (latency hiding). Each batch is capped to not wrap past the CB end, so
        // the contiguous write region [wp, wp + n*page) stays in-bounds.
        uint32_t p = page_start;
        while (p < page_end) {
            uint32_t n = page_end - p;
            if (n > COEFF_READ_BATCH) n = COEFF_READ_BATCH;
            const uint32_t to_end = COEFF_CB_DEPTH - coeff_back_slot;
            if (n > to_end) n = to_end;
            cb_reserve_back(CB_MB_COEFF, n);
            const uint32_t wp = get_write_ptr(CB_MB_COEFF);
            for (uint32_t i = 0; i < n; i++) {
                noc_async_read_tile(p + i, coeff_acc, wp + i * COEFF_PAGE_BYTES);
            }
            noc_async_read_barrier();
            cb_push_back(CB_MB_COEFF, n);
            p += n;
            coeff_back_slot += n;
            if (coeff_back_slot >= COEFF_CB_DEPTH) coeff_back_slot -= COEFF_CB_DEPTH;
        }
    }
}
