// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// Alpha-blend READER kernel (NCRISC, NoC1; see DataMovementProcessor::RISCV_1
// in alpha_blend.cpp). The host launches reader on RISCV_1 / NoC1 and writer
// on RISCV_0 / NoC0 so reads and writes use opposite NoCs.
//
// ROLE
// ----
// Streams the four input data sources from DRAM into the four input CBs that
// feed the compute kernel:
//
//   CB_PX        (0): one bf16 32x32 tile per screen tile (per-pixel x coord)
//   CB_PY        (1): one bf16 32x32 tile per screen tile (per-pixel y coord)
//   CB_SCALARS   (2): one 64-byte fp32 pack per Gaussian (mean/cov/color/opacity)
//   CB_TILE_META (3): one uint32 (g_count) per screen tile
//
// The reader runs on every active core in lock-step with the compute kernel:
// for each screen tile assigned to this core, it pushes the four inputs in
// the order shown, and compute consumes them one tile at a time.
//
// LPT TILE ASSIGNMENT
// -------------------
// This core does NOT process a contiguous tile range. The host runs LPT
// (Longest Processing Time first) load balancing on the per-tile Gaussian
// counts and writes a per-core slice of (possibly non-contiguous) tile IDs
// into a shared DRAM buffer. We read this core's slice into L1 once at
// startup (capped at MAX_TILE_IDS_PER_CORE entries) and index by element
// in the per-tile loop. With non-contiguous IDs we can't carry the previous
// tile's `g_end` forward to the next iteration, so each tile pays an extra
// 2-element offsets[] read — small relative to the per-Gaussian stream.
//
// PER-TILE WORK
// -------------
//   1. fetch tile_offsets[id] and tile_offsets[id+1] → (g_start, g_end)
//   2. push g_count = g_end - g_start as one uint32 to CB_TILE_META
//   3. push px and py tiles for this screen tile to CB_PX, CB_PY
//   4. push g_count scalar packs (one 64-byte page each) to CB_SCALARS
//
// RUNTIME ARGS
//   0: packs_addr         DRAM base of scalar packs buffer (64B page each)
//   1: tile_offsets_addr  DRAM base of tile_offsets[] (4B per uint32)
//   2: px_addr            DRAM base of combined px+py tiles (2KB per bf16 32x32);
//                         tile IDs [0, num_tiles) are px, [num_tiles, 2N) are py.
//   3: py_addr            DRAM base of py tiles (iter-079: same as px_addr —
//                         px and py share one buffer; kept for accessor symmetry)
//   4: tile_ids_addr      DRAM base of concatenated tile-id list (uint32 each)
//   5: tile_ids_start     this core's element offset into that list
//   6: tile_ids_count     number of tile IDs this core handles
//   7: num_tiles_total    iter-079: total tile count, used as page-index
//                         offset to address py tiles within the combined buffer.
//
// COMPILE-TIME ARGS: 5 TensorAccessorArgs in order: packs, tile_offsets,
// px, py, tile_ids. All DRAM-interleaved. iter-079: px and py accessors
// point at the same combined buffer (different runtime base addrs are the
// same value), so layout & bank interleave are identical.

// Max per-core tile IDs we cache in L1. Sized to handle a 4K render
// (120x68 = 8160 tiles, ~128/core after round-robin balancing). At 1080p
// (60x34 = 2040 tiles) average is ~32/core, leaving plenty of headroom.
// Cost: 1024 bytes per core (still trivial against the 1.5 MB total L1).
constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;

void kernel_main() {
    // iter-094: same-across-cores values (DRAM buffer addresses + num_tiles)
    // moved to common runtime args; per-core args slimmed to (start, count).
    // Host now writes 1 SetCommonRuntimeArgs + 64 small SetRuntimeArgs per
    // frame instead of 64 large SetRuntimeArgs.
    uint32_t packs_addr        = get_common_arg_val<uint32_t>(0);
    uint32_t tile_offsets_addr = get_common_arg_val<uint32_t>(1);
    uint32_t px_addr           = get_common_arg_val<uint32_t>(2);
    uint32_t py_addr           = get_common_arg_val<uint32_t>(3);
    uint32_t tile_ids_addr     = get_common_arg_val<uint32_t>(4);
    uint32_t num_tiles_total   = get_common_arg_val<uint32_t>(5);  // iter-079: py offset
    uint32_t tile_ids_start    = get_arg_val<uint32_t>(0);
    uint32_t tile_ids_count    = get_arg_val<uint32_t>(1);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes        = get_tile_size(CB_PX);       // 2048 (32x32 bf16)
    // NOTE: do NOT use get_tile_size(CB_SCALARS) here. CB_SCALARS is configured
    // as DataFormat::Float32 (so the SFPU sees fp32), but with a sub-tile page
    // size of 64 bytes (9 fp32 scalar pack). get_tile_size returns the format
    // tile size (4096 for fp32 32x32), not the configured page size, so using
    // it as the NoC transaction size would overflow the CB by 3840 bytes per
    // Gaussian and trample adjacent L1 (manifests as a multi-tile hang once
    // the corrupted L1 overlaps a CB descriptor).
    constexpr uint32_t pack_bytes_padded = 64;  // matches host SCALAR_PACK_PAGE_BYTES
    constexpr uint32_t tile_ids_page_bytes = 64;  // matches host TILE_IDS_PAGE_BYTES

    constexpr auto packs_args     = TensorAccessorArgs<0>();
    constexpr auto offsets_args   = TensorAccessorArgs<packs_args.next_compile_time_args_offset()>();
    constexpr auto px_args        = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto py_args        = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args  = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();

    const auto packs_acc    = TensorAccessor(packs_args,    packs_addr,        pack_bytes_padded);
    const auto offsets_acc  = TensorAccessor(offsets_args,  tile_offsets_addr, /*page_size=*/4);
    const auto px_acc       = TensorAccessor(px_args,       px_addr,           tile_bytes);
    const auto py_acc       = TensorAccessor(py_args,       py_addr,           tile_bytes);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr,     tile_ids_page_bytes);

    // No-work cores (LPT may leave some cores empty when num_tiles < num_cores).
    if (tile_ids_count == 0) {
        return;
    }

    // Per-tile offset scratch + per-core tile-ID cache. Both live in a small
    // L1 region grabbed from the cb_tile_meta CB write pointer. Layout:
    //   bytes [0, 4)               -> offset scratch slot (overwritten per push)
    //   bytes [4, 4 + 4*count)     -> cached per-core tile-ID list
    //
    // We grab the L1 region from cb_tile_meta because we won't push to it
    // until after all the prefetches are complete. The first per-tile iter
    // refreshes scratch_addr (cb_tile_meta wraps on push_back), so we keep
    // the cache valid by reading it into a local stack array before the loop.
    uint32_t scratch_addr = get_write_ptr(CB_TILE_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    // Read this core's tile-ID slice into L1 with one NoC transaction per
    // page. The slice may straddle multiple 64-byte pages; we loop over
    // pages and copy into a local buffer.
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        // First page that contains tile_ids_start; offset within that page.
        const uint32_t ids_per_page = tile_ids_page_bytes / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        // L1 scratch slot for one page (64B), reuse slot at scratch_addr.
        while (remaining > 0) {
            uint64_t page_noc = get_noc_addr(page_idx, tile_ids_acc);
            noc_async_read(page_noc, scratch_addr, tile_ids_page_bytes);
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

    // -------------------------------------------------------------------
    // Main per-tile loop. For each screen tile assigned to this core:
    //   1) read its [g_start, g_end) range from offsets[]
    //   2) push g_count to CB_TILE_META
    //   3) push the px and py tiles
    //   4) push g_count Gaussian scalar packs to CB_SCALARS
    // -------------------------------------------------------------------
    for (uint32_t t = 0; t < tile_ids_count; t++) {
        // Per-tile profiling zone. No-op in non-profile builds.
        DeviceZoneScopedN("Z_R_tile");

        uint32_t tile_id = tile_ids[t];

        // (1) Read tile_offsets[tile_id] and tile_offsets[tile_id+1] from
        // DRAM via two single-uint32 noc_async_reads. Two reads per tile
        // (vs one carry-forward) is the cost of LPT's non-contiguous
        // tile-id assignment; trivial relative to the per-Gaussian stream.
        {
            uint64_t off_noc = get_noc_addr(tile_id, offsets_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
        }
        uint32_t g_start = scratch_ptr[0];

        {
            uint64_t off_noc = get_noc_addr(tile_id + 1, offsets_acc);
            noc_async_read(off_noc, scratch_addr, 4);
            noc_async_read_barrier();
        }
        uint32_t g_end   = scratch_ptr[0];
        uint32_t g_count = g_end - g_start;

        // (2) Push g_count into CB_TILE_META as a single uint32. Writes
        // directly into the meta CB's write pointer (the scratch slot we
        // just used is safe to overwrite — we no longer need it).
        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        // (3) Push px and py tiles. iter-079: px and py share one DRAM
        // buffer; tile IDs [0, N) are px, [N, 2N) are py.
        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id + num_tiles_total, py_acc, get_write_ptr(CB_PY));
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // (4) Stream Gaussian scalar packs for this tile. One 64-byte
        // page per Gaussian; compute pops one per inner-loop iteration.
        // CB_SCALARS depth (4) lets the reader prefetch ahead of compute.
        //
        // iter-089: wrap-aware K=2 burst. Coalesce 2 noc_async_read_tile
        // calls per barrier when the CB has 2 contiguous slots free at
        // the current write pointer. iter-087 (K=4, no wrap check) hung
        // on CB wrap-straddle; we cap `batch` to pages_until_wrap so the
        // burst always lands in contiguous L1.
        // iter-101 (retry of iter-099/100 after ARC wedge + ird reboot):
        // bump K=2→K=3. CB_SCALARS depth=4, K=3 leaves 1 slot for compute
        // consumption between bursts; halves barrier overhead in steady
        // state. Wrap-aware cap unchanged.
        constexpr uint32_t K = 3;
        const auto& cb_scal = get_local_cb_interface(CB_SCALARS);
        uint32_t g = 0;
        while (g < g_count) {
            // Accumulator zone for per-Gaussian scalar fetch (SumN1 slot 0).
            // Reports total cycles spent in the per-Gaussian DRAM→L1 stream
            // for this RISC, which is the dominant work on the reader.
            DeviceZoneScopedSumN1("Z_R_g");

            uint32_t remaining = g_count - g;
            uint32_t batch = remaining < K ? remaining : K;
            uint32_t wptr  = get_write_ptr(CB_SCALARS);
            // fifo_limit is inclusive; +1 gives bytes through end of CB.
            uint32_t bytes_until_wrap = cb_scal.fifo_limit + 1 - wptr;
            uint32_t pages_until_wrap = bytes_until_wrap / cb_scal.fifo_page_size;
            if (batch > pages_until_wrap) batch = pages_until_wrap;

            cb_reserve_back(CB_SCALARS, batch);
            for (uint32_t k = 0; k < batch; k++) {
                noc_async_read_tile(g_start + g + k, packs_acc,
                                    wptr + k * pack_bytes_padded);
            }
            noc_async_read_barrier();
            cb_push_back(CB_SCALARS, batch);
            g += batch;
        }

        // CB_TILE_META's write pointer wraps after each push_back, so
        // refresh our scratch pointer for the next tile's offset reads.
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
