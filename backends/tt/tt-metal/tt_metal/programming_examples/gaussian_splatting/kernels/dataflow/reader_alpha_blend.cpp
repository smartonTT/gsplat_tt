// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Alpha-blend READER kernel (NCRISC, NoC1; see DataMovementProcessor::RISCV_1
// in alpha_blend.cpp). Streams DRAM inputs into CBs for the compute kernel.
//
// Step 2: per-entry dyn pack (mean + cov_inv) and sorted gids; color/opacity
// are gathered from static_colors_opacity[gid] and composed into CB_SCALARS
// (64-byte, 9 fp32) for the unchanged compute kernel.
//
// RUNTIME ARGS
//   0: dyn_packs_addr
//   1: tile_offsets_addr
//   2: px_addr
//   3: py_addr
//   4: tile_ids_addr
//   5: tile_ids_start
//   6: tile_ids_count
//   7: sorted_gids_addr
//   8: static_colors_opacity_addr
//
// COMPILE-TIME ARGS: 7 TensorAccessorArgs (dyn_packs, offsets, px, py,
// tile_ids, sorted_gids, static_colors_opacity).
//
// iter-020 reader optimizations — three independent changes:
//
//   A) COALESCED OFFSETS READ: The two separate 4-byte reads of
//      offsets[tile_id] and offsets[tile_id+1] (each followed by its own
//      noc_async_read_barrier) are replaced by a single 8-byte read +
//      one barrier. tile_offsets is a non-interleaved, page_size=4 linear
//      buffer; offsets[tile_id] and offsets[tile_id+1] are contiguous in
//      DRAM, and the NoC destination (scratch_addr = CB_TILE_META write ptr)
//      is 4-byte aligned, satisfying Blackhole's alignment requirement.
//      Saves 1 NoC barrier + 1 NoC round-trip per tile (1024 tiles/frame).
//
//   B) PX/PY READS ISSUED BEFORE OFFSETS: Issuing the PX and PY tile reads
//      (each ~2 KB) before the 8-byte offsets read lets all three NoC
//      transactions be in-flight simultaneously. A single combined barrier
//      covers all three, replacing the original 3-barrier sequence (offsets
//      barrier 1 → offsets barrier 2 → PX/PY barrier). Per-tile init cost
//      drops from 3 barriers to 1. cb_reserve_back(CB_PX/PY) is called
//      before issuing the respective reads (write pointer must be valid
//      before use); the offsets read reuses scratch_addr from the CB_TILE_META
//      write pointer as before.
//
//   C) PER-GAUSSIAN STATIC PREFETCH PIPELINE: A 2-slot L1 double-buffer for
//      static color/opacity data. In the original code each iteration issues
//      dyn[g] + static[g] simultaneously, then barriers — both reads are
//      outstanding together. In the pipeline, static[g] is pre-issued one
//      iteration early (in iter g-1), so the main barrier in iter g covers
//      only dyn[g] + static[g+1]. Because static[g] completed in iter g-1's
//      barrier, it is guaranteed available when iter g composes, and the
//      effective barrier wait reduces to max(dyn_latency, static[g+1]_latency)
//      from max(dyn_latency, static[g]_latency) — but more importantly, the
//      NCRISC compose CPU work (4 volatile copies) now overlaps with
//      static[g+1]'s DRAM flight rather than happening strictly after it.
//      On gid_page cache MISS for g+1 we cannot resolve gid_{g+1} without
//      the gid_page landing first, so we fall back: issue dyn[g] only,
//      barrier, read new gid_page (extra barrier), issue static[g+1] to
//      nxt slot (no barrier — covered by g+1's main barrier). This path adds
//      exactly one extra barrier vs the pipeline path, matching the original
//      code's cost on misses. Hit rate is typically >50% (16 gids per 64B
//      page; ~1500 Gaussians/tile → ~94 page loads, ~1406 hit iters).

constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;

void kernel_main() {
    uint32_t dyn_packs_addr            = get_arg_val<uint32_t>(0);
    uint32_t tile_offsets_addr         = get_arg_val<uint32_t>(1);
    uint32_t px_addr                   = get_arg_val<uint32_t>(2);
    uint32_t py_addr                   = get_arg_val<uint32_t>(3);
    uint32_t tile_ids_addr             = get_arg_val<uint32_t>(4);
    uint32_t tile_ids_start            = get_arg_val<uint32_t>(5);
    uint32_t tile_ids_count            = get_arg_val<uint32_t>(6);
    uint32_t sorted_gids_addr          = get_arg_val<uint32_t>(7);
    uint32_t static_colors_opacity_addr = get_arg_val<uint32_t>(8);

    constexpr uint32_t CB_PX        = 0;
    constexpr uint32_t CB_PY        = 1;
    constexpr uint32_t CB_SCALARS   = 2;
    constexpr uint32_t CB_TILE_META = 3;

    const uint32_t tile_bytes = get_tile_size(CB_PX);
    constexpr uint32_t dyn_pack_page_bytes = 32;
    constexpr uint32_t static_page_bytes = 32;
    constexpr uint32_t scalar_pack_page_bytes = 64;
    constexpr uint32_t gids_page_bytes = 64;
    constexpr uint32_t tile_ids_page_bytes = 64;
    constexpr uint32_t scalar_payload_bytes = 9 * 4;

    constexpr auto dyn_packs_args = TensorAccessorArgs<0>();
    constexpr auto offsets_args =
        TensorAccessorArgs<dyn_packs_args.next_compile_time_args_offset()>();
    constexpr auto px_args =
        TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto py_args = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args =
        TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto sorted_gids_args =
        TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();
    constexpr auto static_colors_args =
        TensorAccessorArgs<sorted_gids_args.next_compile_time_args_offset()>();

    const auto dyn_packs_acc =
        TensorAccessor(dyn_packs_args, dyn_packs_addr, dyn_pack_page_bytes);
    const auto offsets_acc =
        TensorAccessor(offsets_args, tile_offsets_addr, /*page_size=*/4);
    const auto px_acc = TensorAccessor(px_args, px_addr, tile_bytes);
    const auto py_acc = TensorAccessor(py_args, py_addr, tile_bytes);
    const auto tile_ids_acc =
        TensorAccessor(tile_ids_args, tile_ids_addr, tile_ids_page_bytes);
    const auto sorted_gids_acc =
        TensorAccessor(sorted_gids_args, sorted_gids_addr, gids_page_bytes);
    const auto static_colors_acc = TensorAccessor(
        static_colors_args, static_colors_opacity_addr, static_page_bytes);

    if (tile_ids_count == 0) {
        return;
    }

    uint32_t scratch_addr = get_write_ptr(CB_TILE_META);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t ids_per_page = tile_ids_page_bytes / 4;
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            uint64_t page_noc = get_noc_addr(page_idx, tile_ids_acc);
            noc_async_read(page_noc, scratch_addr, tile_ids_page_bytes);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) {
                take = remaining;
            }
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            }
            out_idx += take;
            remaining -= take;
            page_idx += 1;
            in_page = 0;
        }
    }

    // NoC destinations MUST live in worker L1 (NoC-addressable), NOT in
    // NCRISC's private IRAM (the kernel's stack). The watcher will fault
    // with "Local L1 address overflow" if we noc_async_read into a
    // stack-allocated buffer.
    //
    // CB_READER_SCRATCH is a dedicated reader-only L1 CB (depth=1,
    // page_size=READER_SCRATCH_PAGE_BYTES=128). We never push or pop it;
    // the L1 region lives at a stable address for the kernel's lifetime
    // and is reused as scratch by the inner Gaussian loop.
    //
    // L1 layout (iter-020: slot B promoted from "reserved"):
    //   [0,  64)  -> sorted_gids page cache (16 uint32 gids per 64B page)
    //   [64, 96)  -> static color/opacity scratch SLOT A (8 fp32 = 32B)
    //   [96,128)  -> static color/opacity scratch SLOT B (8 fp32 = 32B)
    constexpr uint32_t CB_READER_SCRATCH  = 24;
    constexpr uint32_t gids_per_page      = gids_page_bytes / 4;
    constexpr uint32_t L1_OFF_GIDS        = 0;
    constexpr uint32_t L1_OFF_STATIC_A    = 64;   // iter-020: was L1_OFF_STATIC
    constexpr uint32_t L1_OFF_STATIC_B    = 96;   // iter-020: was "reserved"
    const uint32_t reader_scratch_addr = get_write_ptr(CB_READER_SCRATCH);
    volatile uint32_t* gids_l1 =
        reinterpret_cast<volatile uint32_t*>(reader_scratch_addr + L1_OFF_GIDS);

    for (uint32_t t = 0; t < tile_ids_count; t++) {
        uint32_t tile_id = tile_ids[t];

        // B: Issue PX/PY reads first so they overlap with the offsets read.
        // cb_reserve_back must precede the write-ptr use for each CB.
        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));

        // A: Coalesced offsets read. offsets[tile_id] and offsets[tile_id+1]
        // are contiguous in the non-interleaved linear tile_offsets buffer
        // (page_size=4, sequential pages are adjacent in DRAM). One 8B read
        // to scratch_addr (= CB_TILE_META write ptr, 4B-aligned) replaces the
        // original two 4B reads + two barriers.
        uint64_t off_noc = get_noc_addr(tile_id, offsets_acc);
        noc_async_read(off_noc, scratch_addr, 8);

        // Combined barrier: covers PX tile, PY tile, and 8B offsets read.
        // Replaces the original three-barrier sequence (offsets×2, PX/PY×1).
        noc_async_read_barrier();

        uint32_t g_start = scratch_ptr[0];
        uint32_t g_end   = scratch_ptr[1];
        uint32_t g_count = g_end - g_start;

        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // C: Two-slot static prefetch pipeline (see file-level comment).
        // Invariant entering iter g: gids_l1 is valid for entry_id = g_start+g,
        // and the static read for gid_{g} has been issued (outstanding or done)
        // into the slot pointed to by cur_static_off.
        uint32_t last_gid_page   = UINT32_MAX;
        uint32_t cur_static_off  = L1_OFF_STATIC_A;  // slot for current iter
        uint32_t nxt_static_off  = L1_OFF_STATIC_B;  // slot to prefetch into

        // Pre-loop: load gid_page for entry 0 (always a cache miss) and
        // issue static[gid_0] into cur slot. The barrier here covers only
        // the gid_page read; the static read is left outstanding and will be
        // gathered by the g=0 iteration's main barrier below.
        if (g_count > 0) {
            uint32_t entry_id_0  = g_start;
            uint32_t gid_page_0  = entry_id_0 / gids_per_page;
            uint64_t gids_noc    = get_noc_addr(gid_page_0, sorted_gids_acc);
            noc_async_read(gids_noc, reader_scratch_addr + L1_OFF_GIDS, gids_page_bytes);
            noc_async_read_barrier();
            last_gid_page = gid_page_0;

            uint32_t gid_0           = gids_l1[entry_id_0 % gids_per_page];
            uint64_t static_noc_0    = get_noc_addr(gid_0, static_colors_acc);
            noc_async_read(static_noc_0, reader_scratch_addr + cur_static_off, static_page_bytes);
            // Outstanding: static[gid_0] → cur slot. No barrier yet.
        }

        for (uint32_t g = 0; g < g_count; g++) {
            uint32_t entry_id = g_start + g;
            // gids_l1 is valid for entry_id (maintained by pre-loop setup and
            // the gid_page miss handler at the bottom of this loop body).

            cb_reserve_back(CB_SCALARS, 1);
            uint32_t cb_addr = get_write_ptr(CB_SCALARS);

            // Issue dyn pack for g directly into CB_SCALARS (32B, 4B-aligned).
            uint64_t dyn_noc = get_noc_addr(entry_id, dyn_packs_acc);
            noc_async_read(dyn_noc, cb_addr, dyn_pack_page_bytes);

            // Prefetch static for g+1 when gid_page stays the same (cache hit
            // path). Both dyn[g] and static[g+1] are now in-flight together.
            bool prefetched_next = false;
            if (g + 1 < g_count) {
                uint32_t next_entry_id  = g_start + g + 1;
                uint32_t next_gid_page  = next_entry_id / gids_per_page;
                if (next_gid_page == last_gid_page) {
                    uint32_t next_gid        = gids_l1[next_entry_id % gids_per_page];
                    uint64_t next_static_noc = get_noc_addr(next_gid, static_colors_acc);
                    noc_async_read(next_static_noc,
                                   reader_scratch_addr + nxt_static_off,
                                   static_page_bytes);
                    prefetched_next = true;
                }
            }

            // Main barrier: covers dyn[g] (just issued), static[g] (issued in
            // iter g-1 / pre-loop, guaranteed done by prev barrier or now), and
            // optionally static[g+1] (just issued on cache hit path).
            noc_async_read_barrier();

            // Cache miss for g+1: we now know the gid_page has changed, so we
            // must read the new gid_page before we can resolve gid_{g+1} and
            // issue its static read. This costs one extra barrier (same as the
            // original code on a gid_page miss). The static read is issued with
            // no barrier here — it will be caught by iter g+1's main barrier.
            if (g + 1 < g_count && !prefetched_next) {
                uint32_t next_entry_id  = g_start + g + 1;
                uint32_t next_gid_page  = next_entry_id / gids_per_page;
                uint64_t gids_noc       = get_noc_addr(next_gid_page, sorted_gids_acc);
                noc_async_read(gids_noc, reader_scratch_addr + L1_OFF_GIDS, gids_page_bytes);
                noc_async_read_barrier();   // must land before we read gid_{g+1}
                last_gid_page = next_gid_page;

                uint32_t next_gid        = gids_l1[next_entry_id % gids_per_page];
                uint64_t next_static_noc = get_noc_addr(next_gid, static_colors_acc);
                noc_async_read(next_static_noc,
                               reader_scratch_addr + nxt_static_off,
                               static_page_bytes);
                // Outstanding: static[gid_{g+1}] → nxt slot. Covered by g+1 barrier.
            }

            // Compose CB_SCALARS: dyn pack covers [0..4] (mean_x, mean_y,
            // cov_a, 2·cov_b, cov_c); static[g] covers [5..8] (R, G, B, opacity).
            // static[g] is guaranteed complete: it was included in either this
            // iteration's main barrier (g=0 path, or after a gid_page miss in
            // iter g-1) or in iter g-1's main barrier (cache-hit pipeline path).
            volatile uint32_t* static_cur =
                reinterpret_cast<volatile uint32_t*>(reader_scratch_addr + cur_static_off);
            volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(cb_addr);
            for (uint32_t i = 0; i < 4; i++) {
                out[5 + i] = static_cur[i];
            }
            // Elements [9..15] are CB pad; compute never reads them.

            cb_push_back(CB_SCALARS, 1);

            // Swap slots: nxt becomes cur for the next iteration.
            uint32_t tmp    = cur_static_off;
            cur_static_off  = nxt_static_off;
            nxt_static_off  = tmp;
        }

        // Refresh scratch_addr for the next tile's coalesced offsets read.
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr  = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
