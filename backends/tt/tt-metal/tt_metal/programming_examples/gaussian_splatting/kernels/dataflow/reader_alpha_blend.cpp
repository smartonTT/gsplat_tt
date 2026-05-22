// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Alpha-blend READER kernel (NCRISC, NoC1; see DataMovementProcessor::RISCV_1
// in alpha_blend.cpp). Streams DRAM inputs into CBs for the compute kernel.
//
// Step 2: per-entry dyn pack (basis-form quadratic coeffs) and sorted gids;
// color/opacity are gathered from static_colors_opacity[gid] and composed into
// CB_SCALARS (64-byte, 10 fp32) for the compute kernel.
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
    constexpr uint32_t scalar_payload_bytes = 10 * 4;

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
    // and is reused as scratch by the inner gaussian loop.
    //
    // L1 layout:
    //   [0,  64)   -> sorted_gids page cache (16 uint32 gids per page)
    //   [64, 96)   -> static color/opacity scratch (8 fp32 per gid)
    //   [96,128)   -> reserved
    constexpr uint32_t CB_READER_SCRATCH = 24;
    constexpr uint32_t gids_per_page = gids_page_bytes / 4;
    constexpr uint32_t L1_OFF_GIDS = 0;
    constexpr uint32_t L1_OFF_STATIC = 64;
    const uint32_t reader_scratch_addr = get_write_ptr(CB_READER_SCRATCH);
    volatile uint32_t* gids_l1 =
        reinterpret_cast<volatile uint32_t*>(reader_scratch_addr + L1_OFF_GIDS);
    volatile uint32_t* static_l1 =
        reinterpret_cast<volatile uint32_t*>(reader_scratch_addr + L1_OFF_STATIC);

    for (uint32_t t = 0; t < tile_ids_count; t++) {
        uint32_t tile_id = tile_ids[t];

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
        uint32_t g_end = scratch_ptr[0];
        uint32_t g_count = g_end - g_start;

        cb_reserve_back(CB_TILE_META, 1);
        auto meta_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TILE_META));
        meta_ptr[0] = g_count;
        cb_push_back(CB_TILE_META, 1);

        cb_reserve_back(CB_PX, 1);
        noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
        cb_reserve_back(CB_PY, 1);
        noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));
        noc_async_read_barrier();
        cb_push_back(CB_PX, 1);
        cb_push_back(CB_PY, 1);

        // Page cache for sorted_gids reads. The cache is invalidated at
        // the start of every tile (last_gid_page = UINT32_MAX). Across
        // a single tile's per-Gaussian loop, consecutive entries in
        // the sorted list are very likely in the same 64-byte gid page,
        // so the cache eliminates the redundant gid-page reads within a
        // tile (typical hit rate >50%).
        uint32_t last_gid_page = UINT32_MAX;

        for (uint32_t g = 0; g < g_count; g++) {
            uint32_t entry_id = g_start + g;
            uint32_t gid_page = entry_id / gids_per_page;
            if (gid_page != last_gid_page) {
                uint64_t gids_noc = get_noc_addr(gid_page, sorted_gids_acc);
                noc_async_read(gids_noc, reader_scratch_addr + L1_OFF_GIDS, gids_page_bytes);
                noc_async_read_barrier();
                last_gid_page = gid_page;
            }
            uint32_t gid = gids_l1[entry_id % gids_per_page];

            cb_reserve_back(CB_SCALARS, 1);
            uint32_t cb_addr = get_write_ptr(CB_SCALARS);

            // dyn pack writes directly into CB_SCALARS at offset 0 (32B,
            // aligned because CB pages are 64B-aligned). The first 6 fp32
            // are valid (A,B,C,D,E,F); the next 2 are zero pad from the
            // DRAM dyn page and are overwritten by the static gather below.
            uint64_t dyn_noc = get_noc_addr(entry_id, dyn_packs_acc);
            noc_async_read(dyn_noc, cb_addr, dyn_pack_page_bytes);

            // static color/opacity goes to the L1 scratch region; we can't
            // NoC-write directly to cb_addr + 24 because offset 24 is not
            // 16-byte aligned (NoC destination alignment requirement).
            uint64_t static_noc = get_noc_addr(gid, static_colors_acc);
            noc_async_read(static_noc, reader_scratch_addr + L1_OFF_STATIC, static_page_bytes);
            noc_async_read_barrier();

            // Compose CB_SCALARS layout the compute kernel expects:
            //   [A, B, C, D, E, F, R, G, B, opacity]
            // dyn already covers elements [0..5]; static gathers [6..9].
            volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(cb_addr);
            for (uint32_t i = 0; i < 4; i++) {
                out[6 + i] = static_l1[i];
            }
            // Elements [10..15] of the 64-byte CB_SCALARS page are CB pad
            // and compute never reads them; we leave them undefined.

            cb_push_back(CB_SCALARS, 1);
        }

        // Refresh scratch_addr for the next tile's offset reads.
        scratch_addr = get_write_ptr(CB_TILE_META);
        scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    }
}
