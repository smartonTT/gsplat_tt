// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-tile STABLE depth-radix sort — amendment-002 tt-003 (S1).
//
// One data-movement kernel (scalar C++ on a RISC core), mirroring the
// gsplat_tt convention of doing integer/scalar "compute" inside a dataflow
// kernel (see tile_assign_bbox / tile_assign_scatter). It reproduces
// gsplat_cpu::sort.cpp Pass 3 (the per-tile sort) EXACTLY: a stable sort on the
// uint32 depth_bits key (n <= 16: insertion sort; else the adaptive LSD radix
// of sort_radix_tile_algo.h, R11: digits sized to the tile's key range and n).
// The (key, id) pair is moved together; only the depth_bits key drives order.
//
// Each core processes an LPT-assigned slice of NON-EMPTY tiles. Tiles live in
// a PAGE-ALIGNED DRAM layout (each tile owns ceil(n/16) exclusive 64B pages),
// so every read/write is a whole page exclusive to one tile — no cross-tile
// page sharing, no write races. The host compacts the aligned segments back
// into the CPU-contiguous order after readback.
//
// RUNTIME ARGS
//   0: keys_addr       DRAM base, uint32 SoA depth_bits (aligned layout)
//   1: ids_addr        DRAM base, uint32 SoA gaussian ids (aligned layout)
//   2: out_addr        DRAM base, uint32 SoA sorted ids (aligned layout, out)
//   3: tile_ids_addr   DRAM base, uint32 LPT tile-id list
//   4: tmeta_addr      DRAM base, uint32 [pstart_page, n] pair per tile
//   5: tile_ids_start  this core's element offset into the LPT list
//   6: tile_ids_count  number of tiles this core handles
//
// COMPILE-TIME ARGS: 5 TensorAccessorArgs (keys, ids, out, tile_ids, tmeta).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_radix_tile_algo.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;        // 16 uint32 per page
constexpr uint32_t ELEMS_PER_PAGE = 16;

// L1 scratch CB ids (declared in sort_device.cpp). Used as fixed scratch
// regions via get_write_ptr (no cb_reserve/push).
constexpr uint32_t CB_KIN  = 0;   // keys ping
constexpr uint32_t CB_IIN  = 1;   // ids  ping
constexpr uint32_t CB_KOUT = 2;   // keys pong
constexpr uint32_t CB_IOUT = 3;   // ids  pong
constexpr uint32_t CB_TIDS = 4;   // tile-id list scratch (one page)
constexpr uint32_t CB_META = 5;   // tmeta scratch (one page)
constexpr uint32_t CB_HIST = 6;   // radix histograms (HIST_WORDS uint32)

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("sort_tile_depth");
    const uint32_t keys_addr      = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr       = get_arg_val<uint32_t>(1);
    const uint32_t out_addr       = get_arg_val<uint32_t>(2);
    const uint32_t tile_ids_addr  = get_arg_val<uint32_t>(3);
    const uint32_t tmeta_addr     = get_arg_val<uint32_t>(4);
    const uint32_t tile_ids_start = get_arg_val<uint32_t>(5);
    const uint32_t tile_ids_count = get_arg_val<uint32_t>(6);

    constexpr auto keys_args     = TensorAccessorArgs<0>();
    constexpr auto ids_args      = TensorAccessorArgs<keys_args.next_compile_time_args_offset()>();
    constexpr auto out_args      = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto tmeta_args    = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();

    const auto keys_acc     = TensorAccessor(keys_args,     keys_addr,     PAGE_BYTES);
    const auto ids_acc      = TensorAccessor(ids_args,      ids_addr,      PAGE_BYTES);
    const auto out_acc      = TensorAccessor(out_args,      out_addr,      PAGE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, PAGE_BYTES);
    const auto tmeta_acc    = TensorAccessor(tmeta_args,    tmeta_addr,    PAGE_BYTES);

    if (tile_ids_count == 0) {
        return;
    }

    const uint32_t tids_scratch = get_write_ptr(CB_TIDS);
    auto tids_ptr = reinterpret_cast<volatile uint32_t*>(tids_scratch);

    const uint32_t kin_l1  = get_write_ptr(CB_KIN);
    const uint32_t iin_l1  = get_write_ptr(CB_IIN);
    const uint32_t kout_l1 = get_write_ptr(CB_KOUT);
    const uint32_t iout_l1 = get_write_ptr(CB_IOUT);

    auto kin  = reinterpret_cast<uint32_t*>(kin_l1);
    auto iin  = reinterpret_cast<uint32_t*>(iin_l1);
    auto kout = reinterpret_cast<uint32_t*>(kout_l1);
    auto iout = reinterpret_cast<uint32_t*>(iout_l1);
    auto hist = reinterpret_cast<uint32_t*>(get_write_ptr(CB_HIST));

    const uint32_t meta_scratch = get_write_ptr(CB_META);
    auto meta_ptr = reinterpret_cast<volatile uint32_t*>(meta_scratch);

    // Stream this core's tile-ID slice page by page (no large L1 stack array).
    uint32_t cur_tids_page = 0xFFFFFFFFu;  // which list page is in CB_TIDS
    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t list_idx = tile_ids_start + ti;
        const uint32_t list_page = list_idx / ELEMS_PER_PAGE;
        const uint32_t list_ip   = list_idx % ELEMS_PER_PAGE;
        if (list_page != cur_tids_page) {
            noc_async_read(get_noc_addr(list_page, tile_ids_acc), tids_scratch, PAGE_BYTES);
            noc_async_read_barrier();
            cur_tids_page = list_page;
        }
        const uint32_t t = tids_ptr[list_ip];

        // tmeta[t] = (pstart_page, n) at element indices 2t, 2t+1. Both land
        // in the same 64B page (2t is even, so 2t%16 <= 14).
        const uint32_t meta_page = (t * 2u) / ELEMS_PER_PAGE;
        const uint32_t meta_ip   = (t * 2u) % ELEMS_PER_PAGE;
        noc_async_read(get_noc_addr(meta_page, tmeta_acc), meta_scratch, PAGE_BYTES);
        noc_async_read_barrier();
        const uint32_t pstart_page = meta_ptr[meta_ip];
        const uint32_t n           = meta_ptr[meta_ip + 1];

        if (n == 0) continue;
        const uint32_t npages = (n + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;

        // DMA the tile's exclusive pages into L1 (keys + ids).
        for (uint32_t p = 0; p < npages; p++) {
            noc_async_read(get_noc_addr(pstart_page + p, keys_acc),
                           kin_l1 + p * PAGE_BYTES, PAGE_BYTES);
            noc_async_read(get_noc_addr(pstart_page + p, ids_acc),
                           iin_l1 + p * PAGE_BYTES, PAGE_BYTES);
        }
        noc_async_read_barrier();

        // The NoC wrote kin/iin behind the compiler's back: stop it hoisting
        // the plain loads below above the barrier.
        asm volatile("" ::: "memory");

        // ── Pass 3: per-tile stable sort on the depth_bits key ──────────
        // Adaptive stable LSD radix (sort_radix_tile_algo.h): same permutation
        // as gsplat_cpu radix_sort_tile, fewer passes and no per-pass rescan.
        const bool in_pong = sort_radix_tile::sort_pairs(kin, iin, kout, iout, n, hist);
        const uint32_t sorted_ids_l1 = in_pong ? iout_l1 : iin_l1;

        // Drain the RISC stores before the NoC reads the sorted ids from L1.
        asm volatile("fence" ::: "memory");

        // Write sorted ids back to the tile's exclusive aligned pages.
        for (uint32_t p = 0; p < npages; p++) {
            noc_async_write(sorted_ids_l1 + p * PAGE_BYTES,
                            get_noc_addr(pstart_page + p, out_acc), PAGE_BYTES);
        }
        noc_async_write_barrier();
    }
}
