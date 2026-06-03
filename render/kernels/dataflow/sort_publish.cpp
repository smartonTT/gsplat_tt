// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compact page-aligned radix output (buf_out) into contiguous sort_sorted_ids.
// Mirrors the host Pass4 loop in sort_device.cpp (pstart_elem, counts, starts).
//
// RUNTIME ARGS
//   0: out_addr         DRAM, uint32 aligned sorted ids (per-tile pages)
//   1: sorted_addr      DRAM, uint32 contiguous resident output
//   2: ranges_addr      DRAM, uint32 [start,end) per tile (2 * num_tiles)
//   3: tile_ids_addr    DRAM, uint32 LPT tile-id list
//   4: tmeta_addr       DRAM, uint32 [pstart_page, n_pad] per tile
//   5: tile_ids_start   this core's offset into the LPT list
//   6: tile_ids_count   number of tiles this core handles
//
// COMPILE-TIME ARGS: 5 TensorAccessorArgs (out, sorted, ranges, tile_ids, tmeta).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

constexpr uint32_t CB_TIDS = 0;
constexpr uint32_t CB_META = 1;
constexpr uint32_t CB_SCRATCH = 2;  // tmeta / ranges / one out page

}  // namespace

void kernel_main() {
    const uint32_t out_addr         = get_arg_val<uint32_t>(0);
    const uint32_t sorted_addr      = get_arg_val<uint32_t>(1);
    const uint32_t ranges_addr      = get_arg_val<uint32_t>(2);
    const uint32_t tile_ids_addr    = get_arg_val<uint32_t>(3);
    const uint32_t tmeta_addr       = get_arg_val<uint32_t>(4);
    const uint32_t tile_ids_start   = get_arg_val<uint32_t>(5);
    const uint32_t tile_ids_count   = get_arg_val<uint32_t>(6);

    constexpr auto out_args       = TensorAccessorArgs<0>();
    constexpr auto sorted_args    = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto ranges_args    = TensorAccessorArgs<sorted_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args  = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto tmeta_args     = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();

    const auto out_acc       = TensorAccessor(out_args,       out_addr,       PAGE_BYTES);
    const auto sorted_acc    = TensorAccessor(sorted_args,    sorted_addr,    PAGE_BYTES);
    const auto ranges_acc    = TensorAccessor(ranges_args,    ranges_addr,    PAGE_BYTES);
    const auto tile_ids_acc  = TensorAccessor(tile_ids_args,  tile_ids_addr,  PAGE_BYTES);
    const auto tmeta_acc     = TensorAccessor(tmeta_args,     tmeta_addr,     PAGE_BYTES);

    if (tile_ids_count == 0) {
        return;
    }

    const uint32_t tids_scratch = get_write_ptr(CB_TIDS);
    auto tids_ptr = reinterpret_cast<volatile uint32_t*>(tids_scratch);
    const uint32_t meta_scratch = get_write_ptr(CB_META);
    auto meta_ptr = reinterpret_cast<volatile uint32_t*>(meta_scratch);
    const uint32_t scratch = get_write_ptr(CB_SCRATCH);
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch);

    uint32_t cur_tids_page = 0xFFFFFFFFu;
    uint32_t cur_meta_page = 0xFFFFFFFFu;

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

        const uint32_t meta_page = (t * 2u) / ELEMS_PER_PAGE;
        const uint32_t meta_ip   = (t * 2u) % ELEMS_PER_PAGE;
        if (meta_page != cur_meta_page) {
            noc_async_read(get_noc_addr(meta_page, tmeta_acc), meta_scratch, PAGE_BYTES);
            noc_async_read_barrier();
            cur_meta_page = meta_page;
        }
        const uint32_t src_start = meta_ptr[meta_ip] * ELEMS_PER_PAGE;

        // NOTE: CB_SCRATCH is reused below as the per-page copy buffer, which
        // clobbers this ranges page. Re-read every tile (cheap, 64B) rather than
        // caching — a stale cache here silently binds a tile to the wrong dst
        // slice whenever two LPT-ordered tiles share a ranges page.
        const uint32_t rng_page = (t * 2u) / ELEMS_PER_PAGE;
        const uint32_t rng_ip   = (t * 2u) % ELEMS_PER_PAGE;
        noc_async_read(get_noc_addr(rng_page, ranges_acc), scratch, PAGE_BYTES);
        noc_async_read_barrier();
        const uint32_t dst_start = scratch_ptr[rng_ip];
        const uint32_t dst_end   = scratch_ptr[rng_ip + 1];
        const uint32_t n = (dst_end > dst_start) ? (dst_end - dst_start) : 0u;
        if (n == 0) continue;

        const uint32_t npages = (n + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
        const uint32_t src_base_page = src_start / ELEMS_PER_PAGE;
        const uint32_t dst_base_page = dst_start / ELEMS_PER_PAGE;
        for (uint32_t p = 0; p < npages; p++) {
            noc_async_read(get_noc_addr(src_base_page + p, out_acc), scratch, PAGE_BYTES);
            noc_async_read_barrier();
            noc_async_write(scratch, get_noc_addr(dst_base_page + p, sorted_acc), PAGE_BYTES);
            noc_async_write_barrier();
        }
    }
}
