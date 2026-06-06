// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Post-publish subchunk directory (iter 55 / step B): per-tile blend dispatch
// meta [num_subchunks, count], payload prefix, and per-subchunk dir entries
// (page, L_sub, flags). Serial on core 0 — cheap vs materialize.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

// iter 110 (A2): payload_page is expressed in LARGE slab pages (must match the
// host SubchunkLayout sizing and both readers/materialize/mask-writer). The slab
// is a contiguous array of 32B records; SLAB_RECS_PER_PAGE records per page.
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / 32u;  // 64

constexpr uint32_t CB_SCR = 0;

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("sort_subchunk_dir");
    const uint32_t ranges_addr = get_arg_val<uint32_t>(0);
    const uint32_t blend_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t dir_addr = get_arg_val<uint32_t>(2);
    const uint32_t prefix_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t bucket_fit = get_arg_val<uint32_t>(5);

    constexpr auto ranges_args = TensorAccessorArgs<0>();
    constexpr auto blend_meta_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto dir_args = TensorAccessorArgs<blend_meta_args.next_compile_time_args_offset()>();
    constexpr auto prefix_args = TensorAccessorArgs<dir_args.next_compile_time_args_offset()>();

    const auto ranges_acc = TensorAccessor(ranges_args, ranges_addr, PAGE_BYTES);
    const auto blend_meta_acc = TensorAccessor(blend_meta_args, blend_meta_addr, PAGE_BYTES);
    const auto dir_acc = TensorAccessor(dir_args, dir_addr, PAGE_BYTES);
    const auto prefix_acc = TensorAccessor(prefix_args, prefix_addr, PAGE_BYTES);

    if (num_tiles == 0) {
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    auto scrp = reinterpret_cast<volatile uint32_t*>(scr);

    uint32_t page_cursor = 0;
    uint32_t dir_cursor = 0;

    for (uint32_t t = 0; t < num_tiles; ++t) {
        uint32_t id_start = 0;
        uint32_t id_end = 0;
        {
            const uint32_t e0 = t * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, ranges_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            id_start = scrp[off];
            id_end = (off + 1u < ELEMS_PER_PAGE) ? scrp[off + 1u] : 0u;
            if (off + 1u >= ELEMS_PER_PAGE) {
                noc_async_read(get_noc_addr(pg + 1u, ranges_acc), scr, PAGE_BYTES);
                noc_async_read_barrier();
                id_end = scrp[0];
            }
        }
        const uint32_t count = (id_end > id_start) ? (id_end - id_start) : 0u;
        const uint32_t num_sc =
            count == 0u ? 1u : (count + bucket_fit - 1u) / bucket_fit;

        {
            const uint32_t e0 = t * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, blend_meta_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            // blend_subchunk_meta per tile: [dir_base, num_subchunks] (host layout).
            scrp[off] = dir_cursor;
            scrp[off + 1u] = num_sc;
            noc_async_write(scr, get_noc_addr(pg, blend_meta_acc), PAGE_BYTES);
            noc_async_write_barrier();
        }

        {
            const uint32_t pg = t >> 4;
            const uint32_t off = t & 0xF;
            noc_async_read(get_noc_addr(pg, prefix_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            scrp[off] = page_cursor;
            noc_async_write(scr, get_noc_addr(pg, prefix_acc), PAGE_BYTES);
            noc_async_write_barrier();
        }

        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t l_sub = (sc_off >= count) ? 0u
                : ((count - sc_off > bucket_fit) ? bucket_fit : (count - sc_off));
            const uint32_t flags =
                ((sc > 0u) ? 2u : 0u) | ((sc + 1u == num_sc) ? 1u : 0u);

            const uint32_t e0 = dir_cursor * 4u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, dir_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            scrp[off] = page_cursor;
            scrp[off + 1u] = l_sub;
            scrp[off + 2u] = flags;
            scrp[off + 3u] = 0u;
            noc_async_write(scr, get_noc_addr(pg, dir_acc), PAGE_BYTES);
            noc_async_write_barrier();

            if (l_sub > 0u) {
                page_cursor += (l_sub + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
            }
            dir_cursor += 1u;
        }
    }
}
