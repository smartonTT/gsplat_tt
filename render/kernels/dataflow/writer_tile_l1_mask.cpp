// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-local L1 microblock-cull WRITER (iter 60 / step D).
//
// Consumes CB_KEEP batches from microblock_cull_compute and writes the packed
// 32-bit per-gaussian masks into an L1-interleaved buffer (same page layout as
// the old DRAM cull_masks). The blend reader bulk-loads from this L1 buffer —
// no global cull pass and no DRAM cull_masks on the blend hot path.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_MASK_SCR = 6;
constexpr uint32_t CB_KEEP     = 16;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t BATCH = 32;
constexpr uint32_t NUM_MB = 32;
constexpr uint32_t MASKS_PER_PAGE = 16;
constexpr uint32_t CHUNK_MAX = 16;
#ifndef MB_BUCKET_FIT
constexpr uint32_t MB_BUCKET_FIT = 8192u;
#endif

inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    noc_async_read_tile(elem >> 4, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[elem & 0xF];
}

}  // namespace

void kernel_main() {
    const uint32_t masks_addr        = get_arg_val<uint32_t>(0);
    const uint32_t ranges_addr       = get_arg_val<uint32_t>(1);
    const uint32_t subchunk_meta_addr= get_arg_val<uint32_t>(2);
    const uint32_t cull_base_addr    = get_arg_val<uint32_t>(3);
    const uint32_t tile_ids_addr     = get_arg_val<uint32_t>(4);
    const uint32_t lpt_meta_addr     = get_arg_val<uint32_t>(5);
    const uint32_t core_index        = get_arg_val<uint32_t>(6);

    constexpr auto masks_args  = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<masks_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_meta_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto cull_base_args = TensorAccessorArgs<subchunk_meta_args.next_compile_time_args_offset()>();
    constexpr auto tids_args   = TensorAccessorArgs<cull_base_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();

    const auto masks_acc  = TensorAccessor(masks_args, masks_addr, SOA_PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, ranges_addr, SOA_PAGE_BYTES);
    const auto subchunk_meta_acc = TensorAccessor(subchunk_meta_args, subchunk_meta_addr, SOA_PAGE_BYTES);
    const auto cull_base_acc = TensorAccessor(cull_base_args, cull_base_addr, SOA_PAGE_BYTES);
    const auto tids_acc   = TensorAccessor(tids_args, tile_ids_addr, IDS_PAGE_BYTES);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, SOA_PAGE_BYTES);

    constexpr uint32_t META_ELEMS_PER_PAGE = 16u;
    const uint32_t meta_elem0 = core_index * 2u;
    const uint32_t meta_page0 = meta_elem0 / META_ELEMS_PER_PAGE;
    const uint32_t meta_ip0   = meta_elem0 % META_ELEMS_PER_PAGE;

    const uint32_t scratch_addr = get_write_ptr(CB_MASK_SCR);
    auto scratch_ptr_meta = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    noc_async_read(get_noc_addr(meta_page0, lpt_meta_acc), scratch_addr, 64);
    noc_async_read_barrier();
    uint32_t tile_ids_start = scratch_ptr_meta[meta_ip0];
    uint32_t tile_ids_count = 0;
    if (meta_ip0 + 1u < META_ELEMS_PER_PAGE) {
        tile_ids_count = scratch_ptr_meta[meta_ip0 + 1u];
    } else {
        noc_async_read(get_noc_addr(meta_page0 + 1u, lpt_meta_acc), scratch_addr, 64);
        noc_async_read_barrier();
        tile_ids_count = scratch_ptr_meta[0];
    }

    if (tile_ids_count == 0) {
        return;
    }
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        uint32_t page_idx = tile_ids_start / CHUNK_MAX;
        uint32_t in_page  = tile_ids_start % CHUNK_MAX;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            noc_async_read_tile(page_idx, tids_acc, scratch_addr);
            noc_async_read_barrier();
            uint32_t take = CHUNK_MAX - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        uint32_t id_start = read_soa_u32(ranges_acc, tile_id * 2u + 0u, scratch_addr);
        uint32_t id_end   = read_soa_u32(ranges_acc, tile_id * 2u + 1u, scratch_addr);
        const uint32_t L = id_end - id_start;
        const uint32_t cull_base = read_soa_u32(cull_base_acc, tile_id, scratch_addr);

        uint32_t num_subchunks = 1;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read_tile(pg, subchunk_meta_acc, scratch_addr);
            noc_async_read_barrier();
            num_subchunks = scratch_ptr[off + 1u];
            if (num_subchunks == 0u) num_subchunks = 1u;
        }

        for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
            const uint32_t sc_off = sc * MB_BUCKET_FIT;
            const uint32_t L_sub = (sc_off >= L) ? 0u
                : ((L - sc_off > MB_BUCKET_FIT) ? MB_BUCKET_FIT : (L - sc_off));
            const uint32_t base = cull_base + sc_off;
            if (L_sub == 0) {
                continue;
            }

            uint32_t processed = 0;
            while (processed < L_sub) {
                uint32_t nb = L_sub - processed;
                if (nb > BATCH) nb = BATCH;

                cb_wait_front(CB_KEEP, 1);
                auto keep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_KEEP));

                for (uint32_t g = 0; g < nb; g++) {
                    uint32_t mask = 0u;
                    for (uint32_t m = 0; m < NUM_MB; m++) {
                        if (keep[perm(g, m)] != 0u) {
                            mask |= (1u << m);
                        }
                    }
                    scratch_ptr[g] = mask;
                }
                const uint32_t base_k = base + processed;
                uint32_t nb_pad = (nb + (MASKS_PER_PAGE - 1u)) & ~(MASKS_PER_PAGE - 1u);
                for (uint32_t g = nb; g < nb_pad; g++) scratch_ptr[g] = 0u;
                const uint32_t npg = nb_pad / MASKS_PER_PAGE;
                for (uint32_t pp = 0; pp < npg; pp++) {
                    noc_async_write(scratch_addr + pp * MASKS_PER_PAGE * 4u,
                                    get_noc_addr(base_k / MASKS_PER_PAGE + pp, masks_acc),
                                    MASKS_PER_PAGE * 4u);
                }
                noc_async_write_barrier();
                cb_pop_front(CB_KEEP, 1);

                processed += nb;
            }
        }
    }
}
