// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-local L1 microblock-cull WRITER (iter 102 / M3).
//
// Task #59: consumes the reader's L1 slab slot (CB_BUCKET) and the band cull's
// CB_KEEP tiles. Each keep tile covers COEFF_BATCH records; record i's mask
// halves sit as fp32 2^23 + bits in tile words 64*(i/32) + 2*(i%32) (+1 for
// bits 16-31), so the 32-bit mask is pure integer: (lo & 0xffff) | (hi << 16).
// The mask goes into WORD3 of the record in L1 (word3 is the depth key, dead
// after the sort, so it carries the mask to the blend), then the whole slab is
// written back to sort_subchunk_payload in SLAB_PAGE_BYTES writes, the same
// shape sort_subchunk_materialize emits it in. No DRAM read-modify-write.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_MASK_SCR = 6;
constexpr uint32_t CB_BUCKET   = 8;    // the reader's slab slot (task #59: consumed here)
constexpr uint32_t CB_KEEP     = 16;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t CHUNK_MAX = 16;
#ifndef MB_BUCKET_FIT
constexpr uint32_t MB_BUCKET_FIT = 8192u;
#endif
constexpr uint32_t BULK_REC_SLOT = (MB_BUCKET_FIT + 1u) >> 1;  // CB_BUCKET pages per slot
constexpr uint32_t COEFF_BATCH = 128u;  // records per keep tile (band cull)
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / L1_SPLAT_BYTES;  // 64

template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    noc_async_read_tile(elem >> 4, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[elem & 0xF];
}

}  // namespace

void kernel_main() {
    const uint32_t payload_addr      = get_arg_val<uint32_t>(0);  // sort_subchunk_payload slab
    const uint32_t ranges_addr       = get_arg_val<uint32_t>(1);
    const uint32_t subchunk_meta_addr= get_arg_val<uint32_t>(2);  // [dir_base, num_sc] per tile
    const uint32_t subchunk_dir_addr = get_arg_val<uint32_t>(3);  // dir_base+sc -> payload_page
    const uint32_t tile_ids_addr     = get_arg_val<uint32_t>(4);
    const uint32_t lpt_meta_addr     = get_arg_val<uint32_t>(5);
    const uint32_t core_index        = get_arg_val<uint32_t>(6);

    constexpr auto payload_args = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<payload_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_meta_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_dir_args = TensorAccessorArgs<subchunk_meta_args.next_compile_time_args_offset()>();
    constexpr auto tids_args   = TensorAccessorArgs<subchunk_dir_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();

    const auto payload_acc = TensorAccessor(payload_args, payload_addr, SLAB_PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, ranges_addr, SOA_PAGE_BYTES);
    const auto subchunk_meta_acc = TensorAccessor(subchunk_meta_args, subchunk_meta_addr, SOA_PAGE_BYTES);
    const auto subchunk_dir_acc = TensorAccessor(subchunk_dir_args, subchunk_dir_addr, SOA_PAGE_BYTES);
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

        // blend_subchunk_meta: per-tile (dir_base, num_sc). dir_base indexes
        // sort_subchunk_dir; mirror the cull/blend readers exactly so each
        // subchunk's slab base (payload_page) matches the records the readers
        // load (slab record k == depth-rank k == cull coeff row k).
        uint32_t dir_base = 0;
        uint32_t num_subchunks = 1;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read_tile(pg, subchunk_meta_acc, scratch_addr);
            noc_async_read_barrier();
            dir_base = scratch_ptr[off];
            num_subchunks = scratch_ptr[off + 1u];
            if (num_subchunks == 0u) num_subchunks = 1u;
        }

        for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
            const uint32_t sc_off = sc * MB_BUCKET_FIT;
            const uint32_t L_sub = (sc_off >= L) ? 0u
                : ((L - sc_off > MB_BUCKET_FIT) ? MB_BUCKET_FIT : (L - sc_off));
            if (L_sub == 0) {
                continue;
            }

            uint32_t payload_page = 0;
            {
                const uint32_t de = (dir_base + sc) * 4u;
                const uint32_t dpg = de >> 4;
                const uint32_t dof = de & 0xF;
                noc_async_read_tile(dpg, subchunk_dir_acc, scratch_addr);
                noc_async_read_barrier();
                payload_page = scratch_ptr[dof];
            }

            cb_wait_front(CB_BUCKET, BULK_REC_SLOT);
            const uint32_t slab = get_read_ptr(CB_BUCKET);
            for (uint32_t base = 0; base < L_sub; base += COEFF_BATCH) {
                const uint32_t n = (L_sub - base < COEFF_BATCH) ? (L_sub - base) : COEFF_BATCH;
                cb_wait_front(CB_KEEP, 1);
                auto keep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_KEEP));
                auto rec = reinterpret_cast<volatile uint32_t*>(slab + base * L1_SPLAT_BYTES);
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t o = 64u * (i >> 5) + 2u * (i & 31u);
                    const uint32_t lo = keep[o], hi = keep[o + 1u];
                    rec[i * 8u + 3u] = (lo & 0xffffu) | (hi << 16);
                }
                cb_pop_front(CB_KEEP, 1);
            }
            asm volatile("fence" ::: "memory");  // word3 stores reach L1 before the NoC reads it
            const uint32_t out_pages = (L_sub + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
            for (uint32_t p = 0; p < out_pages; ++p) {
                const uint32_t recs = (p + 1u < out_pages)
                    ? SLAB_RECS_PER_PAGE
                    : (L_sub - p * SLAB_RECS_PER_PAGE);
                noc_async_write(slab + p * SLAB_PAGE_BYTES,
                                get_noc_addr(payload_page + p, payload_acc),
                                recs * L1_SPLAT_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_BUCKET, BULK_REC_SLOT);
        }
    }
}
