// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-local L1 microblock-cull WRITER (iter 102 / M3).
//
// Consumes CB_KEEP batches from microblock_cull_compute and writes the packed
// 32-bit per-gaussian microblock mask into WORD3 of each record in the
// depth-sorted PACK2 slab (sort_subchunk_payload) — the SAME slab the blend
// reader bulk-loads. word3 is the depth key, dead after the sort, so it carries
// the mask between the cull and blend kernel launches (slab lives in DRAM
// between them; normal launch ordering provides cross-kernel visibility).
//
// Write-back strategy (M3): ALIGNED per-batch 64B page read-modify-write. The
// strided 4B word3 default misaligned both ends (DRAM word3 at byte 12/44, L1
// src at +4; Blackhole NoC needs >=16B). Instead we read the batch's 64B slab
// pages into L1, patch word3 of each record (PACK2: record k -> page k/2, half
// k&1; word3 == u32 index (k>>1)*16 + (k&1)*8 + 3), and write the whole 64B
// pages back. processed advances by BATCH=32 == 16 pages, so every batch is
// page-aligned (no partial-page sharing across batches); the RMW preserves the
// untouched record words (cov/mean/op/color).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_MASK_SCR = 6;
constexpr uint32_t CB_KEEP     = 16;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t BATCH = 32;
constexpr uint32_t NUM_MB = 32;
constexpr uint32_t CHUNK_MAX = 16;
#ifndef MB_BUCKET_FIT
constexpr uint32_t MB_BUCKET_FIT = 8192u;
#endif

// PACK2 slab geometry: two 32B splats per 64B page; record g -> page g/2, half
// g&1. As u32 words: page (g>>1) base = (g>>1)*16; half g&1 starts at +8*(g&1);
// word3 (the dead depth key, now the mask) is at +3 within the half.
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
constexpr uint32_t PAGE_U32 = 16u;     // 64B page = 16 u32
constexpr uint32_t HALF_U32 = 8u;      // 32B splat = 8 u32

inline uint32_t word3_u32_index(uint32_t g) {
    return (g >> 1) * PAGE_U32 + (g & 1u) * HALF_U32 + 3u;
}

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

    const auto payload_acc = TensorAccessor(payload_args, payload_addr, L1_PACK_PAGE_BYTES);
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

            uint32_t processed = 0;
            while (processed < L_sub) {
                uint32_t nb = L_sub - processed;
                if (nb > BATCH) nb = BATCH;

                cb_wait_front(CB_KEEP, 1);
                auto keep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_KEEP));

                // processed is a multiple of BATCH (==16 pages), so this batch's
                // records start on a 64B page boundary. RMW the npg pages spanning
                // [processed, processed+nb): read -> patch word3 -> write back.
                const uint32_t page0 = payload_page + (processed >> 1);
                const uint32_t npg = (nb + 1u) >> 1;
                for (uint32_t pp = 0; pp < npg; ++pp) {
                    noc_async_read_tile(page0 + pp, payload_acc,
                                        scratch_addr + pp * L1_PACK_PAGE_BYTES);
                }
                noc_async_read_barrier();

                for (uint32_t g = 0; g < nb; g++) {
                    uint32_t mask = 0u;
                    for (uint32_t m = 0; m < NUM_MB; m++) {
                        if (keep[perm(g, m)] != 0u) {
                            mask |= (1u << m);
                        }
                    }
                    scratch_ptr[word3_u32_index(g)] = mask;
                }

                for (uint32_t pp = 0; pp < npg; ++pp) {
                    noc_async_write(scratch_addr + pp * L1_PACK_PAGE_BYTES,
                                    get_noc_addr(page0 + pp, payload_acc),
                                    L1_PACK_PAGE_BYTES);
                }
                noc_async_write_barrier();
                cb_pop_front(CB_KEEP, 1);

                processed += nb;
            }
        }
    }
}
