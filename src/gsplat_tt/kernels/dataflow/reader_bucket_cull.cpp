// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ROUTE C — SFPU microblock-cull READER over the per-tile DENSE record bucket,
// run INSIDE the sort stage (GSPLAT_TT_BUCKET_MASK).
//
// WHY (the spin kill): the per-tile cull_masks DRAM round-trip needed an ~84 ms
// per-tile busy-wait (MB_CULL_SPIN) because the masks were written by a program
// dispatched immediately before blend; only writes from the genuinely-separate
// sort-stage dispatch read back spin-free (iter 24/26). This reader feeds the
// SAME microblock_cull_compute.cpp SFPU kernel — but it streams each tile's
// records from the DENSE bucket sort_tile_recs (written by sort_bin's scatter,
// in bin/gaussian order, SEQUENTIAL not gathered) instead of gathering the
// proj_m_* SoA by depth-sorted id. The mask it produces is baked back into the
// record (writer_bucket_cull.cpp, record word 10) so the blend reads it spin-
// free with the rest of the L1-resident record.
//
// Per LPT tile this core owns:
//   1. (once) stream the two constant box-origin ramps into CB_BOX_OX/OY.
//   2. read the tile's (rec_start, count) from sort_bucket_meta, push
//      [count, tx_pix, ty_pix] into CB_CULL_COUNTS,
//   3. stream the count records from sort_tile_recs[rec_start..+count) (each a
//      64B AoS page {a,b,c,px,py,op,r,g,b,depth,...}), emit a 6-word cull coeff
//      row {a,b,c,px,py,op} + the per-gaussian Mahalanobis threshold thr into
//      CB_CULL_COEFF — byte-identical to reader_microblock_cull's coeff rows.
//
// PURE INTEGER / DATA-MOVEMENT (one logf/candidate for thr, as in the original
// cull reader; the heavy per-microblock metric stays on the SFPU).
//
// RUNTIME ARGS
//   0: tile_recs base   1: bucket_meta base
//   2: box_ox ramp base 3: box_oy ramp base
//   4: tile_ids base    5: tile_ids_start  6: tile_ids_count  7: tiles_x
//   8: floor_bits
// COMPILE-TIME: 5 DRAM-interleaved TensorAccessorArgs (tile_recs, bucket_meta,
//   box_ox, box_oy, tile_ids).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_BOX_OX     = 0;
constexpr uint32_t CB_BOX_OY     = 1;
constexpr uint32_t CB_CULL_COEFF = 2;
constexpr uint32_t CB_CULL_COUNTS= 3;
constexpr uint32_t CB_SCR        = 4;   // reader-private record/meta/ids scratch

constexpr uint32_t REC_PAGE_BYTES  = 64;
constexpr uint32_t IDS_PAGE_BYTES  = 64;
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t TILE_SIZE       = 32;
constexpr uint32_t CHUNK_MAX       = 16;  // records read per barrier

template <typename Acc>
inline uint32_t read_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    noc_async_read_tile(elem >> 4, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[elem & 0xF];
}

}  // namespace

void kernel_main() {
    const uint32_t tile_recs_addr  = get_arg_val<uint32_t>(0);
    const uint32_t bucket_meta_addr= get_arg_val<uint32_t>(1);
    const uint32_t box_ox_addr     = get_arg_val<uint32_t>(2);
    const uint32_t box_oy_addr     = get_arg_val<uint32_t>(3);
    const uint32_t tile_ids_addr   = get_arg_val<uint32_t>(4);
    const uint32_t tile_ids_start  = get_arg_val<uint32_t>(5);
    const uint32_t tile_ids_count  = get_arg_val<uint32_t>(6);
    const uint32_t tiles_x         = get_arg_val<uint32_t>(7);
    const uint32_t floor_bits      = get_arg_val<uint32_t>(8);
    float contrib_floor;
    __builtin_memcpy(&contrib_floor, &floor_bits, 4);

    constexpr auto recs_args  = TensorAccessorArgs<0>();
    constexpr auto meta_args  = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
    constexpr auto bx_args    = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr auto by_args     = TensorAccessorArgs<bx_args.next_compile_time_args_offset()>();
    constexpr auto tids_args  = TensorAccessorArgs<by_args.next_compile_time_args_offset()>();

    const auto recs_acc = TensorAccessor(recs_args, tile_recs_addr,   REC_PAGE_BYTES);
    const auto meta_acc = TensorAccessor(meta_args, bucket_meta_addr, REC_PAGE_BYTES);
    const auto bx_acc   = TensorAccessor(bx_args,   box_ox_addr,      RAMP_TILE_BYTES);
    const auto by_acc   = TensorAccessor(by_args,   box_oy_addr,      RAMP_TILE_BYTES);
    const auto tids_acc = TensorAccessor(tids_args, tile_ids_addr,    IDS_PAGE_BYTES);

    if (tile_ids_count == 0) {
        return;
    }

    // (0) Stream the two constant box-origin ramps ONCE; compute keeps them
    // resident in the CB (waits once, never pops).
    cb_reserve_back(CB_BOX_OX, 1);
    noc_async_read_tile(0, bx_acc, get_write_ptr(CB_BOX_OX));
    cb_reserve_back(CB_BOX_OY, 1);
    noc_async_read_tile(0, by_acc, get_write_ptr(CB_BOX_OY));
    noc_async_read_barrier();
    cb_push_back(CB_BOX_OX, 1);
    cb_push_back(CB_BOX_OY, 1);

    // Cache this core's tile-ID slice in L1.
    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t scratch_addr = get_write_ptr(CB_SCR);
        auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
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
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;

        // Dense bucket (rec_start, count) from sort_bucket_meta (start,count pair).
        uint32_t rec_start, L;
        {
            const uint32_t scr = get_write_ptr(CB_SCR);
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read_tile(pg, meta_acc, scr);
            noc_async_read_barrier();
            auto mp = reinterpret_cast<volatile uint32_t*>(scr);
            rec_start = mp[off];
            L = mp[off + 1u];  // off even -> off+1 same page
        }

        cb_reserve_back(CB_CULL_COUNTS, 1);
        {
            auto cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COUNTS));
            cnt[0] = L;
            cnt[1] = tx * TILE_SIZE;
            cnt[2] = ty * TILE_SIZE;
        }
        cb_push_back(CB_CULL_COUNTS, 1);

        const uint32_t rec_scr = get_write_ptr(CB_SCR);
        uint32_t processed = 0;
        while (processed < L) {
            uint32_t take = L - processed;
            if (take > CHUNK_MAX) take = CHUNK_MAX;
            // Bulk-load `take` SEQUENTIAL record pages under one barrier.
            for (uint32_t j = 0; j < take; ++j) {
                noc_async_read_tile(rec_start + processed + j, recs_acc,
                                    rec_scr + j * REC_PAGE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < take; ++j) {
                auto recp = reinterpret_cast<volatile uint32_t*>(rec_scr + j * REC_PAGE_BYTES);
                cb_reserve_back(CB_CULL_COEFF, 1);
                auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COEFF));
                row[0] = recp[0];  // cov_a
                row[1] = recp[1];  // cov_b
                row[2] = recp[2];  // cov_c
                row[3] = recp[3];  // mean_x (image space)
                row[4] = recp[4];  // mean_y
                row[5] = recp[5];  // opacity
                // Per-gaussian Mahalanobis threshold thr = -2*log(floor/op),
                // computed HERE on the data mover (logf is exact on the RISC,
                // garbage on the compute TRISC). <0 sentinel for op<=floor.
                {
                    const uint32_t op_bits = recp[5];
                    float opf;
                    __builtin_memcpy(&opf, &op_bits, 4);
                    float thrf;
                    if (opf <= contrib_floor) {
                        thrf = -1.0f;
                    } else {
                        thrf = -2.0f * __builtin_logf(contrib_floor / opf);
                    }
                    uint32_t thr_bits;
                    __builtin_memcpy(&thr_bits, &thrf, 4);
                    row[6] = thr_bits;
                }
                cb_push_back(CB_CULL_COEFF, 1);
            }
            processed += take;
        }
    }
}
