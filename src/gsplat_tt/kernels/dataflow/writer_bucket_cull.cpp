// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ROUTE C — SFPU microblock-cull WRITER, run INSIDE the sort stage
// (GSPLAT_TT_BUCKET_MASK). Consumes the per-32-gaussian-batch fp32 keep tiles
// microblock_cull_compute.cpp produces in CB_KEEP, packs the 32-bit per-gaussian
// microblock mask (the SAME perm(g,m) CB-linear unpack as writer_microblock_cull
// .cpp), and BAKES it into word 10 of each gaussian's 64B record in the dense
// bucket sort_tile_recs via a read-modify-write.
//
// WHY a sort-stage write: the cull_masks DRAM round-trip used by the blend-side
// cull needed an ~84 ms per-tile read-completion/settle busy-wait before the
// blend could read it back (iter 24/26 — only writes from a genuinely separate
// SORT-stage dispatch read back spin-free). Baking the mask into the record HERE
// lets the downstream L1-resident blend read recp[10] with NO spin and NO
// separate cull_masks buffer.
//
// The RMW preserves record words 0..9 (cov a/b/c, image-space center, opacity,
// colour, depth key) and only sets word 10. Records are processed in the SAME
// per-tile, candidate-major, batch-of-32 order the reader streamed them and the
// compute consumed them, so CB_KEEP batch k holds the masks for bucket slots
// [rec_start+k*32, +nb).
//
// RUNTIME ARGS
//   0: tile_recs base   1: bucket_meta base
//   2: tile_ids base    3: tile_ids_start  4: tile_ids_count
// COMPILE-TIME: 3 DRAM-interleaved TensorAccessorArgs (tile_recs, bucket_meta,
//   tile_ids). tile_recs is read AND written.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_KEEP  = 16;  // compute -> writer fp32 keep tiles (one per 32-gaussian batch)
constexpr uint32_t CB_WSCR  = 6;   // RMW record scratch + ids/meta scratch

constexpr uint32_t REC_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t CHUNK_MAX      = 16;   // ids/u32 per 64B page
constexpr uint32_t BATCH          = 32;   // gaussians per CB_KEEP tile
constexpr uint32_t NUM_MB         = 32;

// CB-linear position of the keep flag for (gaussian g, microblock m). Identical
// to writer_microblock_cull.cpp::perm — the copy/pack round-trip is lane-order
// invariant, so this is the inverse of the host box-ramp layout.
inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

}  // namespace

void kernel_main() {
    const uint32_t tile_recs_addr   = get_arg_val<uint32_t>(0);
    const uint32_t bucket_meta_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_ids_addr    = get_arg_val<uint32_t>(2);
    const uint32_t tile_ids_start   = get_arg_val<uint32_t>(3);
    const uint32_t tile_ids_count   = get_arg_val<uint32_t>(4);

    constexpr auto recs_args = TensorAccessorArgs<0>();
    constexpr auto meta_args = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
    constexpr auto tids_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();

    const auto recs_acc = TensorAccessor(recs_args, tile_recs_addr,   REC_PAGE_BYTES);
    const auto meta_acc = TensorAccessor(meta_args, bucket_meta_addr, REC_PAGE_BYTES);
    const auto tids_acc = TensorAccessor(tids_args, tile_ids_addr,    IDS_PAGE_BYTES);

    if (tile_ids_count == 0) {
        return;
    }

    // Cache this core's tile-ID slice in L1 (same order as the reader).
    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t scratch_addr = get_write_ptr(CB_WSCR);
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

        uint32_t rec_start, L;
        {
            const uint32_t scr = get_write_ptr(CB_WSCR);
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read_tile(pg, meta_acc, scr);
            noc_async_read_barrier();
            auto mp = reinterpret_cast<volatile uint32_t*>(scr);
            rec_start = mp[off];
            L = mp[off + 1u];
        }

        const uint32_t wscr = get_write_ptr(CB_WSCR);
        uint32_t processed = 0;
        while (processed < L) {
            uint32_t nb = L - processed;
            if (nb > BATCH) nb = BATCH;

            // One CB_KEEP tile per 32-gaussian batch (matches the compute).
            cb_wait_front(CB_KEEP, 1);
            auto keep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_KEEP));

            // Load nb SEQUENTIAL record pages, pack+set word 10, write back.
            for (uint32_t g = 0; g < nb; ++g) {
                noc_async_read_tile(rec_start + processed + g, recs_acc,
                                    wscr + g * REC_PAGE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t g = 0; g < nb; ++g) {
                uint32_t mask = 0u;
                for (uint32_t m = 0; m < NUM_MB; ++m) {
                    if (keep[perm(g, m)] != 0u) {
                        mask |= (1u << m);
                    }
                }
                reinterpret_cast<volatile uint32_t*>(wscr + g * REC_PAGE_BYTES)[10] = mask;
            }
            for (uint32_t g = 0; g < nb; ++g) {
                noc_async_write_tile(rec_start + processed + g, recs_acc,
                                     wscr + g * REC_PAGE_BYTES);
            }
            noc_async_write_barrier();

            cb_pop_front(CB_KEEP, 1);
            processed += nb;
        }
    }
}
