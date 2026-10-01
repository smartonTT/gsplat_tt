// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Post-radix subchunk materialize (iter 54 / step A): depth-sorted PACK2 payloads.
// In-budget tiles (count <= bucket_fit): bulk-copy buf_l1_recs + L1 radix
// permute (no per-splat blendrec gather). iter 113 (sort Stage 1): the depth
// permutation is applied L1->L1 into a contiguous slab scratch (CB_SLAB) and
// the depth-sorted slab is emitted in coalesced SLAB_PAGE_BYTES page writes —
// NO per-record 32B DRAM scatter (the old emit posted one noc_async_write per
// record). Overflow tiles: sc==0 uses sort_sorted_ids-order blendrec gather
// (masks follow radix ids, not L1 slots); sc>=1 uses the same batched blendrec
// gather into depth-ordered PACK2 slabs (unchanged fallback).
//
// iter 130 (load balance): the unit of work is a (tile, subchunk) item, not a
// tile. iter-130 MEASURED that the dominant materialize cost is the OVERFLOW
// gather (24.6 ms/view on the busiest core vs the in-budget permute's 1.7 ms),
// and that the per-tile LPT (count-weighted, shared with sort/cull/blend)
// overloads whichever cores own the big overflow tiles (max 27.1 ms vs the
// 17.0 ms balanced floor). Each (tile, sc) item is independent — the in-budget
// permute reads buf_l1_recs keyed by tile and writes the payload keyed by
// (tile, sc); the gather reads sort_sorted_ids/blendrec by global id and writes
// the payload by (tile, sc) — so ANY core can process ANY item with byte-
// identical output. The host therefore balances all items across cores with a
// gather-cost-weighted LPT and hands each core its own work-item slice.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"
#include "sort_radix_tile_algo.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t L1_SPLAT_BYTES = 32u;
// Task #86: buf_l1_recs and the overflow region hold REC_PAGE_RECS 32B records
// per REC_PAGE_BYTES DRAM page (== render_config::kRecPageBytes), so a bucket
// is read in 2 KB transfers (was 64 B PACK2 pages: ~1.0 ms/view per mover).
// Record g of a bucket still lands at buck + g*32.
constexpr uint32_t REC_PAGE_BYTES = 2048u;
constexpr uint32_t REC_PAGE_RECS = REC_PAGE_BYTES / L1_SPLAT_BYTES;  // 64
// iter 110 (A2): the depth-sorted slab is materialized into a DRAM buffer with a
// LARGE interleave page (SLAB_PAGE_BYTES) so the cull/blend readers coalesce the
// per-subchunk load into ceil(L/SLAB_RECS_PER_PAGE) big transfers. The slab is
// still a contiguous array of 32B records: subchunk-local record g lives at
// page (sc_page + g/SLAB_RECS_PER_PAGE), byte (g % SLAB_RECS_PER_PAGE)*32. The
// source L1 bucket layout (PACK2, 2 recs / 64B) is unchanged.
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / L1_SPLAT_BYTES;  // 64
constexpr uint32_t TILE_SIZE = 32u;
// iter 76: larger blendrec gather batches (fewer read/write barriers on sc>=1).
constexpr uint32_t REC_BATCH = 32u;
// Task #35: an over-cap gather item covers part `part` of its subchunk, records
// [part * GATHER_PART_RECS, +GATHER_PART_RECS) (host sort_mover_split.h
// kGatherPartRecs). Other items have part 0 and cover their whole tile/subchunk.
constexpr uint32_t GATHER_PART_RECS = 2048u;

// Dual mover: BRISC runs this kernel too, on its own (smaller) copies of every
// CB at id + 16 (MAT_CB_BASE) and its own work-item slice. The host gives it
// only items that fit its buffers (whole-tile items of <= idx_stride records,
// any gather item).
#ifndef MAT_CB_BASE
#define MAT_CB_BASE 0
#endif
constexpr uint32_t CB_SCR = MAT_CB_BASE + 0;
constexpr uint32_t CB_IDS = MAT_CB_BASE + 1;
constexpr uint32_t CB_REC = MAT_CB_BASE + 2;
constexpr uint32_t CB_PACK = MAT_CB_BASE + 3;
constexpr uint32_t CB_BUCKET = MAT_CB_BASE + 4;
constexpr uint32_t CB_BSORT = MAT_CB_BASE + 5;
// iter 113 (sort Stage 1): contiguous L1 scratch the depth permutation lands in
// (record k at byte k*32) so the depth-sorted slab is written to DRAM in
// coalesced SLAB_PAGE_BYTES pages instead of bucket_fit per-record 32B writes.
constexpr uint32_t CB_SLAB = MAT_CB_BASE + 6;

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline uint32_t f_to_bits(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

// Copy the n records buck[sorted[k]] (32 B each; PACK2 slot g sits at byte
// 32g) to slab record k. All 8 words of a record are loaded before any store:
// alternating volatile load/store made every store wait for its own load.
inline void permute_records(uint32_t buck, uint32_t slab, const uint32_t* sorted,
                            uint32_t n) {
    for (uint32_t k = 0; k < n; ++k) {
        auto src = reinterpret_cast<volatile uint32_t*>(buck + sorted[k] * L1_SPLAT_BYTES);
        auto dst = reinterpret_cast<volatile uint32_t*>(slab + k * L1_SPLAT_BYTES);
        const uint32_t w0 = src[0], w1 = src[1], w2 = src[2], w3 = src[3];
        const uint32_t w4 = src[4], w5 = src[5], w6 = src[6], w7 = src[7];
        dst[0] = w0; dst[1] = w1; dst[2] = w2; dst[3] = w3;
        dst[4] = w4; dst[5] = w5; dst[6] = w6; dst[7] = w7;
    }
}

// Read n 32B records starting at page page0 into buck (record g at buck + g*32):
// whole REC_PAGE_BYTES pages, only the used bytes of the last one.
template <typename Acc>
inline void read_bucket(const Acc& acc, uint32_t page0, uint32_t n, uint32_t buck) {
    const uint32_t npages = (n + REC_PAGE_RECS - 1u) / REC_PAGE_RECS;
    for (uint32_t q = 0; q < npages; ++q) {
        const uint32_t recs = (q + 1u < npages) ? REC_PAGE_RECS : (n - q * REC_PAGE_RECS);
        noc_async_read(get_noc_addr(page0 + q, acc), buck + q * REC_PAGE_BYTES,
                       recs * L1_SPLAT_BYTES);
    }
    noc_async_read_barrier();
}

// Task #86: fine per-item zones for attribution (host env
// GSPLAT_TT_MATCULL_PROF=1; compiled out by default).
#if defined(MATCULL_PROF) && MATCULL_PROF
#define MAT_PZ(name) DeviceZoneScopedN(name)
#else
#define MAT_PZ(name) ((void)0)
#endif

// Task #90: the SFPU cull runs in this program (mat_cull_compute.cpp). Each
// mover culls the slab it just depth-sorted before writing it: it transposes
// every COEFF_BATCH records into an fp32 coefficient tile on its own stream
// (CB_COEFF / CB_KEEP), and patches the returned 32-bit microblock mask into
// word3 of each record (the depth key, dead after the sort), exactly what
// reader_tile_l1_cull + writer_tile_l1_mask did in the separate cull program.
// At most CULL_DEPTH batches are in flight per mover (== the CB depths), so
// neither side ever blocks on a full CB. The stream ends with a tile whose
// word END_WORD is non-zero.
#if defined(FUSE_CULL) && FUSE_CULL
constexpr uint32_t CB_COEFF = MAT_CB_BASE + 8;
constexpr uint32_t CB_KEEP = MAT_CB_BASE + 9;
constexpr uint32_t COEFF_BATCH = 128u;      // 4 groups x 32 SFPU lanes
constexpr uint32_t OPQ_BIAS = 0x4B000000u;  // fp32 2^23; field = 2^23 + UNORM16 opacity
constexpr uint32_t END_WORD = 1023u;        // == mat_cull_compute.cpp (outside the SFPU fields)
constexpr uint32_t CULL_DEPTH = FUSE_CULL_DEPTH;

// == reader_tile_l1_cull.cpp fill_coeff_tile: record i -> lane i%32 of group
// i/32; field pairs (A,B), (C,opq), (mx,my) are adjacent tile words.
inline void fill_coeff_tile(uint32_t slab, uint32_t base, uint32_t n, uint32_t tile) {
    for (uint32_t i = 0; i < n; ++i) {
        auto src = reinterpret_cast<volatile uint32_t*>(slab + (base + i) * L1_SPLAT_BYTES);
        auto dst = reinterpret_cast<volatile uint32_t*>(tile) + 192u * (i >> 5) + 2u * (i & 31u);
        const uint32_t w0 = src[0], w1 = src[1], w2 = src[2];
        const uint32_t w4 = src[4], w5 = src[5], w6 = src[6];
        dst[0] = w0;
        dst[1] = w1;
        dst[64] = w2;
        dst[65] = OPQ_BIAS | (w6 & 0xffffu);
        dst[128] = w4;
        dst[129] = w5;
    }
    reinterpret_cast<volatile uint32_t*>(tile)[END_WORD] = 0u;
}

// == writer_tile_l1_mask.cpp: record i's mask halves are 2^23 + bits at keep
// words 64*(i/32) + 2*(i%32) (+1 for bits 16-31).
inline void patch_batch(uint32_t slab, uint32_t base, uint32_t n) {
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

// Cull the n records at slab (record k at slab + 32k): word3 := mask.
inline void cull_slab(uint32_t slab, uint32_t n) {
    MAT_PZ("mat_cull");
    uint32_t pushed = 0, patched = 0;
    for (uint32_t base = 0; base < n; base += COEFF_BATCH) {
        if (pushed - patched == CULL_DEPTH) {
            const uint32_t pb = patched * COEFF_BATCH;
            patch_batch(slab, pb, (n - pb < COEFF_BATCH) ? (n - pb) : COEFF_BATCH);
            ++patched;
        }
        cb_reserve_back(CB_COEFF, 1);
        const uint32_t tile = get_write_ptr(CB_COEFF);
        fill_coeff_tile(slab, base, (n - base < COEFF_BATCH) ? (n - base) : COEFF_BATCH, tile);
        asm volatile("fence" ::: "memory");
        cb_push_back(CB_COEFF, 1);
        ++pushed;
    }
    while (patched < pushed) {
        const uint32_t pb = patched * COEFF_BATCH;
        patch_batch(slab, pb, (n - pb < COEFF_BATCH) ? (n - pb) : COEFF_BATCH);
        ++patched;
    }
    asm volatile("fence" ::: "memory");  // word3 stores reach L1 before the NoC reads the slab
}

// permute_records + cull_slab in one pass: slab[k] = buck[sorted[k]], and the
// coefficient tile is filled from the words already loaded (no re-read of the
// slab). word3 (the depth key) is not copied: patch_batch overwrites it with
// the mask for every k < n.
inline void permute_cull(uint32_t buck, uint32_t slab, const uint32_t* sorted, uint32_t n) {
    uint32_t pushed = 0, patched = 0;
    for (uint32_t base = 0; base < n; base += COEFF_BATCH) {
        const uint32_t nb = (n - base < COEFF_BATCH) ? (n - base) : COEFF_BATCH;
        if (pushed - patched == CULL_DEPTH) {
            const uint32_t pb = patched * COEFF_BATCH;
            patch_batch(slab, pb, (n - pb < COEFF_BATCH) ? (n - pb) : COEFF_BATCH);
            ++patched;
        }
        cb_reserve_back(CB_COEFF, 1);
        const uint32_t tile = get_write_ptr(CB_COEFF);
        for (uint32_t i = 0; i < nb; ++i) {
            const uint32_t k = base + i;
            auto src = reinterpret_cast<volatile uint32_t*>(buck + sorted[k] * L1_SPLAT_BYTES);
            auto dst = reinterpret_cast<volatile uint32_t*>(slab + k * L1_SPLAT_BYTES);
            auto ct = reinterpret_cast<volatile uint32_t*>(tile) + 192u * (i >> 5) + 2u * (i & 31u);
            const uint32_t w0 = src[0], w1 = src[1], w2 = src[2];
            const uint32_t w4 = src[4], w5 = src[5], w6 = src[6], w7 = src[7];
            dst[0] = w0; dst[1] = w1; dst[2] = w2;
            dst[4] = w4; dst[5] = w5; dst[6] = w6; dst[7] = w7;
            ct[0] = w0;
            ct[1] = w1;
            ct[64] = w2;
            ct[65] = OPQ_BIAS | (w6 & 0xffffu);
            ct[128] = w4;
            ct[129] = w5;
        }
        reinterpret_cast<volatile uint32_t*>(tile)[END_WORD] = 0u;
        asm volatile("fence" ::: "memory");
        cb_push_back(CB_COEFF, 1);
        ++pushed;
    }
    while (patched < pushed) {
        const uint32_t pb = patched * COEFF_BATCH;
        patch_batch(slab, pb, (n - pb < COEFF_BATCH) ? (n - pb) : COEFF_BATCH);
        ++patched;
    }
    asm volatile("fence" ::: "memory");  // word3 stores reach L1 before the NoC reads the slab
}

inline void cull_end_stream() {
    cb_reserve_back(CB_COEFF, 1);
    reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_COEFF))[END_WORD] = 1u;
    asm volatile("fence" ::: "memory");
    cb_push_back(CB_COEFF, 1);
}
#if defined(MATCULL_FOLD) && MATCULL_FOLD
#define PERMUTE_CULL(buck, slab, sorted, n) permute_cull((buck), (slab), (sorted), (n))
#else
#define PERMUTE_CULL(buck, slab, sorted, n) \
    do { permute_records((buck), (slab), (sorted), (n)); cull_slab((slab), (n)); } while (0)
#endif
#else
#define PERMUTE_CULL(buck, slab, sorted, n) permute_records((buck), (slab), (sorted), (n))
#endif

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("sort_subchunk_mat");
    const uint32_t sorted_addr    = get_arg_val<uint32_t>(0);
    const uint32_t ranges_addr    = get_arg_val<uint32_t>(1);
    const uint32_t blendrec_addr  = get_arg_val<uint32_t>(2);
    const uint32_t l1_recs_addr   = get_arg_val<uint32_t>(3);
    const uint32_t payload_addr   = get_arg_val<uint32_t>(4);
    const uint32_t blend_meta_addr = get_arg_val<uint32_t>(5);
    const uint32_t dir_addr       = get_arg_val<uint32_t>(6);
    // iter 130: arg7/8/9 carry the per-core (tile, subchunk) WORK-ITEM slice
    // (flat u32 array: item i = {tile_id at 2i, sc at 2i+1}). Replaces the old
    // per-tile tile_ids slice — work is now balanced at subchunk granularity.
    const uint32_t work_addr      = get_arg_val<uint32_t>(7);
    const uint32_t work_start     = get_arg_val<uint32_t>(8);
    const uint32_t work_count     = get_arg_val<uint32_t>(9);
    const uint32_t tiles_x        = get_arg_val<uint32_t>(10);
    const uint32_t bucket_fit     = get_arg_val<uint32_t>(11);
    // iter-138 (Stage-2b overflow pre-pack): the compact overflow region (PACK2,
    // pre-packed by sort_bucket_emit) + per-tile start slot (sentinel 0xFFFFFFFF
    // for non-prepacked tiles) + the L1 cap. For an in-cap overflow tile this core
    // gets ONE work item (sc==0) and processes the WHOLE tile: coalesced bucket
    // read + L1 radix depth-permute + per-subchunk slab emit — NO blendrec gather.
    const uint32_t ov_recs_addr   = get_arg_val<uint32_t>(12);  // overflow region (0=off)
    const uint32_t ov_base_addr   = get_arg_val<uint32_t>(13);  // per-tile slot base (0=off)
    const uint32_t ov_cap         = get_arg_val<uint32_t>(14);  // kOverflowL1Cap
    // In-budget radix index stride = this mover's CB_BSORT record capacity
    // (bucket_fit on NCRISC, the smaller mover-0 cap on BRISC).
    const uint32_t idx_stride     = get_arg_val<uint32_t>(15);

    constexpr auto sorted_args = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<sorted_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto payload_args = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    constexpr auto blend_meta_args = TensorAccessorArgs<payload_args.next_compile_time_args_offset()>();
    constexpr auto dir_args = TensorAccessorArgs<blend_meta_args.next_compile_time_args_offset()>();
    constexpr auto work_args = TensorAccessorArgs<dir_args.next_compile_time_args_offset()>();
    // iter-138: overflow region (PACK2 64B page) + per-tile overflow base row.
    constexpr auto ov_recs_args = TensorAccessorArgs<work_args.next_compile_time_args_offset()>();
    constexpr auto ov_base_args = TensorAccessorArgs<ov_recs_args.next_compile_time_args_offset()>();

    const auto sorted_acc   = TensorAccessor(sorted_args,   sorted_addr,   PAGE_BYTES);
    const auto ranges_acc   = TensorAccessor(ranges_args,   ranges_addr,   PAGE_BYTES);
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    const auto l1_recs_acc  = TensorAccessor(l1_recs_args,  l1_recs_addr,  REC_PAGE_BYTES);
    const auto payload_acc  = TensorAccessor(payload_args,  payload_addr,  SLAB_PAGE_BYTES);
    const auto blend_meta_acc = TensorAccessor(blend_meta_args, blend_meta_addr, PAGE_BYTES);
    const auto dir_acc      = TensorAccessor(dir_args,      dir_addr,      PAGE_BYTES);
    const auto work_acc     = TensorAccessor(work_args,     work_addr,     PAGE_BYTES);
    const bool ov_enabled   = (ov_recs_addr != 0u) && (ov_base_addr != 0u);
    const auto ov_recs_acc  = TensorAccessor(ov_recs_args,  ov_recs_addr,  REC_PAGE_BYTES);
    const auto ov_base_acc  = TensorAccessor(ov_base_args,  ov_base_addr,  PAGE_BYTES);

    if (work_count == 0) {
#if defined(FUSE_CULL) && FUSE_CULL
        cull_end_stream();
#endif
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    auto scrp = reinterpret_cast<volatile uint32_t*>(scr);
    const uint32_t ids_scr = get_write_ptr(CB_IDS);
    auto idsp = reinterpret_cast<volatile uint32_t*>(ids_scr);
    const uint32_t rec_l1 = get_write_ptr(CB_REC);
    const uint32_t pack_l1 = get_write_ptr(CB_PACK);
    (void)pack_l1;  // unused when FUSE_CULL gathers into the L1 slab

    // Radix histograms in local memory (stack), see sort_radix_tile_algo.h.
    // The work items are no longer copied to a 4 KB stack array (that plus
    // the 3 KB histograms would not fit the 8 KB local memory): each item is
    // read from DRAM when it is processed (a few dozen items per core).
    sort_radix_tile::hist_t hist[sort_radix_tile::HIST_ENTRIES];

    for (uint32_t wi = 0; wi < work_count; wi++) {
        // The work buffer is a flat u32 array; item i = {tile_id at 2i,
        // sc | part << 8 at 2i+1}, both in the same 64 B page (2i is even).
        uint32_t tile_id, sc, part;
        {
            MAT_PZ("mat_meta");
            const uint32_t u = (work_start + wi) * 2u;
            noc_async_read(get_noc_addr(u / ELEMS_PER_PAGE, work_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            tile_id = scrp[u % ELEMS_PER_PAGE] & 0xFFFFFFu;
            const uint32_t w1 = scrp[u % ELEMS_PER_PAGE + 1u];
            sc = w1 & 0xFFu;
            part = (w1 >> 8) & 0xFFu;
        }
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;
        const float tx_tile = static_cast<float>(tx * TILE_SIZE);
        const float ty_tile = static_cast<float>(ty * TILE_SIZE);

        uint32_t id_start = 0, id_end = 0;
        {
            MAT_PZ("mat_meta");
            const uint32_t e0 = tile_id * 2u;
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
        if (count == 0u) {
            continue;
        }

        const uint32_t sc_off = sc * bucket_fit;
        const uint32_t L_sub = (sc_off >= count) ? 0u
            : ((count - sc_off > bucket_fit) ? bucket_fit : (count - sc_off));
        if (L_sub == 0u) {
            continue;
        }

        uint32_t dir_base = 0;
        {
            MAT_PZ("mat_meta");
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, blend_meta_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            dir_base = scrp[off];
        }

        // iter-138: in-cap overflow tile pre-pack path. The whole tile's records
        // are pre-packed (gaussian/core-major, identical to buf_l1_recs) in the
        // compact overflow region at tile_ov_base[tile_id]. Read them COALESCED,
        // L1-radix-sort ALL `count` by depth key, then emit each subchunk's slab
        // (coalesced page writes) — byte-identical to the sorted_ids blendrec
        // gather (same keys, same stable order) but no random gather. Only this
        // single sc==0 item is scheduled for the tile (build_mat_worklist).
        if (ov_enabled && count > bucket_fit && count <= ov_cap) {
            uint32_t ov_base = 0xFFFFFFFFu;
            {
                MAT_PZ("mat_meta");
                const uint32_t e0 = tile_id;  // 1 u32 per tile
                const uint32_t pg = e0 >> 4;
                const uint32_t off = e0 & 0xF;
                noc_async_read(get_noc_addr(pg, ov_base_acc), scr, PAGE_BYTES);
                noc_async_read_barrier();
                ov_base = scrp[off];
            }
            if (ov_base != 0xFFFFFFFFu) {
                // Coalesced read of the whole overflow bucket (2 KB pages;
                // ov_base is page-aligned).
                const uint32_t buck = get_write_ptr(CB_BUCKET);
                {
                    MAT_PZ("mat_ov_rd");
                    read_bucket(ov_recs_acc, ov_base / REC_PAGE_RECS, count, buck);
                }
                // Stable sort of ALL `count` records by key word[3] (adaptive
                // radix, sort_radix_tile_algo.h). Keys/ids in CB_BSORT; the
                // ping-pong pair borrows CB_SLAB (2*count u32 <= count*32 B),
                // which is only written after the sort.
                asm volatile("" ::: "memory");  // NoC filled buck behind the compiler
                const uint32_t bs = get_write_ptr(CB_BSORT);
                const uint32_t slab = get_write_ptr(CB_SLAB);
                uint32_t* kA = reinterpret_cast<uint32_t*>(bs);
                uint32_t* kB = reinterpret_cast<uint32_t*>(slab);
                const uint32_t* sorted;
                {
                    MAT_PZ("mat_ov_sort");
                    sorted = sort_radix_tile::sort_record_ids(
                        reinterpret_cast<volatile uint32_t*>(buck), count, kA, kA + ov_cap,
                        kB, kB + count, hist);
                }
                // Emit each subchunk's depth-sorted slab to its directory page run.
                const uint32_t num_sc = (count + bucket_fit - 1u) / bucket_fit;
                for (uint32_t s = 0; s < num_sc; ++s) {
                    const uint32_t sc_off2 = s * bucket_fit;
                    const uint32_t Ls = (count - sc_off2 > bucket_fit)
                        ? bucket_fit : (count - sc_off2);
                    uint32_t scp = 0;
                    {
                        MAT_PZ("mat_meta");
                        const uint32_t e0 = (dir_base + s) * 4u;
                        const uint32_t pg = e0 >> 4;
                        const uint32_t off = e0 & 0xF;
                        noc_async_read(get_noc_addr(pg, dir_acc), scr, PAGE_BYTES);
                        noc_async_read_barrier();
                        scp = scrp[off];
                    }
                    {
                        MAT_PZ("mat_ov_perm");
                        PERMUTE_CULL(buck, slab, sorted + sc_off2, Ls);
                    }
                    MAT_PZ("mat_ov_wr");
                    const uint32_t out_pages =
                        (Ls + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
                    for (uint32_t p = 0; p < out_pages; ++p) {
                        const uint32_t recs = (p + 1u < out_pages)
                            ? SLAB_RECS_PER_PAGE
                            : (Ls - p * SLAB_RECS_PER_PAGE);
                        noc_async_write(
                            slab + p * SLAB_PAGE_BYTES,
                            get_noc_addr(scp + p, payload_acc),
                            recs * L1_SPLAT_BYTES);
                    }
                    noc_async_write_barrier();
                }
                continue;
            }
        }

        // C1b: page index must match sort_subchunk_dir (same field blend reader DMAs).
        uint32_t sc_page = 0;
        {
            MAT_PZ("mat_meta");
            const uint32_t e0 = (dir_base + sc) * 4u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, dir_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            sc_page = scrp[off];
        }

        // In-budget sc==0: buf_l1_recs bulk + L1 depth permute. Overflow sc==0
        // falls through to sorted_ids gather (iter 83: L1 slot order != masks).
        if (sc == 0u && L_sub <= bucket_fit && count <= bucket_fit) {
            const uint32_t L = L_sub;
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                MAT_PZ("mat_rd");
                read_bucket(l1_recs_acc, tile_id * (bucket_fit / REC_PAGE_RECS), L, buck);
            }
            // Stable sort by key word[3] (adaptive radix, histograms in local
            // memory; see the overflow path above for the scratch layout).
            asm volatile("" ::: "memory");  // NoC filled buck behind the compiler
            const uint32_t bs = get_write_ptr(CB_BSORT);
            const uint32_t slab = get_write_ptr(CB_SLAB);
            uint32_t* kA = reinterpret_cast<uint32_t*>(bs);
            uint32_t* kB = reinterpret_cast<uint32_t*>(slab);
            const uint32_t* sorted;
            {
                MAT_PZ("mat_sort");
                sorted = sort_radix_tile::sort_record_ids(
                    reinterpret_cast<volatile uint32_t*>(buck), L, kA, kA + idx_stride,
                    kB, kB + L, hist);
            }
            // Stage 1: apply the radix permutation L1->L1 into a contiguous
            // slab scratch (output order), then emit the depth-sorted slab in
            // coalesced SLAB_PAGE_BYTES page writes (no per-record DRAM scatter).
            {
                MAT_PZ("mat_perm");
                PERMUTE_CULL(buck, slab, sorted, L);
            }
            MAT_PZ("mat_wr");
            const uint32_t out_pages =
                (L + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
            for (uint32_t p = 0; p < out_pages; ++p) {
                const uint32_t recs = (p + 1u < out_pages)
                    ? SLAB_RECS_PER_PAGE
                    : (L - p * SLAB_RECS_PER_PAGE);
                noc_async_write(
                    slab + p * SLAB_PAGE_BYTES,
                    get_noc_addr(sc_page + p, payload_acc),
                    recs * L1_SPLAT_BYTES);
            }
            noc_async_write_barrier();
            continue;
        }

        // sc>=1 / overflow sc==0: batched blendrec gather (iter 76: REC_BATCH=32,
        // per-slot PACK2, one write barrier per batch; reuse sorted-id page).
        MAT_PZ("mat_gather");
        const uint32_t id_start_sc = id_start + sc_off;
        uint32_t processed = part * GATHER_PART_RECS;
        const uint32_t part_end = (L_sub - processed > GATHER_PART_RECS)
            ? processed + GATHER_PART_RECS : L_sub;
        uint32_t nbrec = 0;
        uint32_t brec_out_g[REC_BATCH];
        int32_t sorted_id_page_cached = -1;
#if defined(FUSE_CULL) && FUSE_CULL
        // Task #90: gather the part into the L1 slab (record out_g - part_start
        // at slab + 32 * (out_g - part_start)), cull it, then write it in
        // SLAB_PAGE_BYTES pages (part_start is a multiple of GATHER_PART_RECS,
        // so the part starts on a payload page).
        const uint32_t part_start = processed;
        const uint32_t gslab = get_write_ptr(CB_SLAB);
#endif
        auto flush_brec_batch = [&]() {
            if (nbrec == 0) return;
            noc_async_read_barrier();
            for (uint32_t b = 0; b < nbrec; ++b) {
                const uint32_t slot = rec_l1 + b * PAGE_BYTES;
                auto aos = reinterpret_cast<volatile uint32_t*>(slot);
                // Pack into CB_PACK (not slot+32): blendrec aos[8]/aos[9] live in
                // the upper 32B of the 64B page and overlap PACK2 splat[0..1].
#if defined(FUSE_CULL) && FUSE_CULL
                auto splat = reinterpret_cast<volatile uint32_t*>(
                    gslab + (brec_out_g[b] - part_start) * L1_SPLAT_BYTES);
#else
                auto splat = reinterpret_cast<volatile uint32_t*>(pack_l1);
#endif
                // Tile-local mean fl(m - tile origin) via the integer sub_int
                // (bit-exact, no __subsf3 on NCRISC); float only outside its range.
                // Load every needed word before the first store (see
                // permute_records).
                const uint32_t a0 = aos[0], a1 = aos[1], a2 = aos[2], a3 = aos[3];
                const uint32_t a4 = aos[4], a9 = aos[9], a10 = aos[10], a11 = aos[11];
                uint32_t mxb, myb;
                if (!sort_bin_fp32::sub_int(a3, tx * TILE_SIZE, &mxb))
                    mxb = f_to_bits(bits_to_f(a3) - tx_tile);
                if (!sort_bin_fp32::sub_int(a4, ty * TILE_SIZE, &myb))
                    myb = f_to_bits(bits_to_f(a4) - ty_tile);
                splat[0] = a0;
                splat[1] = a1;
                splat[2] = a2;
                splat[3] = a9;
                splat[4] = mxb;
                splat[5] = myb;
                // iter 132: op/color UNORM16 are packed ONCE per gaussian on the
                // NCRISC side (sort_bin pack_invariants) and published into
                // blendrec[10],[11] (via a full-64B page write-back); this depth-
                // sorted overflow gather only COPIES the two packed words from the
                // 64B blendrec page it already reads — NO re-pack and NO extra read.
                // Bit-identical to iter-131's birth pack (same fp32 inputs, same
                // rounding). iter-131 ablation MEASURED the re-pack at ~5.3 ms/view
                // busiest-core (frame BRISC-FW -7.1) — the dominant overflow-gather
                // cost, now eliminated off the long pole.
                splat[6] = a10;
                splat[7] = a11;
#if !(defined(FUSE_CULL) && FUSE_CULL)
                const uint32_t out_g = brec_out_g[b];
                const uint32_t out_page = sc_page + (out_g / SLAB_RECS_PER_PAGE);
                const uint32_t out_off = (out_g % SLAB_RECS_PER_PAGE) * L1_SPLAT_BYTES;
                noc_async_write(
                    pack_l1,
                    get_noc_addr(out_page, payload_acc) + out_off,
                    L1_SPLAT_BYTES);
#endif
            }
#if !(defined(FUSE_CULL) && FUSE_CULL)
            noc_async_write_barrier();
#endif
            nbrec = 0;
        };
        while (processed < part_end) {
            const uint32_t global_idx = id_start_sc + processed;
            const uint32_t id_page = global_idx >> 4;
            const uint32_t id_ip = global_idx & 0xF;
            if (static_cast<int32_t>(id_page) != sorted_id_page_cached) {
                noc_async_read(get_noc_addr(id_page, sorted_acc), ids_scr, PAGE_BYTES);
                noc_async_read_barrier();
                sorted_id_page_cached = static_cast<int32_t>(id_page);
            }
            uint32_t take = ELEMS_PER_PAGE - id_ip;
            if (take > part_end - processed) take = part_end - processed;
            for (uint32_t j = 0; j < take; ++j) {
                const uint32_t gid = idsp[id_ip + j];
                const uint32_t slot = rec_l1 + nbrec * PAGE_BYTES;
                noc_async_read(get_noc_addr(gid, blendrec_acc), slot, PAGE_BYTES);
                brec_out_g[nbrec] = processed + j;
                nbrec++;
                if (nbrec == REC_BATCH) flush_brec_batch();
            }
            processed += take;
        }
        flush_brec_batch();
#if defined(FUSE_CULL) && FUSE_CULL
        {
            const uint32_t n = part_end - part_start;
            cull_slab(gslab, n);
            const uint32_t page0 = sc_page + part_start / SLAB_RECS_PER_PAGE;
            const uint32_t out_pages = (n + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
            for (uint32_t p = 0; p < out_pages; ++p) {
                const uint32_t recs = (p + 1u < out_pages) ? SLAB_RECS_PER_PAGE
                                                           : (n - p * SLAB_RECS_PER_PAGE);
                noc_async_write(gslab + p * SLAB_PAGE_BYTES,
                                get_noc_addr(page0 + p, payload_acc), recs * L1_SPLAT_BYTES);
            }
            noc_async_write_barrier();
        }
#endif
    }
#if defined(FUSE_CULL) && FUSE_CULL
    cull_end_stream();
#endif
}
