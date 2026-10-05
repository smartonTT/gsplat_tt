// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-local L1 microblock-cull READER (iter 101 / M2).
//
// For each LPT tile (and each post-sort subchunk) this reader:
//   1. reads the dir entry (dir_base+sc -> payload_page) for the subchunk,
//   2. bulk-DMAs the ALREADY depth-sorted PACK2 slab (sort_subchunk_payload)
//      into L1 in one batched read — no l1_recs reload, no L1 re-sort, no
//      blendrec gather (the slab is the single depth-sorted source of truth),
//   3. streams SFPU cull coeff rows {cov, image-space center, opacity, thr}
//      from the L1 slab copy in slab order (depth-rank k == slab record k).
//
// Pairs with microblock_band_cull_compute + writer_tile_l1_mask. Task #59: the
// compute no longer reads the slab; the reader transposes each 128-record batch
// into one fp32 coefficient tile (one gaussian per SFPU lane, see
// fill_coeff_tile) and hands the slab slot itself to the writer, which patches
// the mask into word3 in L1 and writes the slab back.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_COEFF      = 2;   // task #59: fp32 coefficient tiles -> compute
constexpr uint32_t CB_CULL_COUNTS= 3;
constexpr uint32_t CB_SCR_IDS    = 4;
constexpr uint32_t CB_SCR_ATTR   = 5;
constexpr uint32_t CB_CORE_TILES = 7;
constexpr uint32_t CB_BUCKET     = 8;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t CHUNK_MAX = 16;

constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
constexpr uint32_t GATHER_SLOT_BYTES = 64u;
// iter 110 (A2): the depth-sorted slab uses a LARGE DRAM interleave page so the
// per-subchunk bulk load is ceil(L/SLAB_RECS_PER_PAGE) big NoC transfers (was
// ~4096 per-64B-page reads). The slab is a contiguous array of 32B records, so
// each big page maps to a contiguous L1 span and record k still lands at
// buck + k*32 (depth-rank alignment with the blend reader is preserved).
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / L1_SPLAT_BYTES;  // 64

// iter 109: FIXED-SIZE bulk CB slot (mirror the blend reader/compute bulk hand-
// off). CB_BUCKET is circular + accessed with LINEAR pointer arithmetic over a
// whole subchunk's multi-page span, so a wrapping reservation would corrupt the
// tail. A full subchunk is exactly BULK_REC_SLOT pages; the CB is sized as 2
// slots (depth == MB_BUCKET_FIT) so the reader can prefetch subchunk N+1 while
// the cull compute drains subchunk N — no ring straddle.
constexpr uint32_t BULK_REC_SLOT = (MB_BUCKET_FIT + 1u) >> 1;

// Task #59 band cull: one fp32 coefficient tile per COEFF_BATCH records.
constexpr uint32_t COEFF_BATCH = 128u;        // 4 groups x 32 SFPU lanes
constexpr uint32_t OPQ_BIAS = 0x4B000000u;    // fp32 2^23; field = 2^23 + UNORM16 opacity

// Record i of the batch is gaussian i: group g = i/32 (SFPU vectors 6g..6g+5),
// lane l = i%32. Lane l of vector V is tile word 64*(V>>1) + (V&1) + 2l (the
// copy_tile / SFPU layout perm() in writer_microblock_cull.cpp describes), so
// the field pairs (A,B), (C,opq), (mx,my) of a gaussian are adjacent words.
// Must match the F_* field order in microblock_band_cull_compute.cpp.
inline void fill_coeff_tile(uint32_t slab, uint32_t base, uint32_t n, uint32_t tile) {
    for (uint32_t i = 0; i < n; ++i) {
        auto src = reinterpret_cast<volatile uint32_t*>(slab + (base + i) * L1_SPLAT_BYTES);
        auto dst = reinterpret_cast<volatile uint32_t*>(tile) + 192u * (i >> 5) + 2u * (i & 31u);
        const uint32_t w0 = src[0], w1 = src[1], w2 = src[2];
        const uint32_t w4 = src[4], w5 = src[5], w6 = src[6];
        dst[0] = w0;                             // A    (vector 6g)
        dst[1] = w1;                             // B    (6g+1)
        dst[64] = w2;                            // C    (6g+2)
        dst[65] = OPQ_BIAS | (w6 & 0xffffu);     // opq  (6g+3)
        dst[128] = w4;                           // mx   (6g+4)
        dst[129] = w5;                           // my   (6g+5)
    }
}

template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    const uint32_t page = elem >> 4;
    const uint32_t off = elem & 0xF;
    noc_async_read_tile(page, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[off];
}

// L1 store/visibility fence: a producer's stores reach the write-through L1 but
// a `fence` is needed so the freshly DMA'd slab is coherent before cb_push_back
// signals the consumer (and so the compiler does not reorder the bulk reads past
// the push). == invalidate_l1_cache() the runtime uses for cross-proc handoff.
inline void mb_cb_commit_fence() {
    asm volatile("fence" ::: "memory");
}

// Task #86: fine per-subchunk zones for attribution (host env
// GSPLAT_TT_MATCULL_PROF=1; compiled out by default).
#if defined(MATCULL_PROF) && MATCULL_PROF
#define CULL_PZ(name) DeviceZoneScopedN(name)
#else
#define CULL_PZ(name) ((void)0)
#endif

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_l1_cull_rd");
    const uint32_t l1_recs_addr      = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr          = get_arg_val<uint32_t>(1);
    const uint32_t ranges_addr       = get_arg_val<uint32_t>(2);
    const uint32_t blendrec_addr     = get_arg_val<uint32_t>(3);
    const uint32_t bucket_meta_addr  = get_arg_val<uint32_t>(4);
    const uint32_t subchunk_meta_addr= get_arg_val<uint32_t>(5);
    const uint32_t cull_base_addr    = get_arg_val<uint32_t>(6);
    const uint32_t box_ox_addr       = get_arg_val<uint32_t>(7);
    const uint32_t box_oy_addr       = get_arg_val<uint32_t>(8);
    const uint32_t tile_ids_addr     = get_arg_val<uint32_t>(9);
    const uint32_t lpt_meta_addr     = get_arg_val<uint32_t>(10);
    const uint32_t core_index        = get_arg_val<uint32_t>(11);
    const uint32_t tiles_x           = get_arg_val<uint32_t>(12);
    const uint32_t floor_bits        = get_arg_val<uint32_t>(13);
    const uint32_t subchunk_payload_addr = get_arg_val<uint32_t>(14);  // M2: sorted PACK2 slab
    const uint32_t subchunk_dir_addr     = get_arg_val<uint32_t>(15);  // M2: dir_base+sc -> page
    // iter-140 Stage-0 OVERLAP PROBE (gated, default-off). store_reps>0 injects a
    // calibrated NCRISC 32B-store loop (a faithful sort-emit store-bound proxy)
    // AFTER each CB_BUCKET push, so it runs while the SFPU (TRISC) drains the
    // just-pushed slot. Tests whether NCRISC-dataflow store work overlaps the
    // SFPU-compute stage (makespan -> max) or serializes (makespan -> sum). The
    // stores target a throwaway DRAM scratch ring (never read) -> output is
    // bit-identical. reps==0 (default) -> the loop + scratch accessor are inert.
    const uint32_t store_reps   = get_arg_val<uint32_t>(16);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(17);
    float contrib_floor;
    __builtin_memcpy(&contrib_floor, &floor_bits, 4);

    constexpr auto l1_recs_args = TensorAccessorArgs<0>();
    constexpr auto ids_args     = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    constexpr auto ranges_args  = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args= TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto bucket_meta_args = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_meta_args = TensorAccessorArgs<bucket_meta_args.next_compile_time_args_offset()>();
    constexpr auto cull_base_args = TensorAccessorArgs<subchunk_meta_args.next_compile_time_args_offset()>();
    constexpr auto bx_args        = TensorAccessorArgs<cull_base_args.next_compile_time_args_offset()>();
    constexpr auto by_args        = TensorAccessorArgs<bx_args.next_compile_time_args_offset()>();
    constexpr auto tids_args    = TensorAccessorArgs<by_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args  = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_payload_args =
        TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_dir_args =
        TensorAccessorArgs<subchunk_payload_args.next_compile_time_args_offset()>();
    // iter-140 OVERLAP PROBE: throwaway DRAM scratch ring (store-stress target).
    constexpr auto scratch_args =
        TensorAccessorArgs<subchunk_dir_args.next_compile_time_args_offset()>();

    const auto l1_recs_acc   = TensorAccessor(l1_recs_args, l1_recs_addr, L1_PACK_PAGE_BYTES);
    const auto ids_acc       = TensorAccessor(ids_args, ids_addr, IDS_PAGE_BYTES);
    const auto ranges_acc    = TensorAccessor(ranges_args, ranges_addr, SOA_PAGE_BYTES);
    const auto blendrec_acc  = TensorAccessor(blendrec_args, blendrec_addr, SOA_PAGE_BYTES);
    const auto bucket_meta_acc = TensorAccessor(bucket_meta_args, bucket_meta_addr, SOA_PAGE_BYTES);
    const auto subchunk_meta_acc = TensorAccessor(subchunk_meta_args, subchunk_meta_addr, SOA_PAGE_BYTES);
    const auto cull_base_acc = TensorAccessor(cull_base_args, cull_base_addr, SOA_PAGE_BYTES);
    const auto bx_acc        = TensorAccessor(bx_args, box_ox_addr, RAMP_TILE_BYTES);
    const auto by_acc        = TensorAccessor(by_args, box_oy_addr, RAMP_TILE_BYTES);
    const auto tids_acc      = TensorAccessor(tids_args, tile_ids_addr, IDS_PAGE_BYTES);
    const auto lpt_meta_acc  = TensorAccessor(lpt_meta_args, lpt_meta_addr, SOA_PAGE_BYTES);
    const auto subchunk_payload_acc =
        TensorAccessor(subchunk_payload_args, subchunk_payload_addr, SLAB_PAGE_BYTES);
    const auto subchunk_dir_acc =
        TensorAccessor(subchunk_dir_args, subchunk_dir_addr, SOA_PAGE_BYTES);
    // iter-140 OVERLAP PROBE: 64B-page interleaved DRAM scratch (write-only).
    constexpr uint32_t SCRATCH_PAGE_BYTES = 64u;
    constexpr uint32_t SCRATCH_RING_PAGES = 4096u;  // per-core ring (256 KB/core)
    const auto scratch_acc =
        TensorAccessor(scratch_args, scratch_addr, SCRATCH_PAGE_BYTES);
    const uint32_t scratch_ring_base = core_index * SCRATCH_RING_PAGES;
    uint32_t store_phase = 0;
    // M2: l1_recs / ids / blendrec / bucket_meta are dead on the cull hot path
    // (the slab is the depth-sorted source of truth). Bindings kept for ABI
    // parity; cleaned up in M4.
    // iter 109: the reader no longer derives the per-record cull row (op/center);
    // the cull compute reads the raw slab record and derives them. contrib_floor
    // is unused on the reader now (the SFPU thr uses the compute's floor arg).
    (void)contrib_floor;
    (void)l1_recs_acc;
    (void)ids_acc;
    (void)blendrec_acc;
    (void)bucket_meta_acc;

    constexpr uint32_t META_ELEMS_PER_PAGE = 16u;
    const uint32_t meta_elem0 = core_index * 2u;
    const uint32_t meta_page0 = meta_elem0 / META_ELEMS_PER_PAGE;
    const uint32_t meta_ip0   = meta_elem0 % META_ELEMS_PER_PAGE;
    const uint32_t meta_scratch = get_write_ptr(CB_SCR_IDS);
    auto meta_ptr = reinterpret_cast<volatile uint32_t*>(meta_scratch);
    noc_async_read(get_noc_addr(meta_page0, lpt_meta_acc), meta_scratch, 64);
    noc_async_read_barrier();
    uint32_t tile_ids_start = meta_ptr[meta_ip0];
    uint32_t tile_ids_count = 0;
    if (meta_ip0 + 1u < META_ELEMS_PER_PAGE) {
        tile_ids_count = meta_ptr[meta_ip0 + 1u];
    } else {
        noc_async_read(get_noc_addr(meta_page0 + 1u, lpt_meta_acc), meta_scratch, 64);
        noc_async_read_barrier();
        tile_ids_count = meta_ptr[0];
    }

    if (tile_ids_count == 0) {
        cb_reserve_back(CB_CORE_TILES, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CORE_TILES))[0] = 0u;
        cb_push_back(CB_CORE_TILES, 1);
        return;
    }

    // Task #59: the band cull needs no box-origin ramps (bx/by stay bound for ABI).
    (void)bx_acc;
    (void)by_acc;

    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t scratch_addr = get_write_ptr(CB_SCR_IDS);
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

    uint32_t num_work = 0;
    for (uint32_t ti = 0; ti < tile_ids_count; ++ti) {
        const uint32_t tile_id = tile_ids[ti];
        const uint32_t scr = get_write_ptr(CB_SCR_IDS);
        noc_async_read_tile((tile_id * 2u) >> 4, subchunk_meta_acc, scr);
        noc_async_read_barrier();
        const uint32_t meta_off = (tile_id * 2u) & 0xF;
        uint32_t nsc = reinterpret_cast<volatile uint32_t*>(scr)[meta_off + 1u];
        if (nsc == 0u) nsc = 1u;
        num_work += nsc;
    }

    cb_reserve_back(CB_CORE_TILES, 1);
    reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CORE_TILES))[0] = num_work;
    cb_push_back(CB_CORE_TILES, 1);

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;

        uint32_t id_start, id_end;
        {
            CULL_PZ("cr_meta");
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            const uint32_t elem0 = tile_id * 2u;
            const uint32_t page = elem0 >> 4;
            const uint32_t off = elem0 & 0xF;
            auto rng_ptr = reinterpret_cast<volatile uint32_t*>(scr);
            noc_async_read_tile(page, ranges_acc, scr);
            noc_async_read_barrier();
            id_start = rng_ptr[off];
            if (off + 1u < 16u) {
                id_end = rng_ptr[off + 1u];
            } else {
                noc_async_read_tile(page + 1u, ranges_acc, scr);
                noc_async_read_barrier();
                id_end = rng_ptr[0];
            }
        }
        const uint32_t L = id_end - id_start;
        CULL_PZ("cr_tile");
        const uint32_t cull_base = read_soa_u32(cull_base_acc, tile_id, get_write_ptr(CB_SCR_IDS));

        // blend_subchunk_meta: per-tile (dir_base, num_sc) pair. dir_base indexes
        // sort_subchunk_dir; mirror the blend reader exactly so the cull slab load
        // hits the SAME payload pages.
        uint32_t dir_base = 0;
        uint32_t num_subchunks = 1;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            noc_async_read_tile(pg, subchunk_meta_acc, scr);
            noc_async_read_barrier();
            auto smp = reinterpret_cast<volatile uint32_t*>(scr);
            dir_base = smp[off];
            num_subchunks = smp[off + 1u];
            if (num_subchunks == 0u) num_subchunks = 1u;
        }

        for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
            const uint32_t sc_off = sc * MB_BUCKET_FIT;
            const uint32_t L_sub = (sc_off >= L) ? 0u
                : ((L - sc_off > MB_BUCKET_FIT) ? MB_BUCKET_FIT : (L - sc_off));
            const uint32_t cull_base_sc = cull_base + sc_off;

            cb_reserve_back(CB_CULL_COUNTS, 1);
            {
                auto cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COUNTS));
                cnt[0] = L_sub;
                cnt[1] = tx * TILE_SIZE;
                cnt[2] = ty * TILE_SIZE;
                cnt[3] = cull_base_sc;
            }
            cb_push_back(CB_CULL_COUNTS, 1);

            if (L_sub == 0) {
                continue;
            }

            // iter 109: read the dir entry (dir_base+sc -> payload_page) and
            // bulk-DMA the already-depth-sorted PACK2 slab into CB_BUCKET in one
            // batched read (mirrors the blend reader's CB_BUCKET_BULK hand-off).
            // Depth-rank k == slab record k stays aligned with the blend reader.
            uint32_t payload_page = 0;
            {
                CULL_PZ("cr_meta");
                const uint32_t de = (dir_base + sc) * 4u;
                const uint32_t dpg = de >> 4;
                const uint32_t dof = de & 0xF;
                const uint32_t scr = get_write_ptr(CB_SCR_IDS);
                noc_async_read_tile(dpg, subchunk_dir_acc, scr);
                noc_async_read_barrier();
                payload_page = reinterpret_cast<volatile uint32_t*>(scr)[dof];
            }

            const uint32_t rec_pages =
                (L_sub + SLAB_RECS_PER_PAGE - 1u) / SLAB_RECS_PER_PAGE;
            {
                CULL_PZ("cr_slot_wait");
                cb_reserve_back(CB_BUCKET, BULK_REC_SLOT);
            }
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                CULL_PZ("cr_bulk");
                const uint32_t page0 = payload_page;
                uint32_t pp = 0;
                while (pp < rec_pages) {
                    const uint32_t end = (pp + 64u < rec_pages) ? pp + 64u : rec_pages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(page0 + q, subchunk_payload_acc,
                                            buck + q * SLAB_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
            }
            mb_cb_commit_fence();
            // Task #59: the slot goes to the writer (mask -> word3, slab write-back);
            // it only touches word3, the transpose below only reads the others.
            cb_push_back(CB_BUCKET, BULK_REC_SLOT);
            CULL_PZ("cr_fill");
            for (uint32_t base = 0; base < L_sub; base += COEFF_BATCH) {
                const uint32_t n = (L_sub - base < COEFF_BATCH) ? (L_sub - base) : COEFF_BATCH;
                cb_reserve_back(CB_COEFF, 1);
                fill_coeff_tile(buck, base, n, get_write_ptr(CB_COEFF));
                mb_cb_commit_fence();
                cb_push_back(CB_COEFF, 1);
            }

            // iter-140 Stage-0 OVERLAP PROBE: store-stress on NCRISC. With the
            // just-pushed slot now draining on the SFPU (TRISC), issue store_reps
            // 32B writes (sort-emit store-bound proxy) into the throwaway DRAM
            // scratch ring. If NCRISC stores overlap the SFPU cull, the program
            // makespan stays ~max(store,SFPU); if they serialize (NoC/L1
            // contention), it grows toward sum. Single in-order CQ, no CB-handshake
            // change, no Finish -> hang-safe. reps==0 -> inert (production path).
            if (store_reps > 0u) {
                DeviceZoneScopedN("ovlp_store");
                const uint32_t src = get_write_ptr(CB_SCR_ATTR);
                uint32_t issued = 0;
                for (uint32_t r = 0; r < store_reps; ++r) {
                    const uint32_t pg =
                        scratch_ring_base + (store_phase % SCRATCH_RING_PAGES);
                    noc_async_write(src, get_noc_addr(pg, scratch_acc), L1_SPLAT_BYTES);
                    ++store_phase;
                    if (++issued == 64u) { noc_async_write_barrier(); issued = 0; }
                }
                if (issued) noc_async_write_barrier();
            }
        }
    }
}
