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
// Pairs with microblock_cull_compute + writer_tile_l1_mask; masks land in DRAM
// cull_masks at cull_base+k, staying aligned with the blend reader (which reads
// cull_masks[cull_base+k] == depth-rank k == slab record k).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_BOX_OX     = 0;
constexpr uint32_t CB_BOX_OY     = 1;
constexpr uint32_t CB_CULL_COEFF = 2;
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

// M5 (iter 104): double-buffer prefetch. The cull reader is the SOLE driver of
// both the slab DMA and the coeff-row emit; on one RISC those serialize (DMA(N)
// runs with compute idle, then the emit(N) spins on CB_CULL_COEFF backpressure
// with the NoC idle == DMA+SFPU). We hide the slab DMA behind the SFPU by (a) a
// metadata PRE-PASS (so the hot loop issues NO metadata reads whose global
// barrier would prematurely drain the in-flight slab DMA) that records every
// non-empty subchunk's {payload_page, L_sub, tx, ty} and pushes ALL counts in
// order, then (b) a 2-slot ping-pong over CB_BUCKET that interleaves the
// slab(N+1) read-issues INTO the compute-throttled emit(N) loop so the NoC is
// busy exactly when the reader would otherwise spin. CB_BUCKET is 2*BULK_REC
// pages (slot-aligned, never ring-straddles; preserves the M1b invariant).
// MAX_WORK bounds the pre-pass arrays (CB_SCR_ATTR) + CB_CULL_COUNTS depth; if a
// core's work count exceeds it we fall back to the proven synchronous loop.
constexpr uint32_t MAX_WORK = 2048u;  // >= per-core 256 tiles * <=8 subchunks
constexpr uint32_t BULK_REC_SLOT = (MB_BUCKET_FIT + 1u) >> 1;  // pages per slot

inline volatile uint32_t* l1_splat_words(uint32_t buck_base, uint32_t g) {
    return reinterpret_cast<volatile uint32_t*>(
        buck_base + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
}

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

template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    const uint32_t page = elem >> 4;
    const uint32_t off = elem & 0xF;
    noc_async_read_tile(page, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[off];
}

inline void emit_cull_row_from_l1_splat(
    volatile uint32_t* recp32, float tx_tile, float ty_tile, float contrib_floor) {
    cb_reserve_back(CB_CULL_COEFF, 1);
    auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COEFF));
    row[0] = recp32[0];
    row[1] = recp32[1];
    row[2] = recp32[2];
    const float mx_local = bits_to_f(recp32[4]);
    const float my_local = bits_to_f(recp32[5]);
    row[3] = f_to_bits(mx_local + tx_tile);
    row[4] = f_to_bits(my_local + ty_tile);
    constexpr float kUnormInv = 1.0f / 65535.0f;
    const uint32_t w6 = recp32[6];
    const float opf = static_cast<float>(w6 & 0xffffu) * kUnormInv;
    row[5] = f_to_bits(opf);
    float thrf;
    if (opf <= contrib_floor) {
        thrf = -1.0f;
    } else {
        thrf = -2.0f * __builtin_logf(contrib_floor / opf);
    }
    row[6] = f_to_bits(thrf);
    cb_push_back(CB_CULL_COEFF, 1);
}

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
        TensorAccessor(subchunk_payload_args, subchunk_payload_addr, L1_PACK_PAGE_BYTES);
    const auto subchunk_dir_acc =
        TensorAccessor(subchunk_dir_args, subchunk_dir_addr, SOA_PAGE_BYTES);
    // M2: l1_recs / ids / blendrec / bucket_meta are dead on the cull hot path
    // (the slab is the depth-sorted source of truth). Bindings kept for ABI
    // parity; cleaned up in M4.
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

    cb_reserve_back(CB_BOX_OX, 1);
    noc_async_read_tile(0, bx_acc, get_write_ptr(CB_BOX_OX));
    cb_reserve_back(CB_BOX_OY, 1);
    noc_async_read_tile(0, by_acc, get_write_ptr(CB_BOX_OY));
    noc_async_read_barrier();
    cb_push_back(CB_BOX_OX, 1);
    cb_push_back(CB_BOX_OY, 1);

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

    // ==================================================================
    // M5: software-pipelined (double-buffered) slab load.
    // Pre-pass: push ALL counts in order + record non-empty descriptors.
    // Pipeline: 2-slot ping-pong over CB_BUCKET; slab(N+1) reads are issued
    // INTERLEAVED with the compute-throttled emit(N) so the slab DMA overlaps
    // the cull SFPU instead of running in front of it. MAX_WORK bounds the
    // pre-pass arrays (per-core <=256 tiles * <=8 subchunks <= 2048).
    // ==================================================================
    {
        const uint32_t WBASE = get_write_ptr(CB_SCR_ATTR);
        volatile uint32_t* w_pp = reinterpret_cast<volatile uint32_t*>(WBASE);
        volatile uint32_t* w_ls = reinterpret_cast<volatile uint32_t*>(WBASE + 1u * MAX_WORK * 4u);
        volatile uint32_t* w_tx = reinterpret_cast<volatile uint32_t*>(WBASE + 2u * MAX_WORK * 4u);
        volatile uint32_t* w_ty = reinterpret_cast<volatile uint32_t*>(WBASE + 3u * MAX_WORK * 4u);

        // ---- metadata pre-pass: small NoC reads (each self-barriered, no slab
        //      DMA in flight) so the pipeline below has ZERO metadata barriers. ----
        uint32_t m = 0;  // non-empty work items recorded
        for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
            const uint32_t tile_id = tile_ids[ti];
            const uint32_t tx = tile_id % tiles_x;
            const uint32_t ty = tile_id / tiles_x;
            const float tx_tile = static_cast<float>(tx * TILE_SIZE);
            const float ty_tile = static_cast<float>(ty * TILE_SIZE);

            uint32_t id_start, id_end;
            {
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
            const uint32_t cull_base = read_soa_u32(cull_base_acc, tile_id, get_write_ptr(CB_SCR_IDS));

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

                if (L_sub == 0) continue;

                uint32_t payload_page = 0;
                {
                    const uint32_t de = (dir_base + sc) * 4u;
                    const uint32_t dpg = de >> 4;
                    const uint32_t dof = de & 0xF;
                    const uint32_t scr = get_write_ptr(CB_SCR_IDS);
                    noc_async_read_tile(dpg, subchunk_dir_acc, scr);
                    noc_async_read_barrier();
                    payload_page = reinterpret_cast<volatile uint32_t*>(scr)[dof];
                }
                w_pp[m] = payload_page;
                w_ls[m] = L_sub;
                w_tx[m] = f_to_bits(tx_tile);
                w_ty[m] = f_to_bits(ty_tile);
                ++m;
            }
        }

        // ---- double-buffered slab pipeline over the m non-empty items ----
        const uint32_t buck0 = get_write_ptr(CB_BUCKET);
        const uint32_t SLOT_BYTES = BULK_REC_SLOT * L1_PACK_PAGE_BYTES;
        if (m > 0) {
            // Prefetch slab(0) into slot 0 (this first DMA is unavoidably exposed).
            {
                const uint32_t rp = (w_ls[0] + 1u) >> 1;
                const uint32_t pg0 = w_pp[0];
                for (uint32_t q = 0; q < rp; ++q) {
                    noc_async_read_tile(pg0 + q, subchunk_payload_acc, buck0 + q * L1_PACK_PAGE_BYTES);
                }
            }
            for (uint32_t e = 0; e < m; ++e) {
                noc_async_read_barrier();  // slab(e) ready (only outstanding read)
                const uint32_t srcp = buck0 + (e & 1u) * SLOT_BYTES;
                const uint32_t Lp = w_ls[e];
                const float txp = bits_to_f(w_tx[e]);
                const float typ = bits_to_f(w_ty[e]);

                const bool has_next = (e + 1u < m);
                const uint32_t rpn = has_next ? ((w_ls[e + 1u] + 1u) >> 1) : 0u;
                const uint32_t pg0n = has_next ? w_pp[e + 1u] : 0u;
                const uint32_t dstn = buck0 + ((e + 1u) & 1u) * SLOT_BYTES;

                // Emit slab(e)'s coeff rows while issuing slab(e+1)'s reads
                // EVENLY spread across the emit (target = rpn*(k+1)/Lp). The emit
                // spins on CB_CULL_COEFF backpressure at the compute's SFPU pace,
                // so spacing the slab(e+1) reads across it keeps the NoC busy in
                // exactly that otherwise-idle window -> DMA hides behind SFPU.
                uint32_t issued = 0;
                for (uint32_t k = 0; k < Lp; ++k) {
                    const uint32_t target = (rpn * (k + 1u)) / Lp;
                    while (issued < target) {
                        noc_async_read_tile(pg0n + issued, subchunk_payload_acc,
                                            dstn + issued * L1_PACK_PAGE_BYTES);
                        ++issued;
                    }
                    emit_cull_row_from_l1_splat(l1_splat_words(srcp, k), txp, typ, contrib_floor);
                }
                while (issued < rpn) {
                    noc_async_read_tile(pg0n + issued, subchunk_payload_acc,
                                        dstn + issued * L1_PACK_PAGE_BYTES);
                    ++issued;
                }
            }
        }
    }
}
