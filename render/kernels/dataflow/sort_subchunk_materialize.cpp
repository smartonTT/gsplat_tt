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
#include "rec16_codec.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
// R16 (task #23): the emit writes 16B bucket records, four per 64B page (PACK4:
// record i at page i/4, byte 16*(i&3), i.e. byte 16*i of a coalesced page run).
// The depth-sorted slab this kernel writes keeps the 32B layout the cull/blend
// kernels read; expand_rec16 converts one record.
constexpr uint32_t L1_REC16_BYTES = 16u;
// iter 110 (A2): the depth-sorted slab is materialized into a DRAM buffer with a
// LARGE interleave page (SLAB_PAGE_BYTES) so the cull/blend readers coalesce the
// per-subchunk load into ceil(L/SLAB_RECS_PER_PAGE) big transfers. The slab is
// still a contiguous array of 32B records: subchunk-local record g lives at
// page (sc_page + g/SLAB_RECS_PER_PAGE), byte (g % SLAB_RECS_PER_PAGE)*32. The
// source L1 bucket layout is PACK4 (4 x 16B records / 64B page, R16).
constexpr uint32_t SLAB_PAGE_BYTES = 2048u;
constexpr uint32_t SLAB_RECS_PER_PAGE = SLAB_PAGE_BYTES / L1_SPLAT_BYTES;  // 64
constexpr uint32_t TILE_SIZE = 32u;
// iter 76: larger blendrec gather batches (fewer read/write barriers on sc>=1).
constexpr uint32_t REC_BATCH = 32u;

constexpr uint32_t CB_SCR = 0;
constexpr uint32_t CB_IDS = 1;
constexpr uint32_t CB_REC = 2;
constexpr uint32_t CB_PACK = 3;
constexpr uint32_t CB_BUCKET = 4;
constexpr uint32_t CB_BSORT = 5;
// iter 113 (sort Stage 1): contiguous L1 scratch the depth permutation lands in
// (record k at byte k*32) so the depth-sorted slab is written to DRAM in
// coalesced SLAB_PAGE_BYTES pages instead of bucket_fit per-record 32B writes.
constexpr uint32_t CB_SLAB = 6;

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

// 16B bucket record (rec16_codec.h) -> 32B slab record at dst_l1.
inline void expand_rec16(uint32_t dst_l1, uint32_t src_l1, uint32_t key) {
    auto src = reinterpret_cast<volatile uint32_t*>(src_l1);
    auto dst = reinterpret_cast<volatile uint32_t*>(dst_l1);
    const uint32_t w[4] = {src[0], src[1], src[2], src[3]};
    uint32_t out[8];
    rec16::expand(w, key, out);
    dst[0] = out[0]; dst[1] = out[1]; dst[2] = out[2]; dst[3] = out[3];
    dst[4] = out[4]; dst[5] = out[5]; dst[6] = out[6]; dst[7] = out[7];
}

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
    // iter-138 (Stage-2b overflow pre-pack): the compact overflow region (PACK4,
    // pre-packed by sort_bucket_emit) + per-tile start slot (sentinel 0xFFFFFFFF
    // for non-prepacked tiles) + the L1 cap. For an in-cap overflow tile this core
    // gets ONE work item (sc==0) and processes the WHOLE tile: coalesced bucket
    // read + L1 radix depth-permute + per-subchunk slab emit — NO blendrec gather.
    const uint32_t ov_recs_addr   = get_arg_val<uint32_t>(12);  // overflow region (0=off)
    const uint32_t ov_base_addr   = get_arg_val<uint32_t>(13);  // per-tile slot base (0=off)
    const uint32_t ov_cap         = get_arg_val<uint32_t>(14);  // kOverflowL1Cap
    // R16: the depth keys of the bucket/overflow records, read from the emit's
    // page-aligned keys layout (tile t = pages [pstart_page, +tile_pad/16), core-
    // major per-(core,tile) blocks padded with 0xFFFFFFFF). Dropping the padding
    // yields the tile's keys in exactly the bucket slot order.
    const uint32_t keys_addr      = get_arg_val<uint32_t>(15);
    const uint32_t tmeta_addr     = get_arg_val<uint32_t>(16);  // (pstart_page, tile_pad)

    constexpr auto sorted_args = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<sorted_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto payload_args = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    constexpr auto blend_meta_args = TensorAccessorArgs<payload_args.next_compile_time_args_offset()>();
    constexpr auto dir_args = TensorAccessorArgs<blend_meta_args.next_compile_time_args_offset()>();
    constexpr auto work_args = TensorAccessorArgs<dir_args.next_compile_time_args_offset()>();
    // iter-138: overflow region (PACK4 64B page) + per-tile overflow base row.
    constexpr auto ov_recs_args = TensorAccessorArgs<work_args.next_compile_time_args_offset()>();
    constexpr auto ov_base_args = TensorAccessorArgs<ov_recs_args.next_compile_time_args_offset()>();
    constexpr auto keys_args = TensorAccessorArgs<ov_base_args.next_compile_time_args_offset()>();
    constexpr auto tmeta_args = TensorAccessorArgs<keys_args.next_compile_time_args_offset()>();

    const auto sorted_acc   = TensorAccessor(sorted_args,   sorted_addr,   PAGE_BYTES);
    const auto ranges_acc   = TensorAccessor(ranges_args,   ranges_addr,   PAGE_BYTES);
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    const auto l1_recs_acc  = TensorAccessor(l1_recs_args,  l1_recs_addr,  L1_PACK_PAGE_BYTES);
    const auto payload_acc  = TensorAccessor(payload_args,  payload_addr,  SLAB_PAGE_BYTES);
    const auto blend_meta_acc = TensorAccessor(blend_meta_args, blend_meta_addr, PAGE_BYTES);
    const auto dir_acc      = TensorAccessor(dir_args,      dir_addr,      PAGE_BYTES);
    const auto work_acc     = TensorAccessor(work_args,     work_addr,     PAGE_BYTES);
    const bool ov_enabled   = (ov_recs_addr != 0u) && (ov_base_addr != 0u);
    const auto ov_recs_acc  = TensorAccessor(ov_recs_args,  ov_recs_addr,  L1_PACK_PAGE_BYTES);
    const auto ov_base_acc  = TensorAccessor(ov_base_args,  ov_base_addr,  PAGE_BYTES);
    const auto keys_acc     = TensorAccessor(keys_args,     keys_addr,     PAGE_BYTES);
    const auto tmeta_acc    = TensorAccessor(tmeta_args,    tmeta_addr,    PAGE_BYTES);

    if (work_count == 0) {
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    auto scrp = reinterpret_cast<volatile uint32_t*>(scr);
    const uint32_t ids_scr = get_write_ptr(CB_IDS);
    auto idsp = reinterpret_cast<volatile uint32_t*>(ids_scr);
    const uint32_t rec_l1 = get_write_ptr(CB_REC);
    const uint32_t pack_l1 = get_write_ptr(CB_PACK);

    // R16: fill keyc[0..n) with tile_id's depth keys in bucket slot order (see
    // keys_addr above). keyc sits in the upper half of CB_BUCKET: the PACK4
    // records use at most ov_cap*16 B of its max(bucket_fit*64, ov_cap*32) B.
    auto load_tile_keys = [&](uint32_t tile_id, uint32_t keyc_l1, uint32_t n) {
        const uint32_t e0 = tile_id * 2u;
        noc_async_read(get_noc_addr(e0 >> 4, tmeta_acc), scr, PAGE_BYTES);
        noc_async_read_barrier();
        const uint32_t pstart = scrp[e0 & 0xFu];
        const uint32_t npages = scrp[(e0 & 0xFu) + 1u] / ELEMS_PER_PAGE;
        uint32_t pp = 0;
        while (pp < npages) {
            const uint32_t end = (pp + 64u < npages) ? pp + 64u : npages;
            for (uint32_t q = pp; q < end; ++q) {
                noc_async_read(get_noc_addr(pstart + q, keys_acc),
                               keyc_l1 + q * PAGE_BYTES, PAGE_BYTES);
            }
            noc_async_read_barrier();
            pp = end;
        }
        auto kp = reinterpret_cast<volatile uint32_t*>(keyc_l1);
        const uint32_t total = npages * ELEMS_PER_PAGE;
        uint32_t j = 0;
        for (uint32_t i = 0; i < total; ++i) {
            const uint32_t k = kp[i];
            if (k != 0xFFFFFFFFu) kp[j++] = k;
        }
        for (; j < n; ++j) kp[j] = 0xFFFFFFFFu;  // defensive: counts always match
    };

    constexpr uint32_t MAX_WORK = 1024;
    // Packed work item: (tile_id << 8) | sc. tile_id < 2^16, sc < 2^8 — fits.
    uint32_t work_item[MAX_WORK];
    {
        // The work buffer is a flat u32 array; this core's items occupy u32
        // indices [2*work_start, 2*work_start + 2*work_count). Stream the slice
        // through CB_IDS; even u32 = tile_id, odd = sc, packed into work_item.
        const uint32_t total_u32 = work_count * 2u;
        const uint32_t u0 = work_start * 2u;
        uint32_t page_idx = u0 / ELEMS_PER_PAGE;
        uint32_t in_page  = u0 % ELEMS_PER_PAGE;
        uint32_t got = 0;
        uint32_t pend_tile = 0;
        while (got < total_u32) {
            noc_async_read(get_noc_addr(page_idx, work_acc), ids_scr, PAGE_BYTES);
            noc_async_read_barrier();
            uint32_t take = ELEMS_PER_PAGE - in_page;
            if (take > total_u32 - got) take = total_u32 - got;
            for (uint32_t i = 0; i < take; i++) {
                const uint32_t gu = got + i;        // u32 offset within this slice
                const uint32_t val = idsp[in_page + i];
                if ((gu & 1u) == 0u) pend_tile = val;
                else work_item[gu >> 1] = (pend_tile << 8) | (val & 0xFFu);
            }
            got += take;
            page_idx += 1;
            in_page = 0;
        }
    }

    for (uint32_t wi = 0; wi < work_count; wi++) {
        const uint32_t tile_id = work_item[wi] >> 8;
        const uint32_t sc = work_item[wi] & 0xFFu;
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;
        const float tx_tile = static_cast<float>(tx * TILE_SIZE);
        const float ty_tile = static_cast<float>(ty * TILE_SIZE);

        uint32_t id_start = 0, id_end = 0;
        {
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
                const uint32_t e0 = tile_id;  // 1 u32 per tile
                const uint32_t pg = e0 >> 4;
                const uint32_t off = e0 & 0xF;
                noc_async_read(get_noc_addr(pg, ov_base_acc), scr, PAGE_BYTES);
                noc_async_read_barrier();
                ov_base = scrp[off];
            }
            if (ov_base != 0xFFFFFFFFu) {
                // Coalesced read of the whole overflow bucket (PACK4 64B pages).
                const uint32_t npages = (count + 3u) >> 2;
                const uint32_t buck = get_write_ptr(CB_BUCKET);
                {
                    const uint32_t page0 = ov_base >> 2;  // ov_base is 4-aligned
                    uint32_t pp = 0;
                    while (pp < npages) {
                        const uint32_t end = (pp + 64u < npages) ? pp + 64u : npages;
                        for (uint32_t q = pp; q < end; ++q) {
                            noc_async_read_tile(
                                page0 + q, ov_recs_acc, buck + q * L1_PACK_PAGE_BYTES);
                        }
                        noc_async_read_barrier();
                        pp = end;
                    }
                }
                const uint32_t keyc_l1 = buck + ov_cap * L1_REC16_BYTES;
                load_tile_keys(tile_id, keyc_l1, count);
                const uint32_t* keyc = reinterpret_cast<const uint32_t*>(keyc_l1);
                // Stable LSD radix sort over ALL `count` records by depth key.
                const uint32_t bs = get_write_ptr(CB_BSORT);
                uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
                uint32_t* idxB = idxA + ov_cap;
                uint32_t* cnt  = idxB + ov_cap;
                for (uint32_t i = 0; i < count; ++i) idxA[i] = i;
                uint32_t* cur = idxA;
                uint32_t* nxt = idxB;
                for (uint32_t byte = 0; byte < 4u; ++byte) {
                    const uint32_t shift = byte * 8u;
                    for (uint32_t c = 0; c < 256u; ++c) cnt[c] = 0;
                    for (uint32_t i = 0; i < count; ++i) {
                        cnt[(keyc[cur[i]] >> shift) & 0xFFu]++;
                    }
                    uint32_t sum = 0;
                    for (uint32_t c = 0; c < 256u; ++c) {
                        const uint32_t t = cnt[c];
                        cnt[c] = sum;
                        sum += t;
                    }
                    for (uint32_t i = 0; i < count; ++i) {
                        const uint32_t b = (keyc[cur[i]] >> shift) & 0xFFu;
                        nxt[cnt[b]++] = cur[i];
                    }
                    uint32_t* tp = cur;
                    cur = nxt;
                    nxt = tp;
                }
                uint32_t* sorted = cur;
                // Emit each subchunk's depth-sorted slab to its directory page run.
                const uint32_t num_sc = (count + bucket_fit - 1u) / bucket_fit;
                const uint32_t slab = get_write_ptr(CB_SLAB);
                for (uint32_t s = 0; s < num_sc; ++s) {
                    const uint32_t sc_off2 = s * bucket_fit;
                    const uint32_t Ls = (count - sc_off2 > bucket_fit)
                        ? bucket_fit : (count - sc_off2);
                    uint32_t scp = 0;
                    {
                        const uint32_t e0 = (dir_base + s) * 4u;
                        const uint32_t pg = e0 >> 4;
                        const uint32_t off = e0 & 0xF;
                        noc_async_read(get_noc_addr(pg, dir_acc), scr, PAGE_BYTES);
                        noc_async_read_barrier();
                        scp = scrp[off];
                    }
                    for (uint32_t k = 0; k < Ls; ++k) {
                        const uint32_t idx = sorted[sc_off2 + k];
                        expand_rec16(slab + k * L1_SPLAT_BYTES,
                                     buck + idx * L1_REC16_BYTES, keyc[idx]);
                    }
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
            const uint32_t npages = (L + 3u) >> 2;
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                const uint32_t page0 = tile_id * (bucket_fit >> 2);
                uint32_t pp = 0;
                while (pp < npages) {
                    const uint32_t end = (pp + 64u < npages) ? pp + 64u : npages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(
                            page0 + q, l1_recs_acc, buck + q * L1_PACK_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
            }
            const uint32_t keyc_l1 = buck + ov_cap * L1_REC16_BYTES;
            load_tile_keys(tile_id, keyc_l1, L);
            const uint32_t* keyc = reinterpret_cast<const uint32_t*>(keyc_l1);
            const uint32_t bs = get_write_ptr(CB_BSORT);
            uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
            uint32_t* idxB = idxA + bucket_fit;
            uint32_t* cnt  = idxB + bucket_fit;
            uint32_t* sorted;
            if (L <= 16u) {
                for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
                for (uint32_t i = 1; i < L; ++i) {
                    const uint32_t tmp = idxA[i];
                    const uint32_t ki = keyc[tmp];
                    uint32_t j = i;
                    while (j > 0 && keyc[idxA[j - 1]] > ki) {
                        idxA[j] = idxA[j - 1];
                        --j;
                    }
                    idxA[j] = tmp;
                }
                sorted = idxA;
            } else {
                for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
                uint32_t* cur = idxA;
                uint32_t* nxt = idxB;
                for (uint32_t byte = 0; byte < 4u; ++byte) {
                    const uint32_t shift = byte * 8u;
                    for (uint32_t c = 0; c < 256u; ++c) cnt[c] = 0;
                    for (uint32_t i = 0; i < L; ++i) {
                        cnt[(keyc[cur[i]] >> shift) & 0xFFu]++;
                    }
                    uint32_t sum = 0;
                    for (uint32_t c = 0; c < 256u; ++c) {
                        const uint32_t t = cnt[c];
                        cnt[c] = sum;
                        sum += t;
                    }
                    for (uint32_t i = 0; i < L; ++i) {
                        const uint32_t b = (keyc[cur[i]] >> shift) & 0xFFu;
                        nxt[cnt[b]++] = cur[i];
                    }
                    uint32_t* t = cur;
                    cur = nxt;
                    nxt = t;
                }
                sorted = cur;
            }
            // Stage 1: apply the radix permutation L1->L1 into a contiguous
            // slab scratch (output order), then emit the depth-sorted slab in
            // coalesced SLAB_PAGE_BYTES page writes (no per-record DRAM scatter).
            const uint32_t slab = get_write_ptr(CB_SLAB);
            for (uint32_t k = 0; k < L; ++k) {
                const uint32_t idx = sorted[k];
                expand_rec16(slab + k * L1_SPLAT_BYTES, buck + idx * L1_REC16_BYTES,
                             keyc[idx]);
            }
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
        const uint32_t id_start_sc = id_start + sc_off;
        uint32_t processed = 0;
        uint32_t nbrec = 0;
        uint32_t brec_out_g[REC_BATCH];
        int32_t sorted_id_page_cached = -1;
        auto flush_brec_batch = [&]() {
            if (nbrec == 0) return;
            noc_async_read_barrier();
            for (uint32_t b = 0; b < nbrec; ++b) {
                const uint32_t slot = rec_l1 + b * PAGE_BYTES;
                auto aos = reinterpret_cast<volatile uint32_t*>(slot);
                // Pack into CB_PACK (not slot+32): blendrec aos[8]/aos[9] live in
                // the upper 32B of the 64B page and overlap PACK2 splat[0..1].
                auto splat = reinterpret_cast<volatile uint32_t*>(pack_l1);
                float mx = bits_to_f(aos[3]);
                float my = bits_to_f(aos[4]);
                mx -= tx_tile;
                my -= ty_tile;
                splat[0] = aos[0];
                splat[1] = aos[1];
                splat[2] = aos[2];
                splat[3] = aos[9];
                splat[4] = f_to_bits(mx);
                splat[5] = f_to_bits(my);
                // iter 132: op/color UNORM16 are packed ONCE per gaussian on the
                // NCRISC side (sort_bin pack_invariants) and published into
                // blendrec[10],[11] (via a full-64B page write-back); this depth-
                // sorted overflow gather only COPIES the two packed words from the
                // 64B blendrec page it already reads — NO re-pack and NO extra read.
                // Bit-identical to iter-131's birth pack (same fp32 inputs, same
                // rounding). iter-131 ablation MEASURED the re-pack at ~5.3 ms/view
                // busiest-core (frame BRISC-FW -7.1) — the dominant overflow-gather
                // cost, now eliminated off the long pole.
                splat[6] = aos[10];
                splat[7] = aos[11];
                const uint32_t out_g = brec_out_g[b];
                const uint32_t out_page = sc_page + (out_g / SLAB_RECS_PER_PAGE);
                const uint32_t out_off = (out_g % SLAB_RECS_PER_PAGE) * L1_SPLAT_BYTES;
                noc_async_write(
                    pack_l1,
                    get_noc_addr(out_page, payload_acc) + out_off,
                    L1_SPLAT_BYTES);
            }
            noc_async_write_barrier();
            nbrec = 0;
        };
        while (processed < L_sub) {
            const uint32_t global_idx = id_start_sc + processed;
            const uint32_t id_page = global_idx >> 4;
            const uint32_t id_ip = global_idx & 0xF;
            if (static_cast<int32_t>(id_page) != sorted_id_page_cached) {
                noc_async_read(get_noc_addr(id_page, sorted_acc), ids_scr, PAGE_BYTES);
                noc_async_read_barrier();
                sorted_id_page_cached = static_cast<int32_t>(id_page);
            }
            uint32_t take = ELEMS_PER_PAGE - id_ip;
            if (take > L_sub - processed) take = L_sub - processed;
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
    }
}
