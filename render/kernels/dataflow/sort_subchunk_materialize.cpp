// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Post-radix subchunk materialize (iter 54 / step A): depth-sorted PACK2 payloads.
// In-budget tiles (count <= bucket_fit): bulk-copy buf_l1_recs + L1 radix
// permute (no per-splat blendrec gather). Overflow tiles: sc==0 uses
// sort_sorted_ids-order blendrec gather (masks follow radix ids, not L1 slots);
// sc>=1 uses the same batched blendrec gather into depth-ordered PACK2 slabs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
constexpr uint32_t TILE_SIZE = 32u;
// iter 76: larger blendrec gather batches (fewer read/write barriers on sc>=1).
constexpr uint32_t REC_BATCH = 32u;
constexpr uint32_t PACK_OFF = 32u;  // pack PACK2 into 2nd half of each 64B staging slot

constexpr uint32_t CB_SCR = 0;
constexpr uint32_t CB_IDS = 1;
constexpr uint32_t CB_REC = 2;
constexpr uint32_t CB_PACK = 3;
constexpr uint32_t CB_BUCKET = 4;
constexpr uint32_t CB_BSORT = 5;

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

inline uint32_t pack_fp32_unorm16(float v) {
    if (v <= 0.0f) return 0u;
    if (v >= 1.0f) return 65535u;
    return static_cast<uint32_t>(v * 65535.0f + 0.5f);
}

inline volatile uint32_t* l1_splat_words(uint32_t buck_base, uint32_t g) {
    return reinterpret_cast<volatile uint32_t*>(
        buck_base + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
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
    const uint32_t tile_ids_addr  = get_arg_val<uint32_t>(7);
    const uint32_t tile_ids_start = get_arg_val<uint32_t>(8);
    const uint32_t tile_ids_count = get_arg_val<uint32_t>(9);
    const uint32_t tiles_x        = get_arg_val<uint32_t>(10);
    const uint32_t bucket_fit     = get_arg_val<uint32_t>(11);

    constexpr auto sorted_args = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<sorted_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto payload_args = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    constexpr auto blend_meta_args = TensorAccessorArgs<payload_args.next_compile_time_args_offset()>();
    constexpr auto dir_args = TensorAccessorArgs<blend_meta_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args = TensorAccessorArgs<dir_args.next_compile_time_args_offset()>();

    const auto sorted_acc   = TensorAccessor(sorted_args,   sorted_addr,   PAGE_BYTES);
    const auto ranges_acc   = TensorAccessor(ranges_args,   ranges_addr,   PAGE_BYTES);
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    const auto l1_recs_acc  = TensorAccessor(l1_recs_args,  l1_recs_addr,  L1_PACK_PAGE_BYTES);
    const auto payload_acc  = TensorAccessor(payload_args,  payload_addr,  L1_PACK_PAGE_BYTES);
    const auto blend_meta_acc = TensorAccessor(blend_meta_args, blend_meta_addr, PAGE_BYTES);
    const auto dir_acc      = TensorAccessor(dir_args,      dir_addr,      PAGE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, PAGE_BYTES);

    if (tile_ids_count == 0) {
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    auto scrp = reinterpret_cast<volatile uint32_t*>(scr);
    const uint32_t ids_scr = get_write_ptr(CB_IDS);
    auto idsp = reinterpret_cast<volatile uint32_t*>(ids_scr);
    const uint32_t rec_l1 = get_write_ptr(CB_REC);
    const uint32_t pack_l1 = get_write_ptr(CB_PACK);

    constexpr uint32_t MAX_TILE_IDS = 1024;
    uint32_t tile_ids[MAX_TILE_IDS];
    {
        const uint32_t ids_per_page = ELEMS_PER_PAGE;
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            noc_async_read(get_noc_addr(page_idx, tile_ids_acc), ids_scr, PAGE_BYTES);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = idsp[in_page + i];
            }
            out_idx += take;
            remaining -= take;
            page_idx += 1;
            in_page = 0;
        }
    }

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
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
        const uint32_t num_sc =
            (count + bucket_fit - 1u) / bucket_fit;

        uint32_t dir_base = 0;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            noc_async_read(get_noc_addr(pg, blend_meta_acc), scr, PAGE_BYTES);
            noc_async_read_barrier();
            dir_base = scrp[off];
        }

        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t L_sub = (sc_off >= count) ? 0u
                : ((count - sc_off > bucket_fit) ? bucket_fit : (count - sc_off));

            if (L_sub == 0u) {
                continue;
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
                const uint32_t npages = (L + 1u) >> 1;
                const uint32_t buck = get_write_ptr(CB_BUCKET);
                {
                    const uint32_t page0 = tile_id * (bucket_fit >> 1);
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
                const uint32_t bs = get_write_ptr(CB_BSORT);
                uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
                uint32_t* idxB = idxA + bucket_fit;
                uint32_t* cnt  = idxB + bucket_fit;
                uint32_t* sorted;
                if (L <= 16u) {
                    for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
                    for (uint32_t i = 1; i < L; ++i) {
                        const uint32_t tmp = idxA[i];
                        const uint32_t ki = l1_splat_words(buck, tmp)[3];
                        uint32_t j = i;
                        while (j > 0 && l1_splat_words(buck, idxA[j - 1])[3] > ki) {
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
                            cnt[(l1_splat_words(buck, cur[i])[3] >> shift) & 0xFFu]++;
                        }
                        uint32_t sum = 0;
                        for (uint32_t c = 0; c < 256u; ++c) {
                            const uint32_t t = cnt[c];
                            cnt[c] = sum;
                            sum += t;
                        }
                        for (uint32_t i = 0; i < L; ++i) {
                            const uint32_t b =
                                (l1_splat_words(buck, cur[i])[3] >> shift) & 0xFFu;
                            nxt[cnt[b]++] = cur[i];
                        }
                        uint32_t* t = cur;
                        cur = nxt;
                        nxt = t;
                    }
                    sorted = cur;
                }
                for (uint32_t k = 0; k < L; ++k) {
                    const uint32_t idx = sorted[k];
                    const uint32_t out_page = sc_page + (k >> 1);
                    const uint32_t half_off = (k & 1u) * L1_SPLAT_BYTES;
                    const uint32_t src_page = (idx >> 1);
                    const uint32_t src_half = (idx & 1u) * L1_SPLAT_BYTES;
                    noc_async_write(
                        buck + src_page * L1_PACK_PAGE_BYTES + src_half,
                        get_noc_addr(out_page, payload_acc) + half_off,
                        L1_SPLAT_BYTES);
                }
                noc_async_write_barrier();
                continue;
            }

            // sc>=1: batched blendrec gather (iter 76: REC_BATCH=32, per-slot PACK2,
            // one write barrier per batch; reuse sorted-id page across splats).
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
                    auto splat = reinterpret_cast<volatile uint32_t*>(slot + PACK_OFF);
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
                    const float op = bits_to_f(aos[5]);
                    const float cr = bits_to_f(aos[6]);
                    const float cg = bits_to_f(aos[7]);
                    const float cb = bits_to_f(aos[8]);
                    splat[6] = pack_fp32_unorm16(op) | (pack_fp32_unorm16(cr) << 16);
                    splat[7] = pack_fp32_unorm16(cg) | (pack_fp32_unorm16(cb) << 16);
                    const uint32_t out_g = brec_out_g[b];
                    const uint32_t out_page = sc_page + (out_g >> 1);
                    const uint32_t half_off = (out_g & 1u) * L1_SPLAT_BYTES;
                    noc_async_write(
                        slot + PACK_OFF,
                        get_noc_addr(out_page, payload_acc) + half_off,
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
}
