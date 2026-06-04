// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-local L1 microblock-cull READER (iter 60 / step D).
//
// For each LPT tile (and each post-sort subchunk) this reader:
//   1. bulk-loads the subchunk's PACK2 records into L1 (dense bucket or overflow gather),
//   2. depth-sorts in L1 when serving from the dense bucket (stable LSD radix),
//   3. streams SFPU cull coeff rows {cov, image-space center, opacity, thr} from the
//      L1-resident records — NO global sort_sorted_ids gather for in-budget tiles.
//
// Pairs with microblock_cull_compute + writer_tile_l1_mask; masks land in an
// L1-interleaved buffer the blend reader consumes (no DRAM cull_masks hot path).

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
constexpr uint32_t CB_BSORT      = 9;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t TILE_SIZE = 32;
constexpr uint32_t CHUNK_MAX = 16;

constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
constexpr uint32_t GATHER_SLOT_BYTES = 64u;

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

template <typename IDS>
inline uint32_t load_ids_chunk(
    const IDS& ids_acc, uint32_t id_start, uint32_t processed, uint32_t L,
    uint32_t scratch_addr, uint32_t* out) {
    const uint32_t global_idx = id_start + processed;
    const uint32_t page_idx = global_idx / CHUNK_MAX;
    const uint32_t in_page  = global_idx % CHUNK_MAX;
    auto ids_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    noc_async_read_tile(page_idx, ids_acc, scratch_addr);
    noc_async_read_barrier();
    uint32_t take = CHUNK_MAX - in_page;
    if (take > L - processed) take = L - processed;
    for (uint32_t i = 0; i < take; ++i) out[i] = ids_ptr[in_page + i];
    return take;
}

template <typename REC>
inline void issue_chunk_reads_aos(
    const uint32_t* gids, uint32_t take, uint32_t buf_addr, const REC& rec_acc) {
    for (uint32_t j = 0; j < take; ++j) {
        noc_async_read_tile(gids[j], rec_acc, buf_addr + j * GATHER_SLOT_BYTES);
    }
}

inline void pack_blendrec_to_l1(
    volatile uint32_t* aos, volatile uint32_t* splat,
    float tx_tile, float ty_tile) {
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
    auto pack_u16 = [](float v) -> uint32_t {
        if (v <= 0.0f) return 0u;
        if (v >= 1.0f) return 65535u;
        return static_cast<uint32_t>(v * 65535.0f + 0.5f);
    };
    splat[6] = pack_u16(op) | (pack_u16(cr) << 16);
    splat[7] = pack_u16(cg) | (pack_u16(cb) << 16);
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

        uint32_t Lb = 0;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            noc_async_read_tile(pg, bucket_meta_acc, scr);
            noc_async_read_barrier();
            auto bmp = reinterpret_cast<volatile uint32_t*>(scr);
            Lb = bmp[off + 1u];
        }

        uint32_t num_subchunks = 1;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            noc_async_read_tile(pg, subchunk_meta_acc, scr);
            noc_async_read_barrier();
            num_subchunks = reinterpret_cast<volatile uint32_t*>(scr)[off + 1u];
            if (num_subchunks == 0u) num_subchunks = 1u;
        }

        for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
            const uint32_t sc_off = sc * MB_BUCKET_FIT;
            const uint32_t L_sub = (sc_off >= L) ? 0u
                : ((L - sc_off > MB_BUCKET_FIT) ? MB_BUCKET_FIT : (L - sc_off));
            const uint32_t id_start_sc = id_start + sc_off;
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

            if (num_subchunks == 1u && Lb > 0 && Lb <= MB_BUCKET_FIT) {
                const uint32_t Lb_local = Lb;
                const uint32_t npages = (Lb_local + 1u) >> 1;
                cb_reserve_back(CB_BUCKET, npages);
                const uint32_t buck = get_write_ptr(CB_BUCKET);
                {
                    const uint32_t page0 = tile_id * (MB_BUCKET_FIT >> 1);
                    uint32_t pp = 0;
                    while (pp < npages) {
                        const uint32_t end = (pp + 64u < npages) ? pp + 64u : npages;
                        for (uint32_t q = pp; q < end; ++q) {
                            noc_async_read_tile(page0 + q, l1_recs_acc,
                                                buck + q * L1_PACK_PAGE_BYTES);
                        }
                        noc_async_read_barrier();
                        pp = end;
                    }
                }
                const uint32_t bs = get_write_ptr(CB_BSORT);
                uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
                uint32_t* idxB = idxA + MB_BUCKET_FIT;
                uint32_t* cnt  = idxB + MB_BUCKET_FIT;
                auto key_of = [&](uint32_t idx) -> uint32_t {
                    return l1_splat_words(buck, idx)[3];
                };
                uint32_t* sorted;
                if (Lb_local <= 16u) {
                    for (uint32_t i = 0; i < Lb_local; ++i) idxA[i] = i;
                    for (uint32_t i = 1; i < Lb_local; ++i) {
                        const uint32_t tmp = idxA[i];
                        const uint32_t ki = key_of(tmp);
                        uint32_t j = i;
                        while (j > 0 && key_of(idxA[j - 1]) > ki) { idxA[j] = idxA[j - 1]; --j; }
                        idxA[j] = tmp;
                    }
                    sorted = idxA;
                } else {
                    for (uint32_t i = 0; i < Lb_local; ++i) idxA[i] = i;
                    uint32_t* cur = idxA;
                    uint32_t* nxt = idxB;
                    for (uint32_t byte = 0; byte < 4u; ++byte) {
                        const uint32_t shift = byte * 8u;
                        for (uint32_t c = 0; c < 256u; ++c) cnt[c] = 0;
                        for (uint32_t i = 0; i < Lb_local; ++i) cnt[(key_of(cur[i]) >> shift) & 0xFFu]++;
                        uint32_t sum = 0;
                        for (uint32_t c = 0; c < 256u; ++c) { const uint32_t t = cnt[c]; cnt[c] = sum; sum += t; }
                        for (uint32_t i = 0; i < Lb_local; ++i) {
                            const uint32_t b = (key_of(cur[i]) >> shift) & 0xFFu;
                            nxt[cnt[b]++] = cur[i];
                        }
                        uint32_t* t = cur; cur = nxt; nxt = t;
                    }
                    sorted = cur;
                }
                for (uint32_t k = 0; k < Lb_local; ++k) {
                    emit_cull_row_from_l1_splat(
                        l1_splat_words(buck, sorted[k]), tx_tile, ty_tile, contrib_floor);
                }
                continue;
            }

            const uint32_t rec_pages = (L_sub + 1u) >> 1;
            cb_reserve_back(CB_BUCKET, rec_pages);
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                const uint32_t ids_scr = get_write_ptr(CB_SCR_IDS);
                const uint32_t aos_base = get_write_ptr(CB_SCR_ATTR);
                uint32_t gids[16];
                uint32_t processed = 0;
                while (processed < L_sub) {
                    const uint32_t take = load_ids_chunk(
                        ids_acc, id_start_sc, processed, L_sub, ids_scr, gids);
                    issue_chunk_reads_aos(gids, take, aos_base, blendrec_acc);
                    noc_async_read_barrier();
                    for (uint32_t j = 0; j < take; ++j) {
                        pack_blendrec_to_l1(
                            reinterpret_cast<volatile uint32_t*>(
                                aos_base + j * GATHER_SLOT_BYTES),
                            l1_splat_words(buck, processed + j),
                            tx_tile, ty_tile);
                    }
                    processed += take;
                }
            }
            for (uint32_t g = 0; g < L_sub; ++g) {
                emit_cull_row_from_l1_splat(
                    l1_splat_words(buck, g), tx_tile, ty_tile, contrib_floor);
            }
        }
    }
}
