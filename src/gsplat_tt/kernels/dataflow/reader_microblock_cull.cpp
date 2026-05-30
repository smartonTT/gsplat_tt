// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU microblock-cull READER (GSPLAT_TT_SFPU_CULL).
//
// Feeds microblock_cull_compute.cpp. For each screen tile this core owns it:
//   1. (once, at startup) streams the two constant box-origin ramps the compute
//      kernel needs, into CB_BOX_OX / CB_BOX_OY (kept resident there).
//   2. reads the tile's candidate range [id_start,id_end) from the resident
//      sort_tile_ranges, pushes [L, tx_pix, ty_pix] into CB_CULL_COUNTS,
//   3. gathers each candidate's 6 cull attributes (raw cov a/b/c + image-space
//      center + opacity) straight from the resident per-component SoA proj_m_*
//      buffers by id and emits a 6-word coeff row into CB_CULL_COEFF.
//
// PURE INTEGER / DATA-MOVEMENT ONLY: no float ops on this RISC. The cull math
// (conic, constrained-min Mahalanobis, exp threshold) all runs on the SFPU.
//
// RUNTIME ARGS
//   0..5: proj_m_a/b/c/px/py/opacity DRAM bases (64B SoA pages)
//   6:    sort_sorted_ids base
//   7:    sort_tile_ranges base
//   8:    box_ox ramp base   9: box_oy ramp base
//   10:   tile_ids base   11: tile_ids_start   12: tile_ids_count   13: tiles_x
// COMPILE-TIME: 11 DRAM-interleaved TensorAccessorArgs in the above buffer order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(CULL_READER_DEBUG) || defined(CULL_DEBUG_REF)
#include "api/debug/dprint.h"
#endif

namespace {

#if defined(CULL_READER_DEBUG) || defined(CULL_DEBUG_REF)
inline float dbg_bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}
#endif


constexpr uint32_t CB_BOX_OX     = 0;
constexpr uint32_t CB_BOX_OY     = 1;
constexpr uint32_t CB_CULL_COEFF = 2;
constexpr uint32_t CB_CULL_COUNTS= 3;
constexpr uint32_t CB_SCR_IDS    = 4;
constexpr uint32_t CB_SCR_ATTR   = 5;

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t COEFF_ROW_BYTES = 64;  // 16 words, 6 used
constexpr uint32_t TILE_SIZE = 32;

constexpr uint32_t GATHER_FIELDS = 6;
constexpr uint32_t GATHER_SLOT_BYTES = GATHER_FIELDS * SOA_PAGE_BYTES;  // 384B/gaussian
constexpr uint32_t CHUNK_MAX = IDS_PAGE_BYTES / 4;  // 16

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

}  // namespace

void kernel_main() {
    const uint32_t a_addr        = get_arg_val<uint32_t>(0);
    const uint32_t b_addr        = get_arg_val<uint32_t>(1);
    const uint32_t c_addr        = get_arg_val<uint32_t>(2);
    const uint32_t px_addr       = get_arg_val<uint32_t>(3);
    const uint32_t py_addr       = get_arg_val<uint32_t>(4);
    const uint32_t op_addr       = get_arg_val<uint32_t>(5);
    const uint32_t ids_addr      = get_arg_val<uint32_t>(6);
    const uint32_t ranges_addr   = get_arg_val<uint32_t>(7);
    const uint32_t box_ox_addr   = get_arg_val<uint32_t>(8);
    const uint32_t box_oy_addr   = get_arg_val<uint32_t>(9);
    const uint32_t tile_ids_addr = get_arg_val<uint32_t>(10);
    const uint32_t tile_ids_start= get_arg_val<uint32_t>(11);
    const uint32_t tile_ids_count= get_arg_val<uint32_t>(12);
    const uint32_t tiles_x       = get_arg_val<uint32_t>(13);

    constexpr auto a_args      = TensorAccessorArgs<0>();
    constexpr auto b_args      = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args      = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto px_args     = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto py_args     = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto op_args     = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto ids_args    = TensorAccessorArgs<op_args.next_compile_time_args_offset()>();
    constexpr auto ranges_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto bx_args     = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto by_args     = TensorAccessorArgs<bx_args.next_compile_time_args_offset()>();
    constexpr auto tids_args   = TensorAccessorArgs<by_args.next_compile_time_args_offset()>();

    const auto a_acc      = TensorAccessor(a_args,      a_addr,      SOA_PAGE_BYTES);
    const auto b_acc      = TensorAccessor(b_args,      b_addr,      SOA_PAGE_BYTES);
    const auto c_acc      = TensorAccessor(c_args,      c_addr,      SOA_PAGE_BYTES);
    const auto px_acc     = TensorAccessor(px_args,     px_addr,     SOA_PAGE_BYTES);
    const auto py_acc     = TensorAccessor(py_args,     py_addr,     SOA_PAGE_BYTES);
    const auto op_acc     = TensorAccessor(op_args,     op_addr,     SOA_PAGE_BYTES);
    const auto ids_acc    = TensorAccessor(ids_args,    ids_addr,    IDS_PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, ranges_addr, SOA_PAGE_BYTES);
    const auto bx_acc     = TensorAccessor(bx_args,     box_ox_addr, RAMP_TILE_BYTES);
    const auto by_acc     = TensorAccessor(by_args,     box_oy_addr, RAMP_TILE_BYTES);
    const auto tids_acc   = TensorAccessor(tids_args,   tile_ids_addr, IDS_PAGE_BYTES);

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

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;

        // Candidate range from resident sort_tile_ranges (start,end u32 pair).
        uint32_t id_start, id_end;
        {
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            id_start = read_soa_u32(ranges_acc, tile_id * 2u + 0u, scr);
            id_end   = read_soa_u32(ranges_acc, tile_id * 2u + 1u, scr);
        }
        const uint32_t L = id_end - id_start;

        cb_reserve_back(CB_CULL_COUNTS, 1);
        {
            auto cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COUNTS));
            cnt[0] = L;
            cnt[1] = tx * TILE_SIZE;
            cnt[2] = ty * TILE_SIZE;
        }
        cb_push_back(CB_CULL_COUNTS, 1);

        const uint32_t ids_scr = get_write_ptr(CB_SCR_IDS);
        const uint32_t attr_base = get_write_ptr(CB_SCR_ATTR);
        uint32_t gids[CHUNK_MAX];
        uint32_t processed = 0;
        while (processed < L) {
            const uint32_t take = load_ids_chunk(ids_acc, id_start, processed, L, ids_scr, gids);
            // Issue all 6*take SoA reads ahead of one barrier.
            for (uint32_t j = 0; j < take; ++j) {
                const uint32_t g = gids[j];
                const uint32_t pg = g >> 4;
                const uint32_t s = attr_base + j * GATHER_SLOT_BYTES;
                noc_async_read_tile(pg, a_acc,  s + 0u * SOA_PAGE_BYTES);
                noc_async_read_tile(pg, b_acc,  s + 1u * SOA_PAGE_BYTES);
                noc_async_read_tile(pg, c_acc,  s + 2u * SOA_PAGE_BYTES);
                noc_async_read_tile(pg, px_acc, s + 3u * SOA_PAGE_BYTES);
                noc_async_read_tile(pg, py_acc, s + 4u * SOA_PAGE_BYTES);
                noc_async_read_tile(pg, op_acc, s + 5u * SOA_PAGE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < take; ++j) {
                const uint32_t g = gids[j];
                const uint32_t lane = g & 0xF;
                const uint32_t s = attr_base + j * GATHER_SLOT_BYTES;
                cb_reserve_back(CB_CULL_COEFF, 1);
                auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CULL_COEFF));
                row[0] = reinterpret_cast<volatile uint32_t*>(s + 0u * SOA_PAGE_BYTES)[lane];
                row[1] = reinterpret_cast<volatile uint32_t*>(s + 1u * SOA_PAGE_BYTES)[lane];
                row[2] = reinterpret_cast<volatile uint32_t*>(s + 2u * SOA_PAGE_BYTES)[lane];
                row[3] = reinterpret_cast<volatile uint32_t*>(s + 3u * SOA_PAGE_BYTES)[lane];
                row[4] = reinterpret_cast<volatile uint32_t*>(s + 4u * SOA_PAGE_BYTES)[lane];
                row[5] = reinterpret_cast<volatile uint32_t*>(s + 5u * SOA_PAGE_BYTES)[lane];
#if defined(CULL_DEBUG_REF)
                {
                    const uint32_t local = processed + j;
                    static uint32_t ref_n = 0;
                    if (ref_n < 400u) {
                        ref_n++;
                        DPRINT << "CULLCOEF t=" << tile_id << " local=" << local
                               << " a=" << F32(dbg_bits_to_f(row[0]))
                               << " b=" << F32(dbg_bits_to_f(row[1]))
                               << " c=" << F32(dbg_bits_to_f(row[2]))
                               << " mx=" << F32(dbg_bits_to_f(row[3]))
                               << " my=" << F32(dbg_bits_to_f(row[4]))
                               << " op=" << F32(dbg_bits_to_f(row[5]))
                               << " tx=" << (tx * TILE_SIZE) << " ty=" << (ty * TILE_SIZE) << ENDL();
                    }
                }
#endif
#if defined(CULL_READER_DEBUG)
                {
                    static uint32_t dbg_n = 0;
                    if (dbg_n < 12u) {
                        dbg_n++;
                        DPRINT << "CULLIN t=" << tile_id << " g=" << g
                               << " a=" << F32(dbg_bits_to_f(row[0]))
                               << " c=" << F32(dbg_bits_to_f(row[2]))
                               << " mx=" << F32(dbg_bits_to_f(row[3]))
                               << " my=" << F32(dbg_bits_to_f(row[4]))
                               << " op=" << F32(dbg_bits_to_f(row[5]))
                               << " tx=" << (tx * TILE_SIZE) << " ty=" << (ty * TILE_SIZE) << ENDL();
                    }
                }
#endif
                cb_push_back(CB_CULL_COEFF, 1);
            }
            processed += take;
        }
    }
}
