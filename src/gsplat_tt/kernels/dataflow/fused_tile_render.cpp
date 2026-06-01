// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED_TILE render reader (GSPLAT_TT_FUSED_TILE scaffold v1).
// Phase-1: identical to reader_microblock_cull (feeds microblock_cull_compute).
// Phase-2: extend per-tile loop to emit blend coeff rows from L1 masks without
// a separate blend reader enqueue.
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
//   10:   tile_ids base   11: sort_lpt_meta base   12: core_index   13: tiles_x
//   14:   contrib_floor (f32 bits)
// COMPILE-TIME: 12 DRAM-interleaved TensorAccessorArgs in the above buffer order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(CULL_READER_DEBUG) || defined(CULL_DEBUG_REF) || defined(FUSE_AB_ROW) || defined(FUSE_AB)
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
constexpr uint32_t CB_CORE_TILES = 7;  // tile count for compute (no host LPT D2H)
#ifdef FUSE_BLEND
// §8.4: the SAME reader streams the blend coeff rows after the cull rows,
// per tile, reading the 32-bit microblock masks from the L1 CB_TILE_MASKS
// handoff (writer->reader) instead of the cull_masks DRAM round-trip. No spin.
constexpr uint32_t CB_XRAMP      = 8;   // pixel-center x ramp (streamed once)
constexpr uint32_t CB_YRAMP      = 9;   // pixel-center y ramp (streamed once)
constexpr uint32_t CB_MB_COEFF   = 10;  // blend coeff row per gaussian (reader->compute)
constexpr uint32_t CB_MB_COUNTS  = 11;  // per-tile blend gaussian-row count (== L)
constexpr uint32_t CB_SCR_ATTR_B = 12;  // reader-private blend gather scratch
constexpr uint32_t CB_TILE_MASKS = 18;  // L1 mask handoff (writer->reader), 32 u32/page
constexpr uint32_t BLEND_MASKS_PER_PAGE = 32;
constexpr uint32_t BLEND_FIELDS = 8;    // a,b,c,px,py,op + 2 color pages
constexpr uint32_t BLEND_SLOT_BYTES = BLEND_FIELDS * 64u;  // 512B/gaussian
#endif

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

#ifdef FUSE_BLEND
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
#endif

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
    const uint32_t lpt_meta_addr = get_arg_val<uint32_t>(11);
    const uint32_t core_index    = get_arg_val<uint32_t>(12);
    const uint32_t tiles_x       = get_arg_val<uint32_t>(13);
    const uint32_t floor_bits    = get_arg_val<uint32_t>(14);
    float contrib_floor;
    __builtin_memcpy(&contrib_floor, &floor_bits, 4);
#ifdef FUSE_BLEND
    const uint32_t col_addr      = get_arg_val<uint32_t>(15);  // proj_m_colors (AoS M*3)
    const uint32_t xramp_addr    = get_arg_val<uint32_t>(16);  // pixel-center x ramp
    const uint32_t yramp_addr    = get_arg_val<uint32_t>(17);  // pixel-center y ramp
#endif

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
    constexpr auto lpt_meta_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
#ifdef FUSE_BLEND
    constexpr auto col_args    = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto xramp_args  = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
    constexpr auto yramp_args  = TensorAccessorArgs<xramp_args.next_compile_time_args_offset()>();
#endif

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
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, SOA_PAGE_BYTES);
#ifdef FUSE_BLEND
    const auto col_acc    = TensorAccessor(col_args,    col_addr,    SOA_PAGE_BYTES);
    const auto xramp_acc  = TensorAccessor(xramp_args,  xramp_addr,  RAMP_TILE_BYTES);
    const auto yramp_acc  = TensorAccessor(yramp_args,  yramp_addr,  RAMP_TILE_BYTES);
#endif

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
    cb_reserve_back(CB_CORE_TILES, 1);
    reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CORE_TILES))[0] = tile_ids_count;
    cb_push_back(CB_CORE_TILES, 1);

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

#ifdef FUSE_BLEND
    // Stream the two constant pixel-center ramps ONCE for the blend compute
    // phase; compute keeps them resident (waits once, never pops).
    cb_reserve_back(CB_XRAMP, 1);
    noc_async_read_tile(0, xramp_acc, get_write_ptr(CB_XRAMP));
    cb_reserve_back(CB_YRAMP, 1);
    noc_async_read_tile(0, yramp_acc, get_write_ptr(CB_YRAMP));
    noc_async_read_barrier();
    cb_push_back(CB_XRAMP, 1);
    cb_push_back(CB_YRAMP, 1);
#endif

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
                // Per-gaussian Mahalanobis threshold computed HERE on the data
                // mover (BRISC) -- the SAME RISC class on which the soft-float
                // reference's __builtin_logf is exact (and where the compute TRISC
                // logf returns garbage). thr = -2*log(floor/op) == the reference's
                // thresh_m2; <0 sentinel for op<=floor reproduces its early-out
                // (mask 0). One logf per candidate, NOT per microblock; the heavy
                // per-microblock metric stays on the SFPU. The compute kernel then
                // does the divide-free keep test qmin <= det*thr.
                {
                    const uint32_t op_bits = row[5];
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

#ifdef FUSE_BLEND
        // ---- BLEND-GATHER + EMIT (tile t) -----------------------------------
        // The cull phase above pushed all L cull rows; by now (concurrently) the
        // compute culled them -> CB_KEEP -> the writer packed the 32-bit masks
        // into the L1 CB_TILE_MASKS handoff. Emit one blend coeff row per
        // candidate, reading its mask from CB_TILE_MASKS (popped every 32) — NO
        // cull_masks DRAM round-trip, NO settle spin.
        //
        // §2.3 invariant: push the per-tile COUNT (== L) BEFORE the first blend
        // coeff row, so the compute learns num_g immediately and drains the
        // depth-bounded CB_MB_COEFF one row at a time (no CB-overflow deadlock).
        cb_reserve_back(CB_MB_COUNTS, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS))[0] = L;
        cb_push_back(CB_MB_COUNTS, 1);

        if (L > 0) {
            const float tx_tile = static_cast<float>(tx * TILE_SIZE);
            const float ty_tile = static_cast<float>(ty * TILE_SIZE);
            const uint32_t b_ids_scr  = get_write_ptr(CB_SCR_IDS);
            const uint32_t b_attr_base = get_write_ptr(CB_SCR_ATTR_B);
            uint32_t b_gids[CHUNK_MAX];
            volatile uint32_t* mask_ptr = nullptr;  // current CB_TILE_MASKS page
            uint32_t b_processed = 0;
            while (b_processed < L) {
                const uint32_t take = load_ids_chunk(ids_acc, id_start, b_processed, L, b_ids_scr, b_gids);
                // Issue this chunk's blend gather (a,b,c,px,py,op + colors).
                for (uint32_t j = 0; j < take; ++j) {
                    const uint32_t g = b_gids[j];
                    const uint32_t pg = g >> 4;
                    const uint32_t s = b_attr_base + j * BLEND_SLOT_BYTES;
                    noc_async_read_tile(pg, a_acc,  s + 0u * SOA_PAGE_BYTES);
                    noc_async_read_tile(pg, b_acc,  s + 1u * SOA_PAGE_BYTES);
                    noc_async_read_tile(pg, c_acc,  s + 2u * SOA_PAGE_BYTES);
                    noc_async_read_tile(pg, px_acc, s + 3u * SOA_PAGE_BYTES);
                    noc_async_read_tile(pg, py_acc, s + 4u * SOA_PAGE_BYTES);
                    noc_async_read_tile(pg, op_acc, s + 5u * SOA_PAGE_BYTES);
                    // AoS M*3 colors: r/g/b are 3 consecutive elems spanning 1-2
                    // 16-elem pages. Read page0 into slot 6, page1 into slot 7
                    // only on a straddle.
                    const uint32_t e0 = g * 3u;
                    const uint32_t cpg0 = e0 >> 4;
                    const uint32_t cpg1 = (e0 + 2u) >> 4;
                    noc_async_read_tile(cpg0, col_acc, s + 6u * SOA_PAGE_BYTES);
                    if (cpg1 != cpg0) {
                        noc_async_read_tile(cpg1, col_acc, s + 7u * SOA_PAGE_BYTES);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < take; ++j) {
                    const uint32_t p = b_processed + j;
                    // Pop a fresh 32-mask page from the L1 handoff at each 32
                    // boundary. cb_wait/cb_pop fencing guarantees the masks are
                    // present + committed by construction (no DRAM settle).
                    if ((p & (BLEND_MASKS_PER_PAGE - 1u)) == 0u) {
                        if (mask_ptr != nullptr) {
                            cb_pop_front(CB_TILE_MASKS, 1);
                        }
                        cb_wait_front(CB_TILE_MASKS, 1);
                        mask_ptr = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_MASKS));
                    }
                    const uint32_t mask = mask_ptr[p & (BLEND_MASKS_PER_PAGE - 1u)];

                    const uint32_t g = b_gids[j];
                    const uint32_t lane = g & 0xF;
                    const uint32_t s = b_attr_base + j * BLEND_SLOT_BYTES;
                    const uint32_t cov_a_bits = reinterpret_cast<volatile uint32_t*>(s + 0u * SOA_PAGE_BYTES)[lane];
                    const uint32_t cov_b_bits = reinterpret_cast<volatile uint32_t*>(s + 1u * SOA_PAGE_BYTES)[lane];
                    const uint32_t cov_c_bits = reinterpret_cast<volatile uint32_t*>(s + 2u * SOA_PAGE_BYTES)[lane];
                    const uint32_t mx_bits    = reinterpret_cast<volatile uint32_t*>(s + 3u * SOA_PAGE_BYTES)[lane];
                    const uint32_t my_bits    = reinterpret_cast<volatile uint32_t*>(s + 4u * SOA_PAGE_BYTES)[lane];
                    const uint32_t op_bits    = reinterpret_cast<volatile uint32_t*>(s + 5u * SOA_PAGE_BYTES)[lane];
                    const uint32_t e0 = g * 3u;
                    const uint32_t cbase = (e0 >> 4) * 16u;
                    auto cwin = reinterpret_cast<volatile uint32_t*>(s + 6u * SOA_PAGE_BYTES);
                    const uint32_t cr = cwin[(e0 + 0u) - cbase];
                    const uint32_t cg = cwin[(e0 + 1u) - cbase];
                    const uint32_t cb = cwin[(e0 + 2u) - cbase];

                    const float mean_x = bits_to_f(mx_bits);
                    const float mean_y = bits_to_f(my_bits);

                    cb_reserve_back(CB_MB_COEFF, 1);
                    auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COEFF));
                    row[0] = cov_a_bits;
                    row[1] = cov_b_bits;
                    row[2] = cov_c_bits;
                    row[3] = f_to_bits(mean_x - tx_tile);   // mx_local
                    row[4] = f_to_bits(mean_y - ty_tile);   // my_local
                    row[5] = 0u;
                    row[6] = op_bits;
                    row[7] = cr;
                    row[8] = cg;
                    row[9] = cb;
                    row[10] = mask;
                    row[11] = 0u;
                    row[12] = 0u;
                    row[13] = 0u;
                    row[14] = 0u;
                    row[15] = 0u;
#if defined(FUSE_AB_ROW) || defined(FUSE_AB)
                    if (p < 6u) {
                        static uint32_t _ab_n = 0;
                        if (_ab_n < 240u) { _ab_n++;
                            DPRINT << "ABROW t=" << tile_id << " p=" << p << " g=" << g
                                   << " a=" << row[0]
                                   << " mxl=" << row[3]
                                   << " op=" << row[6]
                                   << " m=" << row[10] << ENDL();
                        }
                    }
#endif
                    cb_push_back(CB_MB_COEFF, 1);
                }
                b_processed += take;
            }
            if (mask_ptr != nullptr) {
                cb_pop_front(CB_TILE_MASKS, 1);
            }
        }
#endif  // FUSE_BLEND
    }
}
