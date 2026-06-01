// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage C2 SEQUENTIAL blend reader (GSPLAT_TT_BLEND_PAYLOAD).
//
// The blend attributes + 32-bit microblock mask of every depth-sorted candidate
// have already been packed (by payload_pack.cpp) into a CONTIGUOUS per-candidate
// 64B row in `blend_payload`, indexed by GLOBAL candidate index (== id_start[t]
// + p), byte-identical to the CB_MB_COEFF row the compute kernel consumes:
//   [cov_a, cov_b, cov_c, mx_local, my_local, 0, opacity, cr, cg, cb, mask, 0..]
//
// So this reader does ZERO random SoA gather, ZERO on-core cull, and ZERO mask
// fetch (the mask rides the row). For each tile it just STREAMS the contiguous
// payload pages [id_start, id_end) -- a sequential, prefetchable, multi-page,
// amortized-barrier read -- and pushes each row verbatim into CB_MB_COEFF. This
// kills the ~1.9 GB random attr gather that dominated the devcull blend.
//
// Because the reads are sequential (one barrier per ~CHUNK candidates, the whole
// chunk landed before any row is consumed) and the producer (pack) ran in a
// prior program with a Finish between, NO per-candidate read-completion spin is
// needed -- unlike the random single-page cull_masks read (iter 15). An optional
// per-chunk settle spin (GSPLAT_TT_CULL_SPIN -> MB_PAY_SPIN) is wired for the
// spin/PSNR sweep; default OFF.
//
// RUNTIME ARGS
//   0: ranges_addr     sort_tile_ranges (uint32 [start,end] pair per tile)
//   1: xramp_addr      shared permuted x ramp tile
//   2: yramp_addr      shared permuted y ramp tile
//   3: tile_ids_addr   this frame's LPT tile-id list (uint32)
//   4: lpt_meta_addr   sort_lpt_meta (per-core [start,count])
//   5: core_index      this core's slot in lpt_meta
//   6: payload_addr    blend_payload DRAM base (64B rows, global-index)
//
// COMPILE-TIME ARGS: 6 DRAM-interleaved TensorAccessorArgs: ranges, xramp,
// yramp, tile_ids, lpt_meta, payload.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#if defined(MB_PAYLOAD_DEBUG)
#include "api/debug/dprint.h"
#endif

namespace {

constexpr uint32_t ROW_BYTES = 64;          // 16 fp32, full DRAM page
constexpr uint32_t ROW_WORDS = ROW_BYTES / 4;
constexpr uint32_t IDS_PAGE_BYTES = 64;     // 16 uint32 per page
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;

constexpr uint32_t CB_XRAMP      = 0;
constexpr uint32_t CB_YRAMP      = 1;
constexpr uint32_t CB_MB_COEFF   = 2;
constexpr uint32_t CB_MB_COUNTS  = 3;
constexpr uint32_t CB_SCR_IDS    = 4;   // reader-private: meta/ranges/tile-id scratch
constexpr uint32_t CB_SCR_PAY    = 5;   // reader-private: payload chunk double-buffer
constexpr uint32_t CB_CORE_TILES = 7;   // hand tile count to compute (no host arg)

constexpr uint32_t CHUNK_MAX = 16;      // payload rows prefetched per barrier
constexpr uint32_t CHUNK_BYTES = CHUNK_MAX * ROW_BYTES;  // 1024B per buffer

}  // namespace

void kernel_main() {
    const uint32_t ranges_addr   = get_arg_val<uint32_t>(0);
    const uint32_t xramp_addr    = get_arg_val<uint32_t>(1);
    const uint32_t yramp_addr    = get_arg_val<uint32_t>(2);
    const uint32_t tile_ids_addr = get_arg_val<uint32_t>(3);
    const uint32_t lpt_meta_addr = get_arg_val<uint32_t>(4);
    const uint32_t core_index    = get_arg_val<uint32_t>(5);
    const uint32_t payload_addr  = get_arg_val<uint32_t>(6);

    constexpr auto ranges_args   = TensorAccessorArgs<0>();
    constexpr auto xramp_args     = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto yramp_args     = TensorAccessorArgs<xramp_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args  = TensorAccessorArgs<yramp_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args  = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();
    constexpr auto payload_args   = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();

    constexpr uint32_t SOA_PAGE_BYTES = 64;
    const auto ranges_acc   = TensorAccessor(ranges_args,   ranges_addr,   SOA_PAGE_BYTES);
    const auto xramp_acc    = TensorAccessor(xramp_args,    xramp_addr,    RAMP_TILE_BYTES);
    const auto yramp_acc    = TensorAccessor(yramp_args,    yramp_addr,    RAMP_TILE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, 64);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, 64);
    const auto payload_acc  = TensorAccessor(payload_args,  payload_addr,  ROW_BYTES);

    // Host-free LPT: this core's (start,count) from resident sort_lpt_meta.
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

    // Cache this core's tile-ID slice in L1 (private ids scratch CB).
    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t scratch_addr = get_write_ptr(CB_SCR_IDS);
        auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
        const uint32_t ids_per_page = 64 / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            noc_async_read(get_noc_addr(page_idx, tile_ids_acc), scratch_addr, 64);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            }
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    // Constant permuted coordinate ramps: stream once per core for compute.
    cb_reserve_back(CB_XRAMP, 1);
    noc_async_read_tile(0, xramp_acc, get_write_ptr(CB_XRAMP));
    cb_reserve_back(CB_YRAMP, 1);
    noc_async_read_tile(0, yramp_acc, get_write_ptr(CB_YRAMP));
    noc_async_read_barrier();
    cb_push_back(CB_XRAMP, 1);
    cb_push_back(CB_YRAMP, 1);

    const uint32_t pay_base = get_write_ptr(CB_SCR_PAY);

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];

        // Per-tile candidate id range [id_start, id_end) from sort_tile_ranges.
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

        // Per-tile candidate count (compute reads slot 0). One row per candidate
        // (mask==0 candidates dispatch nothing on the compute side).
        cb_reserve_back(CB_MB_COUNTS, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS))[0] = L;
        cb_push_back(CB_MB_COUNTS, 1);

        if (L == 0) {
            continue;
        }

        // DIAG(iter19): dead-simple SERIAL stream (one row per barrier) to test
        // whether the double-buffered chunk pipeline corrupts rows past chunk 0.
        for (uint32_t p = 0; p < L; ++p) {
            noc_async_read_tile(id_start + p, payload_acc, pay_base);
            noc_async_read_barrier();
            for (volatile int _s = 0; _s < 1024; ++_s) { }
            auto src = reinterpret_cast<volatile uint32_t*>(pay_base);
            cb_reserve_back(CB_MB_COEFF, 1);
            auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COEFF));
            for (uint32_t w = 0; w < ROW_WORDS; ++w) {
                row[w] = src[w];
            }
            cb_push_back(CB_MB_COEFF, 1);
        }
    }
}
