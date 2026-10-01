// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN kernel — R4/R5 resident-pairs handoff (GSPLAT_TT_RESIDENT_PAIRS).
//
// Replaces the host binning (Pass1 counts + Pass2 stable scatter) of
// gsplat_cpu::sort.cpp with a multi-core device pass that reads the RESIDENT
// tile_assign outputs (full-P gaussian-major (gid,tid) pairs + keep mask) and
// the resident proj_m_depth, and writes the PAGE-ALIGNED per-tile (key,id)
// layout the device radix kernel (sort_radix_tile.cpp) consumes — entirely on
// device. No D2H of the pairs, no H2D of keys/ids.
//
// The full-P pairs are split across cores in contiguous 16-pair page ranges
// (identical split for count + scatter). Two modes (host launches both):
//
//   mode 0 (count): each core builds a per-tile histogram of its KEPT pairs in
//     L1 and writes that row into bin2d[core_id*stride .. +num_tiles).
//
//   mode 1 (scatter): the host has, from the per-core histograms, computed the
//     page-aligned per-tile starts and each core's per-tile global base offset
//     (an exclusive prefix over cores within each tile), written back into
//     bin2d. The core does a LOCAL counting-sort of its kept pairs into L1
//     (grouped by tile, gaussian-major within each tile), then writes each
//     tile's contiguous block to DRAM at base_row[t] using batched, page-
//     congruent sub-page NoC writes (the gather_visible_scatter pattern). The
//     within-tile order across cores is gaussian-major (core c's block before
//     core c+1's) — exactly the CPU's stable pre-sort order.
//
// RUNTIME ARGS (all uint32):
//   0: gids_addr   1: tids_addr   2: keep_addr   3: depth_addr  (resident in)
//   4: bin2d_addr  (per-core 2D: hist out / base in, stride-padded rows)
//   5: keys_out_addr  6: ids_out_addr  (page-aligned layout, scatter out)
//   7: page_start  8: page_count  9: P  10: num_tiles  11: stride
//   12: core_id    13: mode (0=count, 1=scatter)
//   15..23: tile_bucket / l1_record args (see below)
//   24: mover (0 = BRISC, 1 = NCRISC)   25: dual (1 = scatter split over both)
//   26: h0_rows_addr   27: pg_mid (count pass split point)
//   28: own fill-done semaphore id   29: peer fill-done semaphore id
//
// DUAL DATA MOVER (T-C). The same kernel runs on BRISC (mover 0) and NCRISC
// (mover 1) of every core. With dual=1 the scatter splits the core's page range
// [lo, hi) at mid: mover 0 takes [lo, mid), mover 1 takes [mid, hi). Both fill
// the core's ONE shared counting-sort region (CB_KS/CB_IS) and write the same
// per-(core, tile) record / overflow slots as the single-mover pass: a tile's
// k-th kept pair in page order lands at offset k, and mover 1's cursors start
// at h0[t] (= mover 0's kept count for tile t, snapshotted by the count pass at
// mid). Pairs are gaussian-major, so [lo, mid) precedes [mid, hi) in the
// single-mover order and every DRAM output is byte-identical to it. After both
// fills (a fill-done semaphore each), mover 0 writes out tiles [0, t_split) and
// mover 1 [t_split, num_tiles). All other staging CBs are private per mover:
// mover 0 uses CB id + MOVER0_CB_OFFSET. The count pass stays on NCRISC.
//
// COMPILE-TIME ARGS: 7 TensorAccessorArgs
//   gids, tids, keep, depth, bin2d, keys_out, ids_out.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
// == render_config::kRecPageBytes / kRecsPerPage (buf_l1_recs + overflow region).
constexpr uint32_t REC_PAGE_BYTES = 2048;
constexpr uint32_t REC_PAGE_RECS = REC_PAGE_BYTES / 32u;
// Mover 0 (BRISC) uses its own copy of every staging CB at this id offset
// (allocated in sort_device.cpp build_program_bin); mover 1 uses the base ids.
constexpr uint32_t MOVER0_CB_OFFSET = 16;

// Task #98: profiling-only ablations of the emit (GSPLAT_TT_EMIT_ABLATE=<mask>).
// Each set bit removes one part so an untraced run bounds what it costs; the
// output is WRONG with any bit set. 1: 32 B record writes, 2: 16 B packoc
// writes, 4: pack_invariants + pack_rec, 8: blendrec prefetch reads,
// 16: depth page reads, 32: the whole per-pair loop (pair pages are still
// read). Unset (default) builds the unchanged kernel.
#ifndef EMIT_ABLATE
#define EMIT_ABLATE 0u
#endif
// Task #100 emit knobs (host env GSPLAT_TT_EMIT_PB / _RING / _PUBOC; see the
// scatter sub-pass 2 below). EMIT_RING_TILES: tile capacity of the ring CB.
#ifndef EMIT_PB
#define EMIT_PB 1u
#endif
#ifndef EMIT_RING
#define EMIT_RING 0u
#endif
#ifndef EMIT_RING_TILES
#define EMIT_RING_TILES 1024u
#endif
#ifndef EMIT_PUBOC
#define EMIT_PUBOC 0u
#endif

// Bit-exact IEEE 754 fp32→fp16 (round-to-nearest-even, no flush-to-zero).
// Used for packing the 32B L1 record (M0, GSPLAT_TT_L1_RECORD).
inline uint16_t fp32_to_fp16(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    const uint32_t sign  = (u >> 31) & 1u;
    const uint32_t exp32 = (u >> 23) & 0xffu;
    const uint32_t mant  = u & 0x7fffffu;
    // NaN / Inf
    if (exp32 == 0xff) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }
    int32_t e = static_cast<int32_t>(exp32) - 127 + 15;
    if (e >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);  // overflow → inf
    }
    if (e <= 0) {
        // Denormal or underflow: encode as denormal fp16 (or 0).
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        const uint32_t m = (mant | 0x800000u) >> (1 - e + 13);
        const uint32_t round = (mant | 0x800000u) >> (-e + 12) & 1u;
        return static_cast<uint16_t>((sign << 15) | m + round);
    }
    // Normal.
    const uint32_t m16 = mant >> 13;
    const uint32_t round = (mant >> 12) & 1u;
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(e) << 10) | m16 + round);
}

}  // namespace

void kernel_main() {
    const uint32_t gids_addr   = get_arg_val<uint32_t>(0);
    const uint32_t tids_addr   = get_arg_val<uint32_t>(1);
    const uint32_t keep_addr   = get_arg_val<uint32_t>(2);
    const uint32_t depth_addr  = get_arg_val<uint32_t>(3);
    const uint32_t bin2d_addr  = get_arg_val<uint32_t>(4);
    const uint32_t keys_addr   = get_arg_val<uint32_t>(5);
    const uint32_t ids_addr    = get_arg_val<uint32_t>(6);
    const uint32_t page_start  = get_arg_val<uint32_t>(7);
    const uint32_t page_count  = get_arg_val<uint32_t>(8);
    const uint32_t P           = get_arg_val<uint32_t>(9);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(10);
    const uint32_t stride      = get_arg_val<uint32_t>(11);
    const uint32_t core_id     = get_arg_val<uint32_t>(12);
    const uint32_t mode        = get_arg_val<uint32_t>(13);
    // The Tracy device zone is opened inside each pass below (histogram vs emit)
    // with a COMPILE-TIME-LITERAL name. DeviceZoneScopedN hashes its argument via
    // Hash16_CT(const char (&)[N]); a runtime ternary decays to const char* and
    // fails that template's N deduction (kernel_profiler.hpp:110) under
    // TT_METAL_DEVICE_PROFILER=1. Two static-named zones keep the per-pass labels.
    // TILE_BUCKET: scatter the FULL projected record into a per-tile contiguous
    // bucket in arbitrary (bin/gaussian) order, so the depth sort can move into
    // the per-tile L1 pass and NO random DRAM read is needed to build the bucket.
    // proj_m_blendrec is 1 record (64B) per page (page==g); tile_recs mirrors the
    // keys/ids layout (record page e ↔ keys/ids element e).
    const uint32_t blendrec_addr = get_arg_val<uint32_t>(15);
    const uint32_t tile_recs_addr= get_arg_val<uint32_t>(16);
    // Arg 17 (formerly the retired dense tile_recs base): the count pass's
    // histogram rows. In scatter mode this core's row is exactly the per-tile
    // kept counts sub-pass 1 would recompute (same page range, same keep test),
    // so it is read instead of re-scanning every tid/keep page. 0 => recount.
    const uint32_t hist_rows_addr = get_arg_val<uint32_t>(17);
    // L1_RECORD (PACK2): scatter 32B records into pre-sized per-tile buckets.
    // buf_l1_recs is BUCKET_FIT*num_tiles logical slots, two per 64B page;
    // buf_l1_rec_base
    // provides per-(core,tile) start slot = t*BUCKET_FIT + prefix.
    const uint32_t l1_recs_addr  = get_arg_val<uint32_t>(18);
    const uint32_t l1_base_addr  = get_arg_val<uint32_t>(19);
    const uint32_t l1_bucket_fit = get_arg_val<uint32_t>(20);  // per-tile bucket slot count
    const uint32_t l1_tiles_x    = get_arg_val<uint32_t>(21);  // tiles per row (tile-local mean)
    // iter-138 (Stage-2b overflow pre-pack): for overflow tiles whose count fits
    // the materialize L1 cap, emit pre-packs the FULL tile (all records, not just
    // the first BUCKET_FIT) into a SEPARATE compact overflow region so the
    // materialize overflow path reads them COALESCED + L1-radix-permutes (like the
    // in-budget path) instead of random-gathering blendrec[gid]. The per-(core,tile)
    // overflow slot base row is l1_ov_base (sentinel 0xFFFFFFFF for non-overflow
    // tiles ⇒ those keep the buf_l1_recs bucket path). Both 0 ⇒ feature disabled
    // (e.g. device-layout path) ⇒ legacy behavior (overflow records dropped).
    const uint32_t l1_ov_addr      = get_arg_val<uint32_t>(22);  // overflow region base (0=off)
    const uint32_t l1_ov_base_addr = get_arg_val<uint32_t>(23);  // per-(core,tile) overflow slot base
    // T-C dual data mover (see header).
    const uint32_t mover        = get_arg_val<uint32_t>(24);
    const bool dual             = get_arg_val<uint32_t>(25) != 0u;
    const uint32_t h0_rows_addr = get_arg_val<uint32_t>(26);
    const uint32_t pg_mid       = get_arg_val<uint32_t>(27);
    const uint32_t sem_own_id   = get_arg_val<uint32_t>(28);
    const uint32_t sem_peer_id  = get_arg_val<uint32_t>(29);
    // BRISC has no count work, and no scatter work unless the split is on.
    if (mover == 0 && (mode == 0 || !dual)) return;
    constexpr uint32_t L1_TILE_SIZE = 32u;  // microblock tile = 32x32 px
    // iter 135 (bit-identical strength reduction): the per-pair tile-local-mean
    // recompute in pack_rec needs tt/l1_tiles_x and tt%l1_tiles_x. l1_tiles_x is
    // a RUNTIME arg, so the compiler cannot strength-reduce it and emits a soft
    // __udivmodsi4 PER KEPT PAIR on the divider-less NCRISC (this is the dominant
    // sort_bucket_emit zone, every pair is compute-exposed once blendrec is
    // cached). When the tile grid is a power of two (e.g. 1024px/32 => 32 tiles
    // per row) the divide is EXACTLY a right shift and the modulo EXACTLY a mask
    // — BIT-IDENTICAL for unsigned (same quotient/remainder, no rounding). Detect
    // the power-of-two case ONCE here; non-power-of-two grids fall back to the
    // exact divmod so output is unchanged for any tiles_x.
    const bool tx_is_pow2 = (l1_tiles_x != 0u) && ((l1_tiles_x & (l1_tiles_x - 1u)) == 0u);
    uint32_t tx_shift = 0u;
    if (tx_is_pow2) { uint32_t v = l1_tiles_x; while (v > 1u) { v >>= 1; tx_shift++; } }
    const uint32_t tx_mask = l1_tiles_x - 1u;

    constexpr auto gids_args  = TensorAccessorArgs<0>();
    constexpr auto tids_args  = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto keep_args  = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto depth_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto bin2d_args = TensorAccessorArgs<depth_args.next_compile_time_args_offset()>();
    constexpr auto keys_args  = TensorAccessorArgs<bin2d_args.next_compile_time_args_offset()>();
    constexpr auto ids_args   = TensorAccessorArgs<keys_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto tile_recs_args= TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto recbase_args  = TensorAccessorArgs<tile_recs_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args  = TensorAccessorArgs<recbase_args.next_compile_time_args_offset()>();
    constexpr auto l1_base_args  = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    // iter-138: overflow region (PACK2, 64B page) + per-(core,tile) overflow base row.
    constexpr auto l1_ov_args      = TensorAccessorArgs<l1_base_args.next_compile_time_args_offset()>();
    constexpr auto l1_ov_base_args = TensorAccessorArgs<l1_ov_args.next_compile_time_args_offset()>();

    const auto gids_acc  = TensorAccessor(gids_args,  gids_addr,  PAGE_BYTES);
    const auto tids_acc  = TensorAccessor(tids_args,  tids_addr,  PAGE_BYTES);
    const auto keep_acc  = TensorAccessor(keep_args,  keep_addr,  PAGE_BYTES);
    const auto depth_acc = TensorAccessor(depth_args, depth_addr, PAGE_BYTES);
    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto keys_acc  = TensorAccessor(keys_args,  keys_addr,  PAGE_BYTES);
    const auto ids_acc   = TensorAccessor(ids_args,   ids_addr,   PAGE_BYTES);
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    // tile_recs (dense 64B record buffer) is parsed to keep the accessor offsets
    // aligned with the host's CT-arg order, but the blend reader serves every
    // tile from the 32B buf_l1_recs (L1_RECORD) and never reads tile_recs, so it
    // is intentionally unwritten here.
    const auto tile_recs_acc= TensorAccessor(tile_recs_args, tile_recs_addr, PAGE_BYTES);
    const auto hist_rows_acc = TensorAccessor(recbase_args, hist_rows_addr, PAGE_BYTES);
    // T-C: per-core histogram of [page_start, pg_mid) (same interleaved layout).
    const auto h0_rows_acc = TensorAccessor(recbase_args, h0_rows_addr, PAGE_BYTES);
    (void)tile_recs_acc;
    // 32B record slots, REC_PAGE_RECS per REC_PAGE_BYTES DRAM page (task #86;
    // was PACK2, two per 64B page): slot s => page s/64, byte (s%64)*32.
    const auto l1_recs_acc  = TensorAccessor(l1_recs_args,  l1_recs_addr,  REC_PAGE_BYTES);
    const auto l1_base_acc  = TensorAccessor(l1_base_args,  l1_base_addr,  PAGE_BYTES);
    // iter-138: overflow region uses the same page layout as buf_l1_recs.
    const bool l1_ov_enabled = (l1_ov_addr != 0u) && (l1_ov_base_addr != 0u);
    const auto l1_ov_acc      = TensorAccessor(l1_ov_args,      l1_ov_addr,      REC_PAGE_BYTES);
    const auto l1_ov_base_acc = TensorAccessor(l1_ov_base_args, l1_ov_base_addr, PAGE_BYTES);

    // CB layout (declared in sort_device.cpp binning program):
    //   0 gid_in (64B)  1 tid_in (64B)  2 keep_in (64B)  3 depth (64B)
    //   4 row    (MAX_BIN_TILES*4: hist out / base in)
    //   5 cur    (MAX_BIN_TILES*4: local per-tile cursor)
    //   6 off    (MAX_BIN_TILES*4: local per-tile L1 offset)
    //   7 ksort  (BIN_LOCAL_MAX*4: L1 counting-sort keys)
    //   8 isort  (BIN_LOCAL_MAX*4: L1 counting-sort ids)
    // CB_KS / CB_IS are the core's shared counting-sort regions; every other
    // CB is this mover's private copy (mover 0 at id + MOVER0_CB_OFFSET).
    const uint32_t cbo = (mover == 0) ? MOVER0_CB_OFFSET : 0u;
    const uint32_t CB_GID = 0 + cbo, CB_TID = 1 + cbo, CB_KEEP = 2 + cbo, CB_DEP = 3 + cbo,
                   CB_ROW = 4 + cbo, CB_CUR = 5 + cbo, CB_OFF = 6 + cbo;
    constexpr uint32_t CB_KS = 7, CB_IS = 8;
    const uint32_t CB_REC = 9 + cbo;        // 64B blendrec staging (read page g, +depth, write bucket)
    const uint32_t CB_L1BASE = 11 + cbo;    // per-(core,tile) L1 slot base row (= t*BUCKET_FIT + prefix)
    const uint32_t CB_L1SCRATCH = 12 + cbo; // 32B staging buffer for pack → noc write
    const uint32_t CB_PACKOC = 13 + cbo;    // iter 132: 64B-per-gaussian blendrec page write-back ring (publishes packed op/color)
    const uint32_t CB_L1OVBASE = 14 + cbo;  // iter-138: per-(core,tile) overflow slot base row (sentinel = non-overflow tile)
    const uint32_t CB_RING = 15 + cbo;      // task #100: per-tile record staging runs + start cursors

    const uint32_t gid_l1  = get_write_ptr(CB_GID);
    const uint32_t tid_l1  = get_write_ptr(CB_TID);
    const uint32_t keep_l1 = get_write_ptr(CB_KEEP);
    const uint32_t dep_l1  = get_write_ptr(CB_DEP);
    const uint32_t row_l1  = get_write_ptr(CB_ROW);

    auto gidp  = reinterpret_cast<volatile int32_t*>(gid_l1);
    auto tidp  = reinterpret_cast<volatile int32_t*>(tid_l1);
    auto keepp = reinterpret_cast<volatile int32_t*>(keep_l1);
    auto depp  = reinterpret_cast<volatile uint32_t*>(dep_l1);
    auto rowp  = reinterpret_cast<volatile uint32_t*>(row_l1);

    const uint32_t row_pages = (num_tiles + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
    const uint32_t base_page = core_id * (stride / ELEMS_PER_PAGE);
    const uint32_t pg_lo = page_start;
    const uint32_t pg_hi = page_start + page_count;

    if (mode == 0) {
        DeviceZoneScopedN("sort_bin_hist");
        // ── count: per-tile histogram of kept pairs ─────────────────────
        // Reads are batched CNT_BATCH pages per barrier (one exposed NoC round
        // trip per batch instead of per 16 pairs), staged in the counting-sort
        // regions, which the count pass does not otherwise use.
        constexpr uint32_t CNT_BATCH = 32u;
        const uint32_t btid_l1 = get_write_ptr(CB_KS);
        const uint32_t bkeep_l1 = get_write_ptr(CB_IS);
        auto btidp = reinterpret_cast<volatile int32_t*>(btid_l1);
        auto bkeepp = reinterpret_cast<volatile int32_t*>(bkeep_l1);
        for (uint32_t t = 0; t < num_tiles; t++) rowp[t] = 0;
        // T-C: with the scatter split, snapshot the running histogram at pg_mid
        // — mover 0's kept count per tile, i.e. mover 1's cursor start — into
        // h0_rows, then keep counting. Batches stop at the split so the snapshot
        // covers exactly [pg_lo, pg_mid). One inline loop on purpose: wrapping it
        // in a by-reference lambda called twice cost +0.22 ms on this pass
        // (measured, yyzo-bh-07).
        const uint32_t split = dual ? pg_mid : pg_hi + 1u;  // unreachable when single
        auto snapshot_h0 = [&]() {
            for (uint32_t pp = 0; pp < row_pages; pp++) {
                noc_async_write(row_l1 + pp * PAGE_BYTES,
                                get_noc_addr(base_page + pp, h0_rows_acc), PAGE_BYTES);
            }
            noc_async_write_barrier();  // row_l1 keeps counting below
        };
        if (split == pg_lo) snapshot_h0();
        for (uint32_t pg0 = pg_lo; pg0 < pg_hi;) {
            const uint32_t lim = (pg0 < split && split < pg_hi) ? split : pg_hi;
            const uint32_t nb = (lim - pg0 < CNT_BATCH) ? (lim - pg0) : CNT_BATCH;
            for (uint32_t b = 0; b < nb; b++) {
                noc_async_read(get_noc_addr(pg0 + b, tids_acc),
                               btid_l1 + b * PAGE_BYTES, PAGE_BYTES);
                noc_async_read(get_noc_addr(pg0 + b, keep_acc),
                               bkeep_l1 + b * PAGE_BYTES, PAGE_BYTES);
            }
            noc_async_read_barrier();
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t pg = pg0 + b;
                const uint32_t e0 = b * ELEMS_PER_PAGE;
                for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                    const uint32_t p = pg * ELEMS_PER_PAGE + j;
                    if (p >= P) break;
                    if (bkeepp[e0 + j] == 0) continue;
                    rowp[static_cast<uint32_t>(btidp[e0 + j])]++;
                }
            }
            pg0 += nb;
            if (pg0 == split) snapshot_h0();
        }
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_write(row_l1 + pp * PAGE_BYTES,
                            get_noc_addr(base_page + pp, bin2d_acc), PAGE_BYTES);
        }
        noc_async_write_barrier();
        return;
    }

    DeviceZoneScopedN("sort_bucket_emit");
    // ── scatter ─────────────────────────────────────────────────────────
    // Load this core's global base row.
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, bin2d_acc),
                       row_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
    // Load this core's L1 slot base row (per-tile slot index = t*BUCKET_FIT + prefix).
    const uint32_t l1base_l1 = get_write_ptr(CB_L1BASE);
    auto l1basep = reinterpret_cast<volatile uint32_t*>(l1base_l1);
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, l1_base_acc),
                       l1base_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
    // iter-138: load this core's per-(core,tile) overflow slot base row. Entry is
    // the absolute slot in the overflow region (sentinel 0xFFFFFFFF for tiles that
    // are NOT pre-packed overflow tiles ⇒ keep the buf_l1_recs bucket path).
    const uint32_t ov_base_l1 = get_write_ptr(CB_L1OVBASE);
    auto ov_basep = reinterpret_cast<volatile uint32_t*>(ov_base_l1);
    if (l1_ov_enabled) {
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_read(get_noc_addr(base_page + pp, l1_ov_base_acc),
                           ov_base_l1 + pp * PAGE_BYTES, PAGE_BYTES);
        }
        noc_async_read_barrier();
    }
    // The local slot cursor reuses curp[t]: it gets reset to 0 in the prefix loop
    // below, then incremented in the scatter loop alongside the (key,id) write.
    const uint32_t l1_scratch = get_write_ptr(CB_L1SCRATCH);

    const uint32_t cur_l1 = get_write_ptr(CB_CUR);
    const uint32_t off_l1 = get_write_ptr(CB_OFF);
    auto curp = reinterpret_cast<volatile uint32_t*>(cur_l1);
    auto offp = reinterpret_cast<volatile uint32_t*>(off_l1);

    // Sub-pass 1: local per-tile kept count -> curp (reused as scratch count).
    // With the dual-mover split this must be the WHOLE core's row (the host only
    // splits when the count pass wrote it): a recount would see one sub-range.
    if (hist_rows_addr != 0u) {
        // The count pass wrote this core's row; one 4 KB read replaces a second
        // full scan of the core's tid/keep pages.
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_read(get_noc_addr(base_page + pp, hist_rows_acc),
                           cur_l1 + pp * PAGE_BYTES, PAGE_BYTES);
        }
        noc_async_read_barrier();
    } else {
        for (uint32_t t = 0; t < num_tiles; t++) curp[t] = 0;
        for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
            noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
            noc_async_read_barrier();
            for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                const uint32_t p = pg * ELEMS_PER_PAGE + j;
                if (p >= P) break;
                if (keepp[j] == 0) continue;
                curp[static_cast<uint32_t>(tidp[j])]++;
            }
        }
    }
    const uint32_t ks_l1 = get_write_ptr(CB_KS);
    const uint32_t is_l1 = get_write_ptr(CB_IS);
    auto ksp = reinterpret_cast<volatile uint32_t*>(ks_l1);
    auto isp = reinterpret_cast<volatile uint32_t*>(is_l1);

    // PAGE-ALIGNED exclusive prefix over tiles -> local L1 block offsets. Each
    // tile's L1 block is rounded up to a whole page so the write-out can emit
    // only whole-page, page-aligned DRAM transfers (no sub-page writes -> no
    // multi-writer/same-chunk DRAM write race).
    uint32_t run = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        offp[t] = run;
        const uint32_t c = curp[t];
        run += ((c + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
    }
    // Tiles this mover writes out: all of them single-mover; with the split,
    // mover 0 takes the tiles whose blocks start in the first half of the
    // core's pages and mover 1 the rest (both compute the same t_split).
    uint32_t t_split = num_tiles;
    if (dual) {
        const uint32_t half = run / 2u;
        t_split = 0;
        while (t_split < num_tiles && offp[t_split] < half) t_split++;
    }
    const uint32_t wt_lo = (dual && mover == 1) ? t_split : 0u;
    const uint32_t wt_hi = (dual && mover == 0) ? t_split : num_tiles;
    // Pre-fill the tail slots of each written-out tile's page-block with the max
    // key (0xffffffff) / 0 id: padding the stable radix pushes to the end of the
    // tile. Only the tails: every slot before them is written by the fill below
    // (by either mover), so this leaves the same bytes as a full pre-fill and
    // never races the peer mover's entries.
    for (uint32_t t = wt_lo; t < wt_hi; t++) {
        const uint32_t end = (t + 1u < num_tiles) ? offp[t + 1u] : run;
        for (uint32_t i = offp[t] + curp[t]; i < end; i++) { ksp[i] = 0xffffffffu; isp[i] = 0u; }
    }
    // Local cursors: 0, or for mover 1 of the split, mover 0's kept count per
    // tile (the count pass's snapshot at pg_mid), so its pairs land right
    // after mover 0's in every per-tile block, record bucket and overflow run.
    if (dual && mover == 1) {
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_read(get_noc_addr(base_page + pp, h0_rows_acc),
                           cur_l1 + pp * PAGE_BYTES, PAGE_BYTES);
        }
        noc_async_read_barrier();
    } else {
        for (uint32_t t = 0; t < num_tiles; t++) curp[t] = 0;
    }

    int32_t dep_cached_page = -1;

    // ── Sub-pass 2 (task #100 rewrite) ──────────────────────────────────
    // Counting-sort the kept pairs into L1 (key = depth, id = g) and emit each
    // pair's 32B record. Knobs (host env, see env_config.h):
    //   EMIT_PB     pair pages per read batch. 1: one page per barrier and its
    //               blendrec reads under a second barrier (the pre-#100 loop).
    //               >1: batch k+1's pair pages and blendrec reads are in flight
    //               while batch k packs (3 pair buffers, 2 blendrec rings).
    //   EMIT_RING   R records per per-tile L1 staging run. Run slots are aligned
    //               to absolute record slots (slot & (R-1)), so a full run is one
    //               R*32 B write inside one 2 KB record page; partial runs at a
    //               mover's first/last slot of a tile are clipped to its own
    //               slots. 0: one 32 B write per record (pre-#100).
    //   EMIT_PUBOC  the gather published op/color UNORM16 (blendrec[10], [11])
    //               and the depth key ([12]): no pack, no 16 B write-back, no
    //               depth pages.
    // Pairs are gaussian-major, so consecutive kept pairs share g; blendrec[g]
    // is read once per run of equal g (prefetched per batch) and the
    // per-gaussian words are held in registers. The bytes written to every
    // output are identical for every knob setting.
    constexpr uint32_t PB = EMIT_PB;
    constexpr bool PIPE = PB > 1u;
    constexpr uint32_t NPBUF = PIPE ? 3u : 1u;
    constexpr uint32_t BATCH_ELEMS = PB * ELEMS_PER_PAGE;
    constexpr uint32_t R = EMIT_RING;
    constexpr bool PUBOC = EMIT_PUBOC != 0u;
    static_assert(PB >= 1u && PB <= 16u, "EMIT_PB in 1..16");
    static_assert(R == 0u || ((R & (R - 1u)) == 0u && R <= 16u && REC_PAGE_RECS % R == 0u),
                  "EMIT_RING: power of two dividing the record page");
    const bool ring_on = (R != 0u) && num_tiles <= EMIT_RING_TILES;

    constexpr uint32_t REC_BATCH = 16u;
    const uint32_t rec_cache_l1 = get_write_ptr(CB_REC);  // PB*16 x 64B blendrec ring(s)
    volatile uint32_t* cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1);
    int32_t blendrec_cached_g = -1;
    uint32_t brec_l1_slot[REC_BATCH];  // abs slot (in buf_l1_recs OR overflow region; 0xFFFFFFFF = drop)
    uint32_t brec_is_ov[REC_BATCH];    // iter-138: 1 ⇒ slot is in the overflow region, 0 ⇒ buf_l1_recs
    uint32_t nbrec = 0;
    // EMIT_RING=0: records stage in l1_scratch and flush REC_BATCH 32B writes
    // at a time. Each targets either the per-tile buf_l1_recs bucket (in-budget
    // tiles) or the compact overflow region (overflow tiles within the
    // materialize L1 cap); both use the 2 KB record page layout.
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        if constexpr ((EMIT_ABLATE & 1u) != 0u) { nbrec = 0; return; }
        for (uint32_t b = 0; b < nbrec; b++) {
            // Skip the 32B scatter for over-cap overflow records (gather fallback).
            if (brec_l1_slot[b] == 0xFFFFFFFFu) continue;
            const uint32_t slot = brec_l1_slot[b];
            const uint32_t page = slot / REC_PAGE_RECS;
            const uint32_t half_off = (slot % REC_PAGE_RECS) * 32u;
            if (brec_is_ov[b]) {
                noc_async_write(l1_scratch + b * 32u,
                                get_noc_addr(page, l1_ov_acc) + half_off,
                                32u);
            } else {
                noc_async_write(l1_scratch + b * 32u,
                                get_noc_addr(page, l1_recs_acc) + half_off,
                                32u);
            }
        }
        noc_async_writes_flushed();  // T-B(4): staging reuse only needs the data sent
        nbrec = 0;
    };

    // EMIT_RING: per-tile staging runs at ring_l1 + (t*R + slot%R)*32, and each
    // tile's first local cursor (0, or mover 0's count for mover 1 of the split)
    // in startp[t]: a run never writes below this mover's first slot.
    const uint32_t ring_l1 = (R != 0u) ? get_write_ptr(CB_RING) : 0u;
    auto startp = reinterpret_cast<volatile uint32_t*>(ring_l1 + EMIT_RING_TILES * R * 32u);
    if (ring_on) {
        for (uint32_t t = 0; t < num_tiles; t++) startp[t] = curp[t];
    }
    auto flush_run = [&](uint32_t t, uint32_t last, uint32_t ovb) {
        if constexpr ((EMIT_ABLATE & 1u) != 0u) { return; }
        const uint32_t first_ok = ((ovb != 0xFFFFFFFFu) ? ovb : l1basep[t]) + startp[t];
        const uint32_t grp = last & ~(R - 1u);
        const uint32_t s0 = grp > first_ok ? grp : first_ok;
        const uint32_t src = ring_l1 + (t * R + (s0 & (R - 1u))) * 32u;
        const uint32_t off = (s0 % REC_PAGE_RECS) * 32u;
        const uint32_t bytes = (last + 1u - s0) * 32u;
        if (ovb != 0xFFFFFFFFu) {
            noc_async_write(src, get_noc_addr(s0 / REC_PAGE_RECS, l1_ov_acc) + off, bytes);
        } else {
            noc_async_write(src, get_noc_addr(s0 / REC_PAGE_RECS, l1_recs_acc) + off, bytes);
        }
    };

    // iter 132 (EMIT_PUBOC=0): publish the per-gaussian packed op/color words
    // (inv_opr, inv_cgb) into blendrec[10],[11] so the depth-sorted materialize
    // overflow gather COPIES them. 16B write at byte-offset 32 = words
    // [8,9,10,11] (16B is the DRAM write granule; 8B/4B splats did not land);
    // words 8,9 are re-written with their original gather values, so a gaussian
    // processed by two cores writes byte-identical 16B. Staged in a ring (one
    // flush per batch), mirroring flush_recs.
    constexpr uint32_t PACKOC_BATCH = 16u;
    constexpr uint32_t PACKOC_ENT_W = 4u;   // 16B chunk = 4 u32 (words 8,9,10,11)
    const uint32_t packoc_l1 = get_write_ptr(CB_PACKOC);
    auto packocp = reinterpret_cast<volatile uint32_t*>(packoc_l1);
    uint32_t packoc_g[PACKOC_BATCH];
    uint32_t n_packoc = 0;
    auto flush_packoc = [&]() {
        if (n_packoc == 0) return;
        if constexpr ((EMIT_ABLATE & 2u) != 0u) { n_packoc = 0; return; }
        for (uint32_t b = 0; b < n_packoc; b++) {
            noc_async_write(packoc_l1 + b * (PACKOC_ENT_W * 4u),
                            get_noc_addr(packoc_g[b], blendrec_acc) + 32u,
                            PACKOC_ENT_W * 4u);
        }
        noc_async_writes_flushed();  // T-B(4): staging reuse only needs the data sent
        n_packoc = 0;
    };

    // 32B record (M0): [0..2] fp32 cov a,b,c (full fp32: the blend's det
    // a*c - b*b needs it) [3] u32 depth key [4,5] fp32 tile-local mean x,y
    // [6] unorm16 op,r [7] unorm16 g,b. 64B blendrec (fp32 words): 0..2 cov,
    // 3,4 mean, 5 op, 6..8 color, 9 zero, 10,11 packed op/color, 12 depth key
    // (10..12 only with EMIT_PUBOC from the gather; otherwise 10,11 from the
    // write-back above).
    // iter-128: everything but the tile-local mean is per GAUSSIAN and is held
    // in registers (inv_*); per pair only the tile origin is subtracted.
    uint32_t inv_cov0 = 0, inv_cov1 = 0, inv_cov2 = 0, inv_depth = 0,
             inv_opr = 0, inv_cgb = 0;
    float inv_mx = 0.0f, inv_my = 0.0f;
    uint32_t inv_mx_bits = 0, inv_my_bits = 0;
    // Tile-local mean y per gaussian: its pairs walk its tile rectangle row by
    // row, so my only changes with the tile row.
    uint32_t c_ty = 0xFFFFFFFFu, c_my_bits = 0;
    auto pack_invariants = [&](uint32_t depth_key) {
        if constexpr ((EMIT_ABLATE & 4u) != 0u) { inv_depth = depth_key; return; }
        inv_cov0 = cachep[0];
        inv_cov1 = cachep[1];
        inv_cov2 = cachep[2];
        inv_depth = depth_key;
        inv_mx_bits = cachep[3];
        inv_my_bits = cachep[4];
        __builtin_memcpy(&inv_mx, &inv_mx_bits, 4);
        __builtin_memcpy(&inv_my, &inv_my_bits, 4);
        c_ty = 0xFFFFFFFFu;
        if constexpr (PUBOC) {
            inv_opr = cachep[10];
            inv_cgb = cachep[11];
        } else {
            // T-B(3): integer UNORM16 (bit-exact; NaN keeps the float path).
            inv_opr = sort_bin_fp32::to_unorm16(cachep[5]) |
                      (sort_bin_fp32::to_unorm16(cachep[6]) << 16);
            inv_cgb = sort_bin_fp32::to_unorm16(cachep[7]) |
                      (sort_bin_fp32::to_unorm16(cachep[8]) << 16);
        }
    };
    auto pack_rec = [&](volatile uint32_t* p32, uint32_t tt) {
        if constexpr ((EMIT_ABLATE & 4u) != 0u) { (void)p32; (void)tt; return; }
        // tx = tt % l1_tiles_x, ty = tt / l1_tiles_x (shift/mask on a
        // power-of-two grid). Integer fl(mean - origin) (bit-exact); float
        // only outside sort_bin_fp32::sub_int's range.
        const uint32_t txi = tx_is_pow2 ? (tt & tx_mask) : (tt % l1_tiles_x);
        const uint32_t tyi = tx_is_pow2 ? (tt >> tx_shift) : (tt / l1_tiles_x);
        uint32_t mx_bits;
        if (!sort_bin_fp32::sub_int(inv_mx_bits, txi * L1_TILE_SIZE, &mx_bits)) {
            const float mx = inv_mx - static_cast<float>(txi * L1_TILE_SIZE);
            __builtin_memcpy(&mx_bits, &mx, 4);
        }
        if (tyi != c_ty) {
            c_ty = tyi;
            if (!sort_bin_fp32::sub_int(inv_my_bits, tyi * L1_TILE_SIZE, &c_my_bits)) {
                const float my = inv_my - static_cast<float>(tyi * L1_TILE_SIZE);
                __builtin_memcpy(&c_my_bits, &my, 4);
            }
        }
        p32[0] = inv_cov0;
        p32[1] = inv_cov1;
        p32[2] = inv_cov2;
        p32[3] = inv_depth;
        p32[4] = mx_bits;
        p32[5] = c_my_bits;
        p32[6] = inv_opr;
        p32[7] = inv_cgb;
    };

    // Pair-page staging: buffer i of NPBUF at {gid,tid,keep}_l1 + i*PB*64.
    auto issue_pairs = [&](uint32_t pg0, uint32_t nb, uint32_t buf) {
        const uint32_t o = buf * PB * PAGE_BYTES;
        for (uint32_t b = 0; b < nb; b++) {
            noc_async_read(get_noc_addr(pg0 + b, gids_acc), gid_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg0 + b, tids_acc), tid_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg0 + b, keep_acc), keep_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
        }
    };
    // Read the blendrec page of every run of equal kept g in the batch into
    // ring `ring` (slot k = the k-th g change, the same test the pack loop
    // uses). scan_g carries the last kept g across batches.
    int32_t scan_g = -1;
    auto issue_brec = [&](uint32_t pg0, uint32_t nb, uint32_t buf, uint32_t ring) -> uint32_t {
        if constexpr ((EMIT_ABLATE & 8u) != 0u) { return 0u; }
        const uint32_t e0 = buf * BATCH_ELEMS;
        const uint32_t dst0 = rec_cache_l1 + ring * BATCH_ELEMS * PAGE_BYTES;
        uint32_t n_pf = 0;
        const uint32_t n_el = nb * ELEMS_PER_PAGE;
        const uint32_t p0 = pg0 * ELEMS_PER_PAGE;
        for (uint32_t j = 0; j < n_el; j++) {
            if (p0 + j >= P) break;
            if (keepp[e0 + j] == 0) continue;
            const int32_t gj = gidp[e0 + j];
            if (gj != scan_g) {
                noc_async_read(get_noc_addr(static_cast<uint32_t>(gj), blendrec_acc),
                               dst0 + n_pf * PAGE_BYTES, PAGE_BYTES);
                n_pf++;
                scan_g = gj;
            }
        }
        return n_pf;
    };

    auto process_batch = [&](uint32_t pg0, uint32_t nb, uint32_t buf, uint32_t ring) {
        const uint32_t e0 = buf * BATCH_ELEMS;
        const uint32_t ring0 = rec_cache_l1 + ring * BATCH_ELEMS * PAGE_BYTES;
        const uint32_t n_el = nb * ELEMS_PER_PAGE;
        const uint32_t p0 = pg0 * ELEMS_PER_PAGE;
        uint32_t rec_slot = 0;
        for (uint32_t j = 0; j < n_el; j++) {
            if (p0 + j >= P) break;
            if (keepp[e0 + j] == 0) continue;
            const uint32_t g = static_cast<uint32_t>(gidp[e0 + j]);
            const uint32_t t = static_cast<uint32_t>(tidp[e0 + j]);
            if (static_cast<int32_t>(g) != blendrec_cached_g) {
                cachep = reinterpret_cast<volatile uint32_t*>(ring0 + rec_slot * PAGE_BYTES);
                rec_slot++;
                blendrec_cached_g = static_cast<int32_t>(g);
                uint32_t key;
                if constexpr (PUBOC) {
                    key = cachep[12];
                } else {
                    // The depth key is the GAUSSIAN's: one page per 16 g.
                    const int32_t dpg = static_cast<int32_t>(g / ELEMS_PER_PAGE);
                    if (dpg != dep_cached_page) {
                        if constexpr ((EMIT_ABLATE & 16u) == 0u) {
                            noc_async_read(get_noc_addr(static_cast<uint32_t>(dpg), depth_acc),
                                           dep_l1, PAGE_BYTES);
                            noc_async_read_barrier();
                        }
                        dep_cached_page = dpg;
                    }
                    key = depp[g % ELEMS_PER_PAGE];
                }
                {
                    // Accumulating sub-zone (task #27): one summed duration per
                    // RISC per launch with TT_METAL_PROFILER_SUM=1, else empty.
                    DeviceZoneScopedSumN1("emit_pack_invariants");
                    pack_invariants(key);
                }
                if constexpr (!PUBOC) {
                    volatile uint32_t* ent = packocp + n_packoc * PACKOC_ENT_W;
                    ent[0] = cachep[8];   // original cb (fp32) — preserved
                    ent[1] = cachep[9];   // original word 9 — preserved
                    ent[2] = inv_opr;     // -> blendrec[10]
                    ent[3] = inv_cgb;     // -> blendrec[11]
                    packoc_g[n_packoc] = g;
                    n_packoc++;
                    if (n_packoc == PACKOC_BATCH) flush_packoc();
                }
            }
            // This pair's per-tile cursor and bases, each loaded from L1 once.
            const uint32_t ct = curp[t];
            const uint32_t li = offp[t] + ct;
            const uint32_t ovb = l1_ov_enabled ? ov_basep[t] : 0xFFFFFFFFu;
            // Record slot: overflow tiles (within the materialize L1 cap) pre-pack
            // the FULL tile into the overflow region at ov_basep[t] + cursor;
            // other tiles go to their buf_l1_recs bucket at l1basep[t] + cursor
            // (= t*FIT + prefix of earlier cores), clamped to [t*FIT, (t+1)*FIT):
            // records past the bucket are dropped (materialize gathers them).
            // (core, tile) regions are disjoint, so no two writers share a slot.
            uint32_t out_slot;
            if (ovb != 0xFFFFFFFFu) {
                out_slot = ovb + ct;
            } else {
                const uint32_t l1_slot = l1basep[t] + ct;
                out_slot = (l1_slot < (t + 1u) * l1_bucket_fit) ? l1_slot : 0xFFFFFFFFu;
            }
            if (ring_on) {
                if (out_slot != 0xFFFFFFFFu) {
                    const uint32_t ri = out_slot & (R - 1u);
                    // Entry 0 starts a new run: the previous run of this tile
                    // must have left L1.
                    if (ri == 0u) noc_async_writes_flushed();
                    {
                        DeviceZoneScopedSumN2("emit_pack_rec");
                        pack_rec(reinterpret_cast<volatile uint32_t*>(
                                     ring_l1 + (t * R + ri) * 32u), t);
                    }
                    if (ri == R - 1u) flush_run(t, out_slot, ovb);
                }
            } else {
                brec_l1_slot[nbrec] = out_slot;
                brec_is_ov[nbrec] = (ovb != 0xFFFFFFFFu) ? 1u : 0u;
                if (out_slot != 0xFFFFFFFFu) {
                    DeviceZoneScopedSumN2("emit_pack_rec");
                    pack_rec(reinterpret_cast<volatile uint32_t*>(l1_scratch + nbrec * 32u), t);
                }
                nbrec++;
                if (nbrec == REC_BATCH) flush_recs();
            }
            curp[t] = ct + 1u;
            ksp[li] = inv_depth;
            isp[li] = g;
        }
    };

    if constexpr (!PIPE) {
        for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
            issue_pairs(pg, 1u, 0u);
            noc_async_read_barrier();
            if (issue_brec(pg, 1u, 0u, 0u) != 0u) noc_async_read_barrier();
            if constexpr ((EMIT_ABLATE & 32u) != 0u) continue;
            process_batch(pg, 1u, 0u, 0u);
        }
    } else if (pg_lo < pg_hi) {
        // Batch k: pages [pg_lo + k*PB, +nb), pair buffer k%3, blendrec ring k%2.
        auto nb_at = [&](uint32_t pg) { return (pg_hi - pg < PB) ? (pg_hi - pg) : PB; };
        uint32_t pg = pg_lo, nb = nb_at(pg), buf = 0u, ring = 0u;
        issue_pairs(pg, nb, buf);
        noc_async_read_barrier();
        issue_brec(pg, nb, buf, ring);
        uint32_t pg_n = pg + nb;
        if (pg_n < pg_hi) issue_pairs(pg_n, nb_at(pg_n), 1u);
        while (pg < pg_hi) {
            noc_async_read_barrier();  // blendrec of batch k, pairs of batch k+1
            const uint32_t buf_n = (buf == 2u) ? 0u : buf + 1u;
            if (pg_n < pg_hi) {
                const uint32_t nb_n = nb_at(pg_n);
                issue_brec(pg_n, nb_n, buf_n, ring ^ 1u);
                const uint32_t pg_nn = pg_n + nb_n;
                if (pg_nn < pg_hi) issue_pairs(pg_nn, nb_at(pg_nn), (buf_n == 2u) ? 0u : buf_n + 1u);
            }
            if constexpr ((EMIT_ABLATE & 32u) == 0u) process_batch(pg, nb, buf, ring);
            pg = pg_n;
            nb = (pg < pg_hi) ? nb_at(pg) : 0u;
            pg_n = pg + nb;
            buf = buf_n;
            ring ^= 1u;
        }
    }
    flush_recs();    // EMIT_RING=0: drain the partial final batch
    flush_packoc();  // iter 132: drain the partial final packed-op/color batch
    if (ring_on) {
        // Drain each tile's partial final run (full runs flushed in the loop).
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t st = startp[t], en = curp[t];
            if (en == st) continue;
            const uint32_t ovb = l1_ov_enabled ? ov_basep[t] : 0xFFFFFFFFu;
            uint32_t end_slot;
            if (ovb != 0xFFFFFFFFu) {
                end_slot = ovb + en;
            } else {
                const uint32_t cap = (t + 1u) * l1_bucket_fit;
                end_slot = l1basep[t] + en;
                if (end_slot > cap) end_slot = cap;
                if (end_slot <= l1basep[t] + st) continue;
            }
            const uint32_t last = end_slot - 1u;
            if ((last & (R - 1u)) != R - 1u) flush_run(t, last, ovb);
        }
    }

    if (dual) {
        // A tile block holds both movers' entries: publish this mover's fill
        // (fence: its L1 stores land before the flag) and wait for the peer's.
        // The waiter re-arms the peer's flag for the next launch.
        asm volatile("fence" ::: "memory");
        auto own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_own_id));
        auto peer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_peer_id));
        noc_semaphore_set(own, 1u);
        noc_semaphore_wait(peer, 1u);
        noc_semaphore_set(peer, 0u);
    }

    // Write each tile's page-aligned L1 block to DRAM as WHOLE PAGES. The block
    // base (rowp[t]) and L1 source (offp[t]) are both page-aligned, and the size
    // is a page multiple, so every transfer is a clean whole-page write to pages
    // this core owns EXCLUSIVELY. No sub-page writes and no shared pages -> no
    // multi-writer DRAM write race. Tail slots already hold max-key/0 padding
    // (pre-filled above), which the stable radix sorts to the end of the tile;
    // host compaction keeps only the real count. The block size comes from the
    // offsets (= ceil16(count)): mover 0's cursors stop at its own share.
    for (uint32_t t = wt_lo; t < wt_hi; t++) {
        const uint32_t end = (t + 1u < num_tiles) ? offp[t + 1u] : run;
        const uint32_t pages = (end - offp[t]) / ELEMS_PER_PAGE;
        if (pages == 0) continue;
        const uint32_t base_pg = rowp[t] / ELEMS_PER_PAGE;   // page-aligned base
        const uint32_t src = offp[t];                        // page-aligned L1 src
        for (uint32_t pp = 0; pp < pages; pp++) {
            const uint32_t soff = (src + pp * ELEMS_PER_PAGE) * 4;
            noc_async_write(ks_l1 + soff, get_noc_addr(base_pg + pp, keys_acc), PAGE_BYTES);
            noc_async_write(is_l1 + soff, get_noc_addr(base_pg + pp, ids_acc),  PAGE_BYTES);
        }
    }
    // T-B(4): one barrier for all tiles' write-out (was one per tile); it also
    // retires the flushed-but-unacked record and packoc writes above.
    noc_async_write_barrier();
}

