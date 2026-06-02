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
//
// COMPILE-TIME ARGS: 7 TensorAccessorArgs
//   gids, tids, keep, depth, bin2d, keys_out, ids_out.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("sort");  // Tracy device-timeline stage label (sort bin/count)
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
#ifdef BIN_DUMP
    const uint32_t dbg_addr    = get_arg_val<uint32_t>(14);
#endif
#ifdef BIN_EMIT_REC
    // Step T1 (GSPLAT_TT_TILE_BUCKET): scatter the FULL projected record into a
    // per-tile contiguous bucket in arbitrary (bin/gaussian) order, so the depth
    // sort can move into the per-tile L1 pass and NO random DRAM read is needed
    // to build the bucket. proj_m_blendrec is 1 record (64B) per page (page==g);
    // tile_recs mirrors the keys/ids layout: record page e ↔ keys/ids element e.
    const uint32_t blendrec_addr = get_arg_val<uint32_t>(15);
    const uint32_t tile_recs_addr= get_arg_val<uint32_t>(16);
    const uint32_t recbase_addr  = get_arg_val<uint32_t>(17);  // dense per-(core,tile) base
#endif

    constexpr auto gids_args  = TensorAccessorArgs<0>();
    constexpr auto tids_args  = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto keep_args  = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto depth_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto bin2d_args = TensorAccessorArgs<depth_args.next_compile_time_args_offset()>();
    constexpr auto keys_args  = TensorAccessorArgs<bin2d_args.next_compile_time_args_offset()>();
    constexpr auto ids_args   = TensorAccessorArgs<keys_args.next_compile_time_args_offset()>();
#ifdef BIN_EMIT_REC
    constexpr auto blendrec_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto tile_recs_args= TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto recbase_args  = TensorAccessorArgs<tile_recs_args.next_compile_time_args_offset()>();
#endif

    const auto gids_acc  = TensorAccessor(gids_args,  gids_addr,  PAGE_BYTES);
    const auto tids_acc  = TensorAccessor(tids_args,  tids_addr,  PAGE_BYTES);
    const auto keep_acc  = TensorAccessor(keep_args,  keep_addr,  PAGE_BYTES);
    const auto depth_acc = TensorAccessor(depth_args, depth_addr, PAGE_BYTES);
    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto keys_acc  = TensorAccessor(keys_args,  keys_addr,  PAGE_BYTES);
    const auto ids_acc   = TensorAccessor(ids_args,   ids_addr,   PAGE_BYTES);
#ifdef BIN_EMIT_REC
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    const auto tile_recs_acc= TensorAccessor(tile_recs_args, tile_recs_addr, PAGE_BYTES);
    const auto recbase_acc  = TensorAccessor(recbase_args,  recbase_addr,  PAGE_BYTES);
#endif

    // CB layout (declared in sort_device.cpp binning program):
    //   0 gid_in (64B)  1 tid_in (64B)  2 keep_in (64B)  3 depth (64B)
    //   4 row    (MAX_BIN_TILES*4: hist out / base in)
    //   5 cur    (MAX_BIN_TILES*4: local per-tile cursor)
    //   6 off    (MAX_BIN_TILES*4: local per-tile L1 offset)
    //   7 ksort  (BIN_LOCAL_MAX*4: L1 counting-sort keys)
    //   8 isort  (BIN_LOCAL_MAX*4: L1 counting-sort ids)
    constexpr uint32_t CB_GID = 0, CB_TID = 1, CB_KEEP = 2, CB_DEP = 3,
                       CB_ROW = 4, CB_CUR = 5, CB_OFF = 6, CB_KS = 7, CB_IS = 8;
#ifdef BIN_EMIT_REC
    constexpr uint32_t CB_REC = 9;     // 64B blendrec staging (read page g, +depth, write bucket)
    constexpr uint32_t CB_RECROW = 10; // dense per-(core,tile) record base row
#endif

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
        // ── count: per-tile histogram of kept pairs ─────────────────────
        for (uint32_t t = 0; t < num_tiles; t++) rowp[t] = 0;
        for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
            noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
            noc_async_read_barrier();
            for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                const uint32_t p = pg * ELEMS_PER_PAGE + j;
                if (p >= P) break;
                if (keepp[j] == 0) continue;
                rowp[static_cast<uint32_t>(tidp[j])]++;
            }
        }
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_write(row_l1 + pp * PAGE_BYTES,
                            get_noc_addr(base_page + pp, bin2d_acc), PAGE_BYTES);
        }
        noc_async_write_barrier();
        return;
    }

    // ── scatter ─────────────────────────────────────────────────────────
    // Load this core's global base row.
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, bin2d_acc),
                       row_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
#ifdef BIN_EMIT_REC
    // Load this core's DENSE record base row (same per-(core,tile) shape as bin2d).
    const uint32_t recrow_l1 = get_write_ptr(CB_RECROW);
    auto recrowp = reinterpret_cast<volatile uint32_t*>(recrow_l1);
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, recbase_acc),
                       recrow_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
#endif

    const uint32_t cur_l1 = get_write_ptr(CB_CUR);
    const uint32_t off_l1 = get_write_ptr(CB_OFF);
    auto curp = reinterpret_cast<volatile uint32_t*>(cur_l1);
    auto offp = reinterpret_cast<volatile uint32_t*>(off_l1);

    // Sub-pass 1: local per-tile kept count -> curp (reused as scratch count).
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
    const uint32_t ks_l1 = get_write_ptr(CB_KS);
    const uint32_t is_l1 = get_write_ptr(CB_IS);
    auto ksp = reinterpret_cast<volatile uint32_t*>(ks_l1);
    auto isp = reinterpret_cast<volatile uint32_t*>(is_l1);

    // PAGE-ALIGNED exclusive prefix over tiles -> local L1 block offsets; reset
    // curp to 0. Each tile's L1 block is rounded up to a whole page so the
    // write-out can emit only whole-page, page-aligned DRAM transfers (no
    // sub-page writes -> no multi-writer/same-chunk DRAM write race).
    uint32_t run = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        offp[t] = run;
        const uint32_t c = curp[t];
        run += ((c + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
        curp[t] = 0;
    }
    // Pre-fill the L1 counting-sort region with the max key (0xffffffff) / 0 id
    // so unfilled tail slots of each tile's page-block become padding the stable
    // radix pushes to the end of the tile.
    for (uint32_t i = 0; i < run; i++) { ksp[i] = 0xffffffffu; isp[i] = 0u; }

    int32_t dep_cached_page = -1;

#ifdef BIN_EMIT_REC
    // T1b: ring-buffered record scatter. Instead of a synchronous read+write
    // barrier per record (which serialized the bin stage at ~+20 ms), stage up
    // to REC_BATCH records in CB_REC, issue all reads under ONE barrier, inject
    // depth, then issue all writes under ONE barrier. The bucket scatter is off
    // the blend critical path, so this batching just hides the NoC latency.
    constexpr uint32_t REC_BATCH = 16u;
    const uint32_t rec_l1_base = get_write_ptr(CB_REC);
    uint32_t brec_key[REC_BATCH];
    uint32_t brec_page[REC_BATCH];
    uint32_t nbrec = 0;
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        noc_async_read_barrier();
        for (uint32_t b = 0; b < nbrec; b++) {
            const uint32_t slot = rec_l1_base + b * PAGE_BYTES;
            reinterpret_cast<volatile uint32_t*>(slot)[9] = brec_key[b];
            noc_async_write(slot, get_noc_addr(brec_page[b], tile_recs_acc), PAGE_BYTES);
        }
        noc_async_write_barrier();
        nbrec = 0;
    };
#endif

    // Sub-pass 2: counting-sort kept pairs into L1, grouped by tile (gaussian-
    // major within each tile). key = depth_bits[g], id = g.
    for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
        noc_async_read(get_noc_addr(pg, gids_acc), gid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
        noc_async_read_barrier();
        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            const uint32_t p = pg * ELEMS_PER_PAGE + j;
            if (p >= P) break;
            if (keepp[j] == 0) continue;
            const uint32_t g = static_cast<uint32_t>(gidp[j]);
            const uint32_t t = static_cast<uint32_t>(tidp[j]);
#ifdef BIN_NO_DEPTH
            const uint32_t key = 0u;
            (void)depp; (void)dep_l1; (void)dep_cached_page; (void)depth_acc;
#else
            const int32_t dpg = static_cast<int32_t>(g / ELEMS_PER_PAGE);
            if (dpg != dep_cached_page) {
                noc_async_read(get_noc_addr(static_cast<uint32_t>(dpg), depth_acc),
                               dep_l1, PAGE_BYTES);
                noc_async_read_barrier();
                dep_cached_page = dpg;
            }
            const uint32_t key = depp[g % ELEMS_PER_PAGE];
#endif
            const uint32_t li = offp[t] + curp[t];
#ifdef BIN_EMIT_REC
            // Scatter the full record to its per-tile bucket page. DENSE layout:
            // tile t's records occupy pages [starts[t], starts[t]+counts[t]); this
            // core's k-th kept pair for tile t goes to recrowp[t] + curp[t]
            // (recrowp[t] = starts[t] + prefix of cores < this core). Each record
            // is its own 64B page, so even dense, cores never share a page (no
            // race, no atomics). Read blendrec[g]
            // (page g), inject depth=key as the 10th field, write 64B. T1 keeps
            // it correctness-simple: one staging buffer, read+write barriered per
            // record (the scatter is off the blend critical path; T1b can ring-
            // buffer to pipeline if the bin stage cost matters).
            {
                // Stage into the ring; flush in batches (issue-ahead, 1 barrier
                // per REC_BATCH instead of 2 per record). rec_page captured with
                // the CURRENT curp[t] (matches the pre-increment dense layout).
                const uint32_t slot = rec_l1_base + nbrec * PAGE_BYTES;
                noc_async_read(get_noc_addr(g, blendrec_acc), slot, PAGE_BYTES);
                brec_key[nbrec] = key;
                brec_page[nbrec] = recrowp[t] + curp[t];  // DENSE bucket page
                nbrec++;
                if (nbrec == REC_BATCH) flush_recs();
            }
#endif
            curp[t] = curp[t] + 1;
            ksp[li] = key;
            isp[li] = g;
        }
    }
#ifdef BIN_EMIT_REC
    flush_recs();  // drain the partial final batch
#endif

#ifdef BIN_DUMP
    // Each core dumps its tile-0 summary into its own (already-consumed) bin2d
    // base-row page (base_page). Host reads back to reconstruct the cross-core
    // tile-0 layout: base, count, first L1 gid this core placed in tile 0.
    {
        const uint32_t dt = dbg_addr;  // dump_tile (passed via arg14 under BIN_DUMP)
        auto dp = reinterpret_cast<volatile uint32_t*>(gid_l1);  // reuse gid CB as staging
        dp[0] = rowp[dt];                          // global base for tile dt
        dp[1] = curp[dt];                          // this core's tile-dt count
        dp[2] = (curp[dt] > 0) ? isp[offp[dt]] : 0xffffffffu;  // first tile-dt gid in L1
        dp[3] = core_id;
        dp[4] = page_start;
        dp[5] = page_count;
        dp[6] = offp[dt];
        dp[7] = (curp[dt] > 1) ? isp[offp[dt] + 1] : 0xffffffffu;  // 2nd tile-dt gid
        for (uint32_t i = 8; i < ELEMS_PER_PAGE; i++) dp[i] = 0;
        noc_async_write(gid_l1, get_noc_addr(base_page, bin2d_acc), PAGE_BYTES);
        noc_async_write_barrier();
    }
#endif

    // Write each tile's page-aligned L1 block to DRAM as WHOLE PAGES. The block
    // base (rowp[t]) and L1 source (offp[t]) are both page-aligned, and the size
    // is a page multiple, so every transfer is a clean whole-page write to pages
    // this core owns EXCLUSIVELY. No sub-page writes and no shared pages -> no
    // multi-writer DRAM write race. Tail slots already hold max-key/0 padding
    // (pre-filled above), which the stable radix sorts to the end of the tile;
    // host compaction keeps only the real count.
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t n = curp[t];
        if (n == 0) continue;
        const uint32_t base_pg = rowp[t] / ELEMS_PER_PAGE;   // page-aligned base
        const uint32_t src = offp[t];                        // page-aligned L1 src
        const uint32_t pages = (n + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
        for (uint32_t pp = 0; pp < pages; pp++) {
            const uint32_t soff = (src + pp * ELEMS_PER_PAGE) * 4;
            noc_async_write(ks_l1 + soff, get_noc_addr(base_pg + pp, keys_acc), PAGE_BYTES);
            noc_async_write(is_l1 + soff, get_noc_addr(base_pg + pp, ids_acc),  PAGE_BYTES);
        }
        noc_async_write_barrier();
    }
}
