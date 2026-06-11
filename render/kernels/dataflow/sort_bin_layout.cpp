// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN layout — single-core post-count pass (S5.1 device bin layout).
//
// Replaces the host bridge: D2H(bin2d) + page-aligned base layout + LPT +
// H2D(bin2d/tmeta/tile_ids/bucket_meta/l1_rec_base). The host then reads back
// only the resident result buffers (no host histogram D2H, no host LPT).
//
// Mirrors host_bin_layout_from_hist + build_lpt in render/host/sort_device.cpp
// bit-for-bit. The histogram is read/written in whole 64B pages per core row
// (NOT one element per page) so the single-core pass stays cheap; the full
// histogram is never resident — each core row is streamed twice (count prefix,
// then base emit). recbase/bin2d_rec is the retired tile_recs path and is left
// unwritten (the live L1_RECORD scatter consumes l1_rec_base, populated here).
//
// RUNTIME ARGS
//   0: bin2d_addr  1: tmeta_addr  2: tile_ids_addr  3: lpt_meta_addr
//   4: tile_counts_addr  5: ctrl_addr  6: num_cores  7: num_tiles  8: stride
//   9: recbase_addr (unused)  10: bucket_meta_addr (0=skip)  11: tile_ranges_addr
//   12: bucket_fit (l1_record per-tile slot count)  13: l1_base_addr (0=skip)
//
// ctrl page: [0]=P_kept [1]=P_aligned [2]=max_pad_n [3]=status [4]=total_pages
// status 0=ok, 1=BIN_LOCAL_MAX, 2=MAX_TILE_ENTRIES

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t MAX_TILES = 2048;
constexpr uint32_t MAX_CORES = 128;
constexpr uint32_t BIN_LOCAL_MAX = 65536;
constexpr uint32_t MAX_TILE_ENTRIES = 32768;

inline uint32_t ceil_pages(uint32_t h) { return (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE; }

// Bulk-read row_pages contiguous 64B pages starting at page `base_pg` into L1.
template <typename AccT>
inline void read_pages(uint32_t base_pg, uint32_t row_pages, const AccT& acc, uint32_t l1) {
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_pg + pp, acc), l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
}

// Bulk-write row_pages contiguous 64B pages from L1 starting at page `base_pg`.
template <typename AccT>
inline void write_pages(uint32_t base_pg, uint32_t row_pages, const AccT& acc, uint32_t l1) {
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_write(l1 + pp * PAGE_BYTES, get_noc_addr(base_pg + pp, acc), PAGE_BYTES);
    }
    noc_async_write_barrier();
}

}  // namespace

void kernel_main() {
    const uint32_t bin2d_addr = get_arg_val<uint32_t>(0);
    const uint32_t tmeta_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_ids_addr = get_arg_val<uint32_t>(2);
    const uint32_t lpt_meta_addr = get_arg_val<uint32_t>(3);
    const uint32_t tile_counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t ctrl_addr = get_arg_val<uint32_t>(5);
    const uint32_t num_cores = get_arg_val<uint32_t>(6);
    const uint32_t num_tiles = get_arg_val<uint32_t>(7);
    const uint32_t stride = get_arg_val<uint32_t>(8);
    // arg 9 (recbase_addr) is the retired tile_recs base — intentionally unused.
    const uint32_t bucket_meta_addr = get_arg_val<uint32_t>(10);
    const uint32_t tile_ranges_addr = get_arg_val<uint32_t>(11);
    const uint32_t bucket_fit = get_arg_val<uint32_t>(12);
    const uint32_t l1_base_addr = get_arg_val<uint32_t>(13);

    constexpr auto bin2d_args = TensorAccessorArgs<0>();
    constexpr auto tmeta_args =
        TensorAccessorArgs<bin2d_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args =
        TensorAccessorArgs<tmeta_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args =
        TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();
    constexpr auto counts_args =
        TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto ctrl_args =
        TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto recbase_args =
        TensorAccessorArgs<ctrl_args.next_compile_time_args_offset()>();
    constexpr auto bucket_args =
        TensorAccessorArgs<recbase_args.next_compile_time_args_offset()>();
    constexpr auto ranges_args =
        TensorAccessorArgs<bucket_args.next_compile_time_args_offset()>();
    constexpr auto l1base_args =
        TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();

    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto tmeta_acc = TensorAccessor(tmeta_args, tmeta_addr, PAGE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, PAGE_BYTES);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, PAGE_BYTES);
    const auto counts_acc = TensorAccessor(counts_args, tile_counts_addr, PAGE_BYTES);
    const auto ctrl_acc = TensorAccessor(ctrl_args, ctrl_addr, PAGE_BYTES);
    const auto bucket_acc = TensorAccessor(bucket_args, bucket_meta_addr, PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, tile_ranges_addr, PAGE_BYTES);
    const auto l1base_acc = TensorAccessor(l1base_args, l1_base_addr, PAGE_BYTES);

    constexpr uint32_t CB_CTRL = 0;     // 64B ctrl staging
    constexpr uint32_t CB_SCRATCH = 1;  // 5*MAX_TILES + 5*MAX_CORES u32
    constexpr uint32_t CB_ROW = 2;      // one core's hist row (MAX_TILES u32)
    constexpr uint32_t CB_BIN = 3;      // one core's bin2d base row out (MAX_TILES u32)
    constexpr uint32_t CB_L1B = 4;      // one core's l1_rec_base row out (MAX_TILES u32)
    constexpr uint32_t CB_OUT = 5;      // per-tile staging (2*MAX_TILES u32)

    const uint32_t ctrl_l1 = get_write_ptr(CB_CTRL);
    const uint32_t scratch_l1 = get_write_ptr(CB_SCRATCH);
    const uint32_t row_l1 = get_write_ptr(CB_ROW);
    const uint32_t bin_l1 = get_write_ptr(CB_BIN);
    const uint32_t l1b_l1 = get_write_ptr(CB_L1B);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);

    auto* ctrl = reinterpret_cast<volatile uint32_t*>(ctrl_l1);
    auto* rowp = reinterpret_cast<volatile uint32_t*>(row_l1);
    auto* binp = reinterpret_cast<volatile uint32_t*>(bin_l1);
    auto* l1bp = reinterpret_cast<volatile uint32_t*>(l1b_l1);
    auto* outp = reinterpret_cast<volatile uint32_t*>(out_l1);

    auto* sc = reinterpret_cast<volatile uint32_t*>(scratch_l1);
    volatile uint32_t* counts = sc + 0u * MAX_TILES;
    volatile uint32_t* tpad = sc + 1u * MAX_TILES;   // tile_pad (=pad_pages*16)
    volatile uint32_t* starts = sc + 2u * MAX_TILES;
    volatile uint32_t* page_acc = sc + 3u * MAX_TILES;  // tile_start_page, then running page base
    volatile uint32_t* rec_acc = sc + 4u * MAX_TILES;   // running real-count base per tile

    const uint32_t pages_per_core = stride / ELEMS_PER_PAGE;
    const uint32_t row_pages = ceil_pages(num_tiles);
    const uint32_t row_span = row_pages * ELEMS_PER_PAGE;  // padded tiles in a row buffer

    for (uint32_t t = 0; t < num_tiles; t++) {
        counts[t] = 0;
        page_acc[t] = 0;  // accumulate tile_pad_pages here in pass 1
    }

    // ── Pass 1: stream each core row; per-tile count + padded-page sum, and the
    //    per-core padded total (BIN_LOCAL_MAX guard). ────────────────────────
    uint32_t max_core_padded = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        read_pages(c * pages_per_core, row_pages, bin2d_acc, row_l1);
        uint32_t core_padded = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = rowp[t];
            counts[t] += h;
            const uint32_t cp = ceil_pages(h);
            page_acc[t] += cp;
            core_padded += cp * ELEMS_PER_PAGE;
        }
        if (core_padded > max_core_padded) max_core_padded = core_padded;
    }

    uint32_t status = (max_core_padded > BIN_LOCAL_MAX) ? 1u : 0u;

    uint32_t cstart = 0;
    uint32_t apage = 0;
    uint32_t max_pad_n = 0;

    if (status == 0) {
        // ── Tile-major prefix: starts, tile_start_page, tile_pad, +
        //    counts/tmeta/ranges/bucket_meta resident writes. ────────────────
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t creal = counts[t];
            starts[t] = cstart;
            cstart += creal;
            const uint32_t pad_pages = page_acc[t];
            const uint32_t tsp = apage;
            page_acc[t] = tsp;  // becomes pass-2 running page base (init = tile_start_page)
            tpad[t] = pad_pages * ELEMS_PER_PAGE;
            apage += pad_pages;
            if (tpad[t] > max_pad_n) max_pad_n = tpad[t];
            rec_acc[t] = 0;
            // tmeta[t] = {tile_start_page, tile_pad}
            outp[t * 2 + 0] = tsp;
            outp[t * 2 + 1] = tpad[t];
        }
        if (max_pad_n > MAX_TILE_ENTRIES) status = 2;
    }

    const uint32_t P_kept = cstart;
    const uint32_t total_pages = apage > 0 ? apage : 1u;
    const uint32_t P_aligned = total_pages * ELEMS_PER_PAGE;

    if (status == 0) {
        // tmeta (already built in outp during the prefix loop).
        const uint32_t pad2 = ceil_pages(num_tiles * 2u) * ELEMS_PER_PAGE;
        for (uint32_t k = num_tiles * 2u; k < pad2; k++) outp[k] = 0;
        write_pages(0, ceil_pages(num_tiles * 2u), tmeta_acc, out_l1);

        // counts
        for (uint32_t t = 0; t < num_tiles; t++) outp[t] = counts[t];
        for (uint32_t k = num_tiles; k < row_span; k++) outp[k] = 0;
        write_pages(0, row_pages, counts_acc, out_l1);

        // tile_ranges = {starts[t], creal>0 ? starts+creal : starts}
        if (tile_ranges_addr != 0) {
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t creal = counts[t];
                outp[t * 2 + 0] = starts[t];
                outp[t * 2 + 1] = creal > 0 ? starts[t] + creal : starts[t];
            }
            for (uint32_t k = num_tiles * 2u; k < pad2; k++) outp[k] = 0;
            write_pages(0, ceil_pages(num_tiles * 2u), ranges_acc, out_l1);
        }

        // bucket_meta = {starts[t], creal} for creal>0 else {0,0}
        if (bucket_meta_addr != 0) {
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t creal = counts[t];
                outp[t * 2 + 0] = creal > 0 ? starts[t] : 0u;
                outp[t * 2 + 1] = creal > 0 ? creal : 0u;
            }
            for (uint32_t k = num_tiles * 2u; k < pad2; k++) outp[k] = 0;
            write_pages(0, ceil_pages(num_tiles * 2u), bucket_acc, out_l1);
        }

        // ── Pass 2: stream each core row again; emit bin2d base + l1_rec_base
        //    rows (whole-page bulk writes). page_acc/rec_acc carry the running
        //    per-(core,tile) prefix exactly as the host c-inner loop. ─────────
        for (uint32_t c = 0; c < num_cores; c++) {
            read_pages(c * pages_per_core, row_pages, bin2d_acc, row_l1);
            for (uint32_t t = 0; t < num_tiles; t++) {
                const uint32_t h = rowp[t];
                binp[t] = page_acc[t] * ELEMS_PER_PAGE;
                l1bp[t] = t * bucket_fit + rec_acc[t];
                rec_acc[t] += h;
                if (h > 0) page_acc[t] += ceil_pages(h);
            }
            for (uint32_t t = num_tiles; t < row_span; t++) {
                binp[t] = 0;
                l1bp[t] = 0;
            }
            write_pages(c * pages_per_core, row_pages, bin2d_acc, bin_l1);
            if (l1_base_addr != 0) {
                write_pages(c * pages_per_core, row_pages, l1base_acc, l1b_l1);
            }
        }

        // ── LPT (build_lpt): descending (cost, tile_id) sort of non-empty
        //    tiles, then first-min-load greedy assignment. cost = tile_pad. ──
        volatile uint32_t* lpt_cost = counts;  // reuse; counts already written out
        volatile uint32_t* lpt_tid = starts;
        uint32_t n_nonempty = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            if (counts[t] > 0) {  // note: reading counts[t] before overwrite at n<=t
                lpt_cost[n_nonempty] = tpad[t];
                lpt_tid[n_nonempty] = t;
                n_nonempty++;
            }
        }
        for (uint32_t i = 0; i + 1 < n_nonempty; i++) {
            uint32_t best = i;
            for (uint32_t j = i + 1; j < n_nonempty; j++) {
                if (lpt_cost[j] > lpt_cost[best] ||
                    (lpt_cost[j] == lpt_cost[best] && lpt_tid[j] > lpt_tid[best])) {
                    best = j;
                }
            }
            if (best != i) {
                const uint32_t tc = lpt_cost[i];
                lpt_cost[i] = lpt_cost[best];
                lpt_cost[best] = tc;
                const uint32_t tt = lpt_tid[i];
                lpt_tid[i] = lpt_tid[best];
                lpt_tid[best] = tt;
            }
        }

        // LPT core scratch after the 5 MAX_TILES arrays.
        uint32_t off = 5u * MAX_TILES;
        volatile uint32_t* core_cnt = sc + off; off += MAX_CORES;
        volatile uint64_t* core_load = reinterpret_cast<volatile uint64_t*>(sc + off);
        off += 2u * MAX_CORES;
        volatile uint32_t* core_off = sc + off; off += MAX_CORES;
        volatile uint32_t* core_write = sc + off;

        for (uint32_t c = 0; c < num_cores; c++) {
            core_cnt[c] = 0;
            core_load[c] = 0;
        }
        for (uint32_t i = 0; i < n_nonempty; i++) {
            const uint32_t cost = lpt_cost[i];
            uint32_t pick = 0;
            for (uint32_t c = 1; c < num_cores; c++) {
                if (core_load[c] < core_load[pick]) pick = c;
            }
            core_cnt[pick]++;
            core_load[pick] += cost;
        }
        uint32_t flat = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            core_off[c] = flat;
            flat += core_cnt[c];
            core_write[c] = 0;
            core_load[c] = 0;
        }
        // Emit flat tile_ids in [core][assignment-order]; bulk-write per filled page.
        // Build the whole flat array in outp (n_nonempty <= num_tiles <= MAX_TILES),
        // then bulk-write ceil_pages(flat) pages.
        for (uint32_t i = 0; i < n_nonempty; i++) {
            const uint32_t cost = lpt_cost[i];
            const uint32_t tid = lpt_tid[i];
            uint32_t pick = 0;
            for (uint32_t c = 1; c < num_cores; c++) {
                if (core_load[c] < core_load[pick]) pick = c;
            }
            outp[core_off[pick] + core_write[pick]] = tid;
            core_write[pick]++;
            core_load[pick] += cost;
        }
        if (n_nonempty > 0) {
            const uint32_t tids_pages = ceil_pages(n_nonempty);
            for (uint32_t k = n_nonempty; k < tids_pages * ELEMS_PER_PAGE; k++) outp[k] = 0;
            write_pages(0, tids_pages, tile_ids_acc, out_l1);
        }
        // lpt_meta[c] = {core_off, core_write}
        for (uint32_t c = 0; c < num_cores; c++) {
            outp[c * 2 + 0] = core_off[c];
            outp[c * 2 + 1] = core_write[c];
        }
        const uint32_t meta_pages = ceil_pages(num_cores * 2u);
        for (uint32_t k = num_cores * 2u; k < meta_pages * ELEMS_PER_PAGE; k++) outp[k] = 0;
        write_pages(0, meta_pages, lpt_meta_acc, out_l1);
    } else {
        // status != 0: still publish counts so the host read is well-defined.
        for (uint32_t t = 0; t < num_tiles; t++) outp[t] = counts[t];
        for (uint32_t k = num_tiles; k < row_span; k++) outp[k] = 0;
        write_pages(0, row_pages, counts_acc, out_l1);
    }

    ctrl[0] = P_kept;
    ctrl[1] = P_aligned;
    ctrl[2] = max_pad_n;
    ctrl[3] = status;
    ctrl[4] = total_pages;
    noc_async_write(ctrl_l1, get_noc_addr(0, ctrl_acc), PAGE_BYTES);
    noc_async_write_barrier();
}
