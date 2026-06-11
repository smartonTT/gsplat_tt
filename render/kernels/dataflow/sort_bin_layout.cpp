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
// (NOT one element per page) so the single-core pass stays cheap.
//
// S5.2 perf (recover the iter-121 +23 ms single-core regression, bit-exact):
//   1. The full per-core histogram (num_cores × row_span u32 ≈ 450 KB on hero)
//      is cached in an L1 CB (CB_HCACHE) on the SINGLE count pass, so the base-
//      emit pass (Pass 2) reads it from L1 instead of re-streaming DRAM — one
//      DRAM read pass instead of two. (Falls back to re-reading DRAM if the
//      histogram does not fit the cache CB.)
//   2. LPT's descending (cost, tile_id) order is produced by an LSD radix sort
//      on the composite 32-bit key (cost<<16)|tid instead of the O(n²) selection
//      sort. The key is a strict total order on (cost, tile_id) (tile_ids are
//      unique, cost=tile_pad ≤ MAX_TILE_ENTRIES=32768 < 2^16, tid < 2^16) so the
//      radix order is BIT-IDENTICAL to the selection-sort order. The greedy
//      first-min-load assignment stays sequential/unchanged (its result depends
//      on prior loads — parallelizing it would risk a different tie-break).
//
// recbase/bin2d_rec is the retired tile_recs path and is left unwritten (the
// live L1_RECORD scatter consumes l1_rec_base, populated here).
//
// S5.4 (iter-125): Pass 2 (the per-(core,tile) bin2d + l1_rec_base base-emit, the
//   ~14k single-core DRAM writes that dominate the residual +13 ms) is no longer
//   emitted here. Instead this coordinator publishes the per-WORKER running prefix
//   (page_acc/rec_acc snapshot at each worker's FIRST source-core) into a small
//   checkpoint buffer (CB_OUT staging → ckpt DRAM). The companion emit kernel
//   (sort_bin_emit.cpp) then emits each worker's source-core rows in PARALLEL
//   across num_workers cores (disjoint DRAM regions, contention-free). The scan
//   over source-cores is identical to the old Pass-2 accumulation, so the emitted
//   bases are bit-for-bit identical to the single-core version.
//
// RUNTIME ARGS
//   0: bin2d_addr  1: tmeta_addr  2: tile_ids_addr  3: lpt_meta_addr
//   4: tile_counts_addr  5: ctrl_addr  6: num_cores  7: num_tiles  8: stride
//   9: recbase_addr (unused)  10: bucket_meta_addr (0=skip)  11: tile_ranges_addr
//   12: bucket_fit (l1_record per-tile slot count)  13: l1_base_addr (0=skip)
//   14: num_workers (emit core count for the checkpoint split)  15: ckpt_addr
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
    // args 12 (bucket_fit) and 13 (l1_base_addr) are now consumed by the parallel
    // emit kernel (sort_bin_emit.cpp), not here — intentionally not read.
    const uint32_t num_workers = get_arg_val<uint32_t>(14);
    const uint32_t ckpt_addr = get_arg_val<uint32_t>(15);

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
    constexpr auto ckpt_args =
        TensorAccessorArgs<l1base_args.next_compile_time_args_offset()>();

    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto tmeta_acc = TensorAccessor(tmeta_args, tmeta_addr, PAGE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, PAGE_BYTES);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, PAGE_BYTES);
    const auto counts_acc = TensorAccessor(counts_args, tile_counts_addr, PAGE_BYTES);
    const auto ctrl_acc = TensorAccessor(ctrl_args, ctrl_addr, PAGE_BYTES);
    const auto bucket_acc = TensorAccessor(bucket_args, bucket_meta_addr, PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, tile_ranges_addr, PAGE_BYTES);
    // l1base accessor is constructed in the emit kernel; only its arg-offset is
    // needed here so ckpt chains after it.
    const auto ckpt_acc = TensorAccessor(ckpt_args, ckpt_addr, PAGE_BYTES);

    constexpr uint32_t CB_CTRL = 0;     // 64B ctrl staging
    constexpr uint32_t CB_SCRATCH = 1;  // 5*MAX_TILES + 5*MAX_CORES u32
    constexpr uint32_t CB_ROW = 2;      // one core's hist row (MAX_TILES u32) — fallback only
    constexpr uint32_t CB_BIN = 3;      // one core's bin2d base row out (MAX_TILES u32)
    constexpr uint32_t CB_L1B = 4;      // one core's l1_rec_base row out (MAX_TILES u32)
    constexpr uint32_t CB_OUT = 5;      // per-tile staging (2*MAX_TILES u32)
    constexpr uint32_t CB_HCACHE = 6;   // full per-core histogram cache (HCACHE_CAP u32)

    // Capacity of the histogram cache CB; MUST match build_program_bin_layout's
    // cb(6, HCACHE_CAP*4) in render/host/sort_device.cpp. 128*1024 = 512 KB.
    constexpr uint32_t HCACHE_CAP = 128u * 1024u;

    const uint32_t ctrl_l1 = get_write_ptr(CB_CTRL);
    const uint32_t scratch_l1 = get_write_ptr(CB_SCRATCH);
    const uint32_t row_l1 = get_write_ptr(CB_ROW);
    // CB_BIN / CB_L1B are now consumed by the parallel emit kernel
    // (sort_bin_emit.cpp); the coordinator only stages checkpoints in CB_OUT.
    const uint32_t out_l1 = get_write_ptr(CB_OUT);
    const uint32_t hcache_l1 = get_write_ptr(CB_HCACHE);

    auto* ctrl = reinterpret_cast<volatile uint32_t*>(ctrl_l1);
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

    // Cache the whole per-core histogram in L1 when it fits CB_HCACHE: Pass 1
    // streams DRAM once into the cache, Pass 2 reads the cache (no 2nd DRAM read
    // pass). Each core's row occupies [c*row_span, (c+1)*row_span) in the cache.
    const bool cached =
        (static_cast<uint64_t>(num_cores) * row_span) <= HCACHE_CAP;

    for (uint32_t t = 0; t < num_tiles; t++) {
        counts[t] = 0;
        page_acc[t] = 0;  // accumulate tile_pad_pages here in pass 1
    }

    // ── Pass 1: stream each core row (caching it in L1 when it fits); per-tile
    //    count + padded-page sum, and the per-core padded total (BIN_LOCAL_MAX
    //    guard). ───────────────────────────────────────────────────────────
    uint32_t max_core_padded = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        const uint32_t rl1 = cached ? (hcache_l1 + c * row_span * 4u) : row_l1;
        read_pages(c * pages_per_core, row_pages, bin2d_acc, rl1);
        auto* rp = reinterpret_cast<volatile uint32_t*>(rl1);
        uint32_t core_padded = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = rp[t];
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

        // ── Pass 2 (PARALLEL via sort_bin_emit.cpp): publish per-worker prefix
        //    checkpoints instead of emitting the ~14k single-core base writes.
        //    For each emit worker w (covering a contiguous source-core range),
        //    snapshot the running page_acc (pages) + rec_acc (real counts) at its
        //    FIRST source-core, then advance the prefix over its cores. The advance
        //    loop is the SAME accumulation the old single-core Pass 2 did, so the
        //    workers — which replay it from these checkpoints — emit bit-identical
        //    bin2d/l1_rec_base bases. ckpt slot w = pages [w*2*row_pages ..) holds
        //    {page_acc row, rec_acc row}. ─────────────────────────────────────────
        const uint32_t W = num_workers;
        const uint32_t base_c = num_cores / W;
        const uint32_t rem_c = num_cores % W;
        uint32_t cc = 0;
        for (uint32_t w = 0; w < W; w++) {
            const uint32_t cnt = base_c + (w < rem_c ? 1u : 0u);
            // Snapshot page_acc (pages) → ckpt slot w, page row.
            for (uint32_t t = 0; t < num_tiles; t++) outp[t] = page_acc[t];
            for (uint32_t t = num_tiles; t < row_span; t++) outp[t] = 0;
            write_pages(w * 2u * row_pages, row_pages, ckpt_acc, out_l1);
            // Snapshot rec_acc (real counts) → ckpt slot w, rec row.
            for (uint32_t t = 0; t < num_tiles; t++) outp[t] = rec_acc[t];
            for (uint32_t t = num_tiles; t < row_span; t++) outp[t] = 0;
            write_pages(w * 2u * row_pages + row_pages, row_pages, ckpt_acc, out_l1);
            // Advance the running prefix over this worker's source-cores
            // (ceil_pages(0)==0 so the unconditional add matches the old `if h>0`).
            for (uint32_t k = 0; k < cnt; k++, cc++) {
                uint32_t rl1;
                if (cached) {
                    rl1 = hcache_l1 + cc * row_span * 4u;  // loaded in Pass 1
                } else {
                    rl1 = row_l1;
                    read_pages(cc * pages_per_core, row_pages, bin2d_acc, row_l1);
                }
                auto* rp = reinterpret_cast<volatile uint32_t*>(rl1);
                for (uint32_t t = 0; t < num_tiles; t++) {
                    const uint32_t h = rp[t];
                    rec_acc[t] += h;
                    page_acc[t] += ceil_pages(h);
                }
            }
        }

        // ── LPT (build_lpt): descending (cost, tile_id) order of non-empty
        //    tiles, then first-min-load greedy assignment. cost = tile_pad. ──
        // The descending (cost, tile_id) order is produced by an LSD radix sort
        // on the composite 32-bit key k=(cost<<16)|tid (cost=tile_pad ≤ 32768 <
        // 2^16, tid < num_tiles ≤ 2048 < 2^16). k is a strict total order on
        // (cost, tile_id), so the radix order equals the old selection-sort order
        // bit-for-bit. Build the keys in the now-free page_acc/rec_acc regions,
        // radix-sort ascending (4× 8-bit passes), then walk descending into
        // lpt_cost/lpt_tid.
        volatile uint32_t* lpt_cost = counts;  // reuse; counts already written out
        volatile uint32_t* lpt_tid = starts;
        volatile uint32_t* keyA = page_acc;  // free after Pass 2
        volatile uint32_t* keyB = rec_acc;   // free after Pass 2
        volatile uint32_t* radix_cnt = outp;  // out_l1 (>=256 u32); free until greedy
        uint32_t n_nonempty = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            if (counts[t] > 0) {
                keyA[n_nonempty] = (tpad[t] << 16) | t;
                n_nonempty++;
            }
        }
        {
            volatile uint32_t* src = keyA;
            volatile uint32_t* dst = keyB;
            for (uint32_t pass = 0; pass < 4u; pass++) {
                const uint32_t shift = pass * 8u;
                for (uint32_t d = 0; d < 256u; d++) radix_cnt[d] = 0;
                for (uint32_t i = 0; i < n_nonempty; i++)
                    radix_cnt[(src[i] >> shift) & 0xffu]++;
                uint32_t sum = 0;
                for (uint32_t d = 0; d < 256u; d++) {
                    const uint32_t c0 = radix_cnt[d];
                    radix_cnt[d] = sum;
                    sum += c0;
                }
                for (uint32_t i = 0; i < n_nonempty; i++) {
                    const uint32_t d = (src[i] >> shift) & 0xffu;
                    dst[radix_cnt[d]] = src[i];
                    radix_cnt[d]++;
                }
                volatile uint32_t* tmp = src;
                src = dst;
                dst = tmp;
            }
            // 4 swaps → src == keyA, sorted ascending. Walk descending.
            for (uint32_t i = 0; i < n_nonempty; i++) {
                const uint32_t key = src[n_nonempty - 1u - i];
                lpt_cost[i] = key >> 16;
                lpt_tid[i] = key & 0xffffu;
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
