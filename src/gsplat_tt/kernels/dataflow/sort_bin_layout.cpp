// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN layout — single-core post-count pass (GSPLAT_TT_SORT_DEVICE_LAYOUT).
//
// Replaces host D2H(bin2d) + page-aligned base layout + LPT + metadata H2D.
//
// RUNTIME ARGS
//   0: bin2d_addr  1: tmeta_addr  2: tile_ids_addr  3: lpt_meta_addr
//   4: tile_counts_addr  5: ctrl_addr  6: num_cores  7: num_tiles  8: stride
//   9: recbase_addr (0)  10: bucket_meta_addr (0)  11: tile_ranges_addr
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

template <typename AccT>
inline uint32_t read_u32_elem(uint32_t elem_idx, const AccT& acc, uint32_t l1) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(l1);
    const uint32_t pg = elem_idx / ELEMS_PER_PAGE;
    const uint32_t off = elem_idx % ELEMS_PER_PAGE;
    noc_async_read(get_noc_addr(pg, acc), l1, PAGE_BYTES);
    noc_async_read_barrier();
    return p[off];
}

template <typename AccT>
inline void write_u32_elem(uint32_t elem_idx, uint32_t val, const AccT& acc, uint32_t l1) {
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(l1);
    const uint32_t pg = elem_idx / ELEMS_PER_PAGE;
    const uint32_t off = elem_idx % ELEMS_PER_PAGE;
    noc_async_read(get_noc_addr(pg, acc), l1, PAGE_BYTES);
    noc_async_read_barrier();
    p[off] = val;
    noc_async_write(l1, get_noc_addr(pg, acc), PAGE_BYTES);
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
    const uint32_t recbase_addr = get_arg_val<uint32_t>(9);
    const uint32_t bucket_meta_addr = get_arg_val<uint32_t>(10);
    const uint32_t tile_ranges_addr = get_arg_val<uint32_t>(11);

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

    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto tmeta_acc = TensorAccessor(tmeta_args, tmeta_addr, PAGE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, PAGE_BYTES);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, PAGE_BYTES);
    const auto counts_acc = TensorAccessor(counts_args, tile_counts_addr, PAGE_BYTES);
    const auto ctrl_acc = TensorAccessor(ctrl_args, ctrl_addr, PAGE_BYTES);
    const auto recbase_acc = TensorAccessor(recbase_args, recbase_addr, PAGE_BYTES);
    const auto bucket_acc = TensorAccessor(bucket_args, bucket_meta_addr, PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, tile_ranges_addr, PAGE_BYTES);

    constexpr uint32_t CB_PAGE = 0;
    constexpr uint32_t CB_SCRATCH = 1;
    const uint32_t page_l1 = get_write_ptr(CB_PAGE);
    const uint32_t scratch_l1 = get_write_ptr(CB_SCRATCH);
    auto* page = reinterpret_cast<volatile uint32_t*>(page_l1);
    auto* counts_local = reinterpret_cast<volatile uint32_t*>(scratch_l1);
    auto* starts_local = counts_local + MAX_TILES;
    auto* tile_pad_local = starts_local + MAX_TILES;

    for (uint32_t t = 0; t < num_tiles; t++) {
        counts_local[t] = 0;
    }

    for (uint32_t c = 0; c < num_cores; c++) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t idx = c * stride + t;
            counts_local[t] += read_u32_elem(idx, bin2d_acc, page_l1);
        }
    }

    uint32_t max_core_padded = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t s = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t idx = c * stride + t;
            const uint32_t h = read_u32_elem(idx, bin2d_acc, page_l1);
            s += ceil_pages(h) * ELEMS_PER_PAGE;
        }
        if (s > max_core_padded) {
            max_core_padded = s;
        }
    }

    uint32_t status = 0;
    if (max_core_padded > BIN_LOCAL_MAX) {
        status = 1;
    }

    uint32_t cstart = 0;
    uint32_t apage = 0;
    uint32_t max_pad_n = 0;

    if (status == 0) {
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t creal = counts_local[t];
            starts_local[t] = cstart;
            cstart += creal;
            const uint32_t pstart_page_t = apage;
            const uint32_t tile_start_page = apage;
            uint32_t rec_run = 0;

            for (uint32_t c = 0; c < num_cores; c++) {
                const uint32_t idx = c * stride + t;
                const uint32_t h = read_u32_elem(idx, bin2d_acc, page_l1);
                if (recbase_addr != 0) {
                    write_u32_elem(idx, starts_local[t] + rec_run, recbase_acc, page_l1);
                    rec_run += h;
                }
                write_u32_elem(idx, apage * ELEMS_PER_PAGE, bin2d_acc, page_l1);
                if (h > 0) {
                    apage += ceil_pages(h);
                }
            }

            tile_pad_local[t] = (apage - tile_start_page) * ELEMS_PER_PAGE;
            if (tile_pad_local[t] > max_pad_n) {
                max_pad_n = tile_pad_local[t];
            }
            if (max_pad_n > MAX_TILE_ENTRIES) {
                status = 2;
                break;
            }

            if (tile_ranges_addr != 0) {
                write_u32_elem(t * 2 + 0, starts_local[t], ranges_acc, page_l1);
                write_u32_elem(
                    t * 2 + 1,
                    creal > 0 ? starts_local[t] + creal : starts_local[t],
                    ranges_acc,
                    page_l1);
            }
            if (bucket_meta_addr != 0 && creal > 0) {
                write_u32_elem(t * 2 + 0, starts_local[t], bucket_acc, page_l1);
                write_u32_elem(t * 2 + 1, creal, bucket_acc, page_l1);
            }

            write_u32_elem(t * 2 + 0, pstart_page_t, tmeta_acc, page_l1);
            write_u32_elem(t * 2 + 1, tile_pad_local[t], tmeta_acc, page_l1);
        }
    }

    const uint32_t P_kept = cstart;
    const uint32_t total_pages = apage > 0 ? apage : 1u;
    const uint32_t P_aligned = total_pages * ELEMS_PER_PAGE;

    for (uint32_t t = 0; t < num_tiles; t++) {
        write_u32_elem(t, counts_local[t], counts_acc, page_l1);
    }

    if (status == 0) {
        auto* lpt_cost = counts_local;
        auto* lpt_tid = starts_local;
        uint32_t n_nonempty = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            if (counts_local[t] > 0) {
                lpt_cost[n_nonempty] = tile_pad_local[t];
                lpt_tid[n_nonempty] = t;
                n_nonempty++;
            }
        }
        // Match host build_lpt: std::sort(cost_id, std::greater<>) on pair(cost, tile_id).
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

        // LPT scratch after the three MAX_TILES arrays (u32 offsets from scratch_l1).
        volatile uint32_t* scratch_u32 =
            reinterpret_cast<volatile uint32_t*>(scratch_l1);
        uint32_t lpt_off = 3 * MAX_TILES;
        volatile uint32_t* core_assign_cnt = scratch_u32 + lpt_off;
        lpt_off += MAX_CORES;
        volatile uint64_t* core_load =
            reinterpret_cast<volatile uint64_t*>(scratch_u32 + lpt_off);
        lpt_off += 2 * MAX_CORES;
        volatile uint32_t* core_off = scratch_u32 + lpt_off;
        lpt_off += MAX_CORES;
        volatile uint32_t* core_write = scratch_u32 + lpt_off;

        for (uint32_t c = 0; c < num_cores; c++) {
            core_assign_cnt[c] = 0;
            core_load[c] = 0;
        }
        for (uint32_t i = 0; i < n_nonempty; i++) {
            const uint32_t cost = lpt_cost[i];
            uint32_t pick = 0;
            for (uint32_t c = 1; c < num_cores; c++) {
                if (core_load[c] < core_load[pick]) {
                    pick = c;
                }
            }
            core_assign_cnt[pick]++;
            core_load[pick] += cost;
        }

        uint32_t flat = 0;
        for (uint32_t c = 0; c < num_cores; c++) {
            core_off[c] = flat;
            flat += core_assign_cnt[c];
            core_write[c] = 0;
            core_load[c] = 0;
        }

        for (uint32_t i = 0; i < n_nonempty; i++) {
            const uint32_t cost = lpt_cost[i];
            const uint32_t tid = lpt_tid[i];
            uint32_t pick = 0;
            for (uint32_t c = 1; c < num_cores; c++) {
                if (core_load[c] < core_load[pick]) {
                    pick = c;
                }
            }
            const uint32_t w = core_off[pick] + core_write[pick];
            write_u32_elem(w, tid, tile_ids_acc, page_l1);
            core_write[pick]++;
            core_load[pick] += cost;
        }

        for (uint32_t c = 0; c < num_cores; c++) {
            write_u32_elem(c * 2 + 0, core_off[c], lpt_meta_acc, page_l1);
            write_u32_elem(c * 2 + 1, core_write[c], lpt_meta_acc, page_l1);
        }
    }

    page[0] = P_kept;
    page[1] = P_aligned;
    page[2] = max_pad_n;
    page[3] = status;
    page[4] = total_pages;
    noc_async_write(page_l1, get_noc_addr(0, ctrl_acc), PAGE_BYTES);
    noc_async_write_barrier();
}
