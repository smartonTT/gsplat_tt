// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN emit — PARALLEL companion to sort_bin_layout.cpp (S5.4, iter-125).
//
// The single-core layout kernel's residual cost (~+13 ms vs the iter-116 195.5
// baseline) was dominated by Pass 2: the per-(core,tile) bin2d base + l1_rec_base
// DRAM writes (≈ num_source_cores × ceil_pages(num_tiles) × 2 ≈ 14k page writes)
// done serially on ONE Tensix. This kernel distributes Pass 2 across `num_workers`
// cores: each worker owns a contiguous range of SOURCE-cores and emits only those
// rows. The DRAM regions are disjoint per source-core, so the parallel writes are
// contention-free and need no locks.
//
// Each worker seeds its running per-tile prefix from the checkpoint the
// coordinator published for it (page_acc in PAGES + rec_acc in real counts, at the
// worker's FIRST source-core), then replays the EXACT Pass-2 accumulation over its
// source-cores. Because the checkpoint is the precise boundary value and the inner
// loop is identical to the old single-core Pass 2, the emitted bases are
// bit-for-bit identical.
//
// RUNTIME ARGS
//   0: bin2d_addr  1: l1_base_addr (0=skip l1_rec_base)  2: ckpt_addr  3: ctrl_addr
//   4: num_tiles   5: stride      6: bucket_fit
//   7: core_start  8: core_count (0 => inactive worker, no-op)  9: worker_idx

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

inline uint32_t ceil_pages(uint32_t h) { return (h + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE; }

template <typename AccT>
inline void read_pages(uint32_t base_pg, uint32_t row_pages, const AccT& acc, uint32_t l1) {
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_pg + pp, acc), l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
}

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
    const uint32_t l1_base_addr = get_arg_val<uint32_t>(1);
    const uint32_t ckpt_addr = get_arg_val<uint32_t>(2);
    const uint32_t ctrl_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_tiles = get_arg_val<uint32_t>(4);
    const uint32_t stride = get_arg_val<uint32_t>(5);
    const uint32_t bucket_fit = get_arg_val<uint32_t>(6);
    const uint32_t core_start = get_arg_val<uint32_t>(7);
    const uint32_t core_count = get_arg_val<uint32_t>(8);
    const uint32_t worker_idx = get_arg_val<uint32_t>(9);

    if (core_count == 0) return;  // grid core not assigned an emit range

    constexpr auto bin2d_args = TensorAccessorArgs<0>();
    constexpr auto l1base_args =
        TensorAccessorArgs<bin2d_args.next_compile_time_args_offset()>();
    constexpr auto ckpt_args =
        TensorAccessorArgs<l1base_args.next_compile_time_args_offset()>();
    constexpr auto ctrl_args =
        TensorAccessorArgs<ckpt_args.next_compile_time_args_offset()>();

    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto l1base_acc = TensorAccessor(l1base_args, l1_base_addr, PAGE_BYTES);
    const auto ckpt_acc = TensorAccessor(ckpt_args, ckpt_addr, PAGE_BYTES);
    const auto ctrl_acc = TensorAccessor(ctrl_args, ctrl_addr, PAGE_BYTES);

    constexpr uint32_t CB_CTRL = 0;  // 64B status read
    constexpr uint32_t CB_PAGE = 1;  // page_acc row (pages)
    constexpr uint32_t CB_REC = 2;   // rec_acc row (real counts)
    constexpr uint32_t CB_HIST = 3;  // one source-core hist row in
    constexpr uint32_t CB_BIN = 4;   // bin2d base row out
    constexpr uint32_t CB_L1B = 5;   // l1_rec_base row out

    const uint32_t ctrl_l1 = get_write_ptr(CB_CTRL);
    const uint32_t page_l1 = get_write_ptr(CB_PAGE);
    const uint32_t rec_l1 = get_write_ptr(CB_REC);
    const uint32_t hist_l1 = get_write_ptr(CB_HIST);
    const uint32_t bin_l1 = get_write_ptr(CB_BIN);
    const uint32_t l1b_l1 = get_write_ptr(CB_L1B);

    // Coordinator hard-fail (status != 0) => skip emit; the host hard-fails the
    // frame in that branch, so the bin2d contents are irrelevant — just don't
    // touch DRAM with stale/garbage checkpoints.
    noc_async_read(get_noc_addr(0, ctrl_acc), ctrl_l1, PAGE_BYTES);
    noc_async_read_barrier();
    if (reinterpret_cast<volatile uint32_t*>(ctrl_l1)[3] != 0) return;

    const uint32_t pages_per_core = stride / ELEMS_PER_PAGE;
    const uint32_t row_pages = ceil_pages(num_tiles);
    const uint32_t row_span = row_pages * ELEMS_PER_PAGE;

    auto* page_acc = reinterpret_cast<volatile uint32_t*>(page_l1);
    auto* rec_acc = reinterpret_cast<volatile uint32_t*>(rec_l1);
    auto* binp = reinterpret_cast<volatile uint32_t*>(bin_l1);
    auto* l1bp = reinterpret_cast<volatile uint32_t*>(l1b_l1);

    // Seed the running prefix from this worker's checkpoint (slot worker_idx):
    //   pages [w*2*row_pages ..)            = page_acc row
    //   pages [w*2*row_pages + row_pages ..) = rec_acc row
    read_pages(worker_idx * 2u * row_pages, row_pages, ckpt_acc, page_l1);
    read_pages(worker_idx * 2u * row_pages + row_pages, row_pages, ckpt_acc, rec_l1);

    const uint32_t c_end = core_start + core_count;
    for (uint32_t c = core_start; c < c_end; c++) {
        read_pages(c * pages_per_core, row_pages, bin2d_acc, hist_l1);
        auto* rp = reinterpret_cast<volatile uint32_t*>(hist_l1);
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = rp[t];
            binp[t] = page_acc[t] * ELEMS_PER_PAGE;
            l1bp[t] = t * bucket_fit + rec_acc[t];
            rec_acc[t] += h;
            page_acc[t] += ceil_pages(h);
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
}
