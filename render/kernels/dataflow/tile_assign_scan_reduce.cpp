// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign SCAN phase 1 — per-core reduction of tiles_per_gaussian.
//
// Part of the on-device exclusive prefix-sum (GSPLAT_TT_TA_DEVICE_SCAN) that
// replaces the host D2H(tpg) + host scan + H2D(offs) round-trips. Each core
// owns the SAME contiguous page range it gets in K1's gaussian split. This
// kernel sums tpg[idx] over the core's range (masking idx >= M -> 0, matching
// the host scan which only sums [0,M)) and writes that single uint32 partial
// total into the core's dedicated page in core_total[]. The host then
// exclusive-scans the num_cores partials into per-core base offsets (P =
// sum of all partials) and hands each base back to phase 2 as a runtime arg.
//
// Integer addition only -> byte-identical to the host scan.
//
// RUNTIME ARGS
//   0: tpg_addr     int32 SoA tiles_per_gaussian (input)
//   1: total_addr   uint32 SoA per-core partial totals (output, 1 per page)
//   2: page_start   first 16-elem page this core handles
//   3: page_count   number of pages this core handles
//   4: M            real Gaussian count (entries >= M are padding -> 0)
//   5: core_slot    this core's linear index (page index in core_total[])
//
// COMPILE-TIME ARGS: 2 TensorAccessorArgs (tpg, total), DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
}  // namespace

void kernel_main() {
    const uint32_t tpg_addr   = get_arg_val<uint32_t>(0);
    const uint32_t total_addr = get_arg_val<uint32_t>(1);
    const uint32_t page_start = get_arg_val<uint32_t>(2);
    const uint32_t page_count = get_arg_val<uint32_t>(3);
    const uint32_t M          = get_arg_val<uint32_t>(4);
    const uint32_t core_slot  = get_arg_val<uint32_t>(5);

    constexpr auto tpg_args   = TensorAccessorArgs<0>();
    constexpr auto total_args = TensorAccessorArgs<tpg_args.next_compile_time_args_offset()>();
    const auto tpg_acc   = TensorAccessor(tpg_args,   tpg_addr,   PAGE_BYTES);
    const auto total_acc = TensorAccessor(total_args, total_addr, PAGE_BYTES);

    constexpr uint32_t CB_TPG = 0;
    constexpr uint32_t CB_TOT = 1;
    const uint32_t tpg_l1 = get_write_ptr(CB_TPG);
    const uint32_t tot_l1 = get_write_ptr(CB_TOT);
    auto tpgp = reinterpret_cast<volatile int32_t*>(tpg_l1);
    auto totp = reinterpret_cast<volatile uint32_t*>(tot_l1);

    uint32_t running = 0;
    for (uint32_t pg = 0; pg < page_count; pg++) {
        const uint32_t page = page_start + pg;
        const uint32_t g0 = page * ELEMS_PER_PAGE;
        if (g0 >= M) continue;  // whole page is padding (>= M): contributes 0.
        noc_async_read(get_noc_addr(page, tpg_acc), tpg_l1, PAGE_BYTES);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < ELEMS_PER_PAGE; i++) {
            if (g0 + i < M) running += static_cast<uint32_t>(tpgp[i]);
        }
    }

    // Every core (incl. ones with page_count==0) publishes its partial so the
    // host scan reads a defined value for every slot.
    totp[0] = running;
    noc_async_write(tot_l1, get_noc_addr(core_slot, total_acc), PAGE_BYTES);
    noc_async_write_barrier();
}
