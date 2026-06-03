// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign SCAN phase 2 — per-core exclusive prefix-sum into offs[].
//
// Part of the on-device exclusive prefix-sum (GSPLAT_TT_TA_DEVICE_SCAN). Each
// core re-reads its contiguous tpg page range and writes the final exclusive
// prefix-sum into offs[], seeded with the per-core base offset the host
// computed from phase 1's partials (passed as a runtime arg). The masking
// (idx >= M -> add 0) makes offs[m] = P for every m in [M, ...] exactly like
// the host scan's `offs[m..] = P` tail fill — so offs[M] (read by K2 as the
// upper bound for the last Gaussian) lands on P. Integer addition only ->
// byte-identical to the host scan.
//
// RUNTIME ARGS
//   0: tpg_addr     int32 SoA tiles_per_gaussian (input)
//   1: offs_addr    int32 SoA exclusive prefix-sum (output)
//   2: page_start   first 16-elem page this core handles
//   3: page_count   number of pages this core handles
//   4: M            real Gaussian count (entries >= M are padding -> 0)
//   5: base_addr    per-core exclusive bases (from scan_bases on device)
//   6: core_slot    this core's linear index
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs (tpg, offs, base), DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
}  // namespace

void kernel_main() {
    const uint32_t tpg_addr   = get_arg_val<uint32_t>(0);
    const uint32_t offs_addr  = get_arg_val<uint32_t>(1);
    const uint32_t page_start = get_arg_val<uint32_t>(2);
    const uint32_t page_count = get_arg_val<uint32_t>(3);
    const uint32_t M          = get_arg_val<uint32_t>(4);
    const uint32_t base_addr  = get_arg_val<uint32_t>(5);
    const uint32_t core_slot  = get_arg_val<uint32_t>(6);

    if (page_count == 0) {
        return;
    }

    constexpr auto tpg_args  = TensorAccessorArgs<0>();
    constexpr auto offs_args = TensorAccessorArgs<tpg_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<offs_args.next_compile_time_args_offset()>();
    const auto tpg_acc  = TensorAccessor(tpg_args,  tpg_addr,  PAGE_BYTES);
    const auto offs_acc = TensorAccessor(offs_args, offs_addr, PAGE_BYTES);
    const auto base_acc = TensorAccessor(base_args, base_addr, PAGE_BYTES);

    constexpr uint32_t CB_TPG  = 0;
    constexpr uint32_t CB_OFFS = 1;
    constexpr uint32_t CB_BASE = 2;
    const uint32_t tpg_l1  = get_write_ptr(CB_TPG);
    const uint32_t offs_l1 = get_write_ptr(CB_OFFS);
    const uint32_t base_l1 = get_write_ptr(CB_BASE);
    auto tpgp  = reinterpret_cast<volatile int32_t*>(tpg_l1);
    auto offsp = reinterpret_cast<volatile int32_t*>(offs_l1);
    auto basep = reinterpret_cast<volatile uint32_t*>(base_l1);

    noc_async_read(get_noc_addr(core_slot, base_acc), base_l1, PAGE_BYTES);
    noc_async_read_barrier();
    uint32_t running = basep[0];
    for (uint32_t pg = 0; pg < page_count; pg++) {
        const uint32_t page = page_start + pg;
        const uint32_t g0 = page * ELEMS_PER_PAGE;
        if (g0 >= M) {
            // Whole page is padding (>= M): offs == running == P for all 16.
            // Skip the tpg read (would be out of K1's written range).
            for (uint32_t i = 0; i < ELEMS_PER_PAGE; i++)
                offsp[i] = static_cast<int32_t>(running);
        } else {
            noc_async_read(get_noc_addr(page, tpg_acc), tpg_l1, PAGE_BYTES);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < ELEMS_PER_PAGE; i++) {
                offsp[i] = static_cast<int32_t>(running);
                if (g0 + i < M) running += static_cast<uint32_t>(tpgp[i]);
            }
        }
        noc_async_write(offs_l1, get_noc_addr(page, offs_acc), PAGE_BYTES);
        noc_async_write_barrier();
    }
}
