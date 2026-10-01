// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gather_visible SCAN for lever 2 (task #99, GSPLAT_TT_SFPU_VIS). Single core.
// Replaces proj_count + gather_scan_bases: the pfwc writer already counted the
// visible gaussians and their (gaussian, tile) pairs per 1024-gaussian tile.
// This kernel reads those counts and, with vis_tile::scan_slots,
//   * cuts the gather's compaction sequence (strided tiles, core-major) into
//     num_cores * movers contiguous slot ranges of about equal scatter cost
//     (visible + tile_weight per non-empty tile + empty_weight per tile), or
//     the legacy per-core halves when balance == 0; the compact order is the
//     sequence order either way, so outputs do not depend on the cut;
//   * writes each slot's [base, is_last, q0, qn, pair base] page;
//   * publishes proj_M ([0] = M, [1] = P, the host's one post-chain read),
//     ta_pairs_P ([0] = P (clamped to p_max if set), [1] = its 16-padded value,
//     [2] = overflow, [3] = P) as tile_assign_scan_bases.cpp did, and fills the
//     offs page holding offs[M] with P (the scatter then writes offs[0, M)).
//
// RUNTIME ARGS
//   0: per-tile counts base (1 KB pages)   1: slot pages base (64 B)
//   2: proj_M base   3: ta_pairs_P base   4: offs base (64 B pages)
//   5: num_tiles   6: num_cores   7: movers (1 or 2)   8: balance
//   9: tile_weight   10: empty_weight   11: split permille (balance == 0)
//   12: p_max (0 = no clamp)
//
// COMPILE-TIME ARGS: 5 TensorAccessorArgs (counts, slots, proj_M, ta_pairs_P, offs).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "vis_tile.h"

void kernel_main() {
    DeviceZoneScopedN("proj_vis_scan");
    const uint32_t counts_addr = get_arg_val<uint32_t>(0);
    const uint32_t slots_addr = get_arg_val<uint32_t>(1);
    const uint32_t m_addr = get_arg_val<uint32_t>(2);
    const uint32_t pp_addr = get_arg_val<uint32_t>(3);
    const uint32_t offs_addr = get_arg_val<uint32_t>(4);
    vis_tile::ScanArgs a;
    a.num_tiles = get_arg_val<uint32_t>(5);
    a.num_cores = get_arg_val<uint32_t>(6);
    a.movers = get_arg_val<uint32_t>(7);
    a.balance = get_arg_val<uint32_t>(8);
    a.tile_weight = get_arg_val<uint32_t>(9);
    a.empty_weight = get_arg_val<uint32_t>(10);
    a.split_permille = get_arg_val<uint32_t>(11);
    const uint32_t p_max = get_arg_val<uint32_t>(12);

    constexpr uint32_t PAGE = 64;
    constexpr uint32_t CPAGE = vis_tile::COUNTS_PAGE_BYTES;
    constexpr auto c_args = TensorAccessorArgs<0>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto acc_c = TensorAccessor(c_args, counts_addr, CPAGE);
    const auto acc_s = TensorAccessor(s_args, slots_addr, PAGE);
    const auto acc_m = TensorAccessor(m_args, m_addr, PAGE);
    const auto acc_p = TensorAccessor(p_args, pp_addr, PAGE);
    const auto acc_o = TensorAccessor(o_args, offs_addr, PAGE);

    constexpr uint32_t CB_COUNTS = 0, CB_SLOTS = 1, CB_MISC = 2;
    const uint32_t l1_c = get_write_ptr(CB_COUNTS);
    const uint32_t l1_s = get_write_ptr(CB_SLOTS);
    const uint32_t l1_x = get_write_ptr(CB_MISC);  // 3 pages: proj_M, ta_pairs_P, offs

    const uint32_t n_cpages = (a.num_tiles + vis_tile::COUNTS_TILES_PER_PAGE - 1) /
                              vis_tile::COUNTS_TILES_PER_PAGE;
    for (uint32_t p = 0; p < n_cpages; p++)
        noc_async_read(get_noc_addr(p, acc_c), l1_c + p * CPAGE, CPAGE);
    noc_async_read_barrier();

    uint32_t M = 0, P = 0;
    auto slots = reinterpret_cast<volatile uint32_t*>(l1_s);
    vis_tile::scan_slots(reinterpret_cast<volatile uint32_t*>(l1_c), a, slots, &M, &P);

    const uint32_t S = a.num_cores * a.movers;
    for (uint32_t s = 0; s < S; s++)
        noc_async_write(l1_s + s * PAGE, get_noc_addr(s, acc_s), PAGE);

    auto mp = reinterpret_cast<volatile uint32_t*>(l1_x);
    auto pp = reinterpret_cast<volatile uint32_t*>(l1_x + PAGE);
    auto op = reinterpret_cast<volatile uint32_t*>(l1_x + 2 * PAGE);
    for (uint32_t w = 0; w < 16; w++) {
        mp[w] = 0;
        pp[w] = 0;
        op[w] = P;
    }
    mp[0] = M;
    mp[1] = P;
    const uint32_t overflow = (p_max != 0 && P > p_max) ? 1u : 0u;
    const uint32_t p_pub = overflow ? p_max : P;
    pp[0] = p_pub;
    pp[1] = ((p_pub + 15u) / 16u) * 16u;
    pp[2] = overflow;
    pp[3] = P;
    noc_async_write(l1_x, get_noc_addr(0, acc_m), PAGE);
    noc_async_write(l1_x + PAGE, get_noc_addr(0, acc_p), PAGE);
    noc_async_write(l1_x + 2 * PAGE, get_noc_addr(M / 16u, acc_o), PAGE);
    noc_async_write_barrier();
}
