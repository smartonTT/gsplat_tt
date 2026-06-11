// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign SCAN phase 1.5 — exclusive prefix of per-core partial totals.
//
// Runs on a single core after scan_reduce. Reads each core's partial from
// core_total[], writes exclusive bases to core_base[] (one uint32 per page),
// and publishes P + P_pad into ta_pairs_P for sort / resident-pairs handoff.
// Replaces the host D2H(core_total) + host exclusive-scan loop.
//
// RUNTIME ARGS
//   0: total_addr    per-core partial totals (input, from scan_reduce)
//   1: base_addr     per-core exclusive bases (output)
//   2: pairs_p_addr  64B page: [0]=P_pub, [1]=P_pad, [2]=overflow, [3]=P_true
//   3: num_cores     number of core slots in total/base
//   4: p_max         S5.3 host-free static pair ceiling (0 => no clamp). When
//                    nonzero the PUBLISHED P (and P_pad) are CLAMPED to p_max so
//                    every downstream consumer (K2 scatter, sort binning) reads
//                    a count that can never index past the statically-sized
//                    (p_max) pair buffers — overflow can never corrupt memory.
//                    The TRUE (unclamped) P is published at [3] and an overflow
//                    flag at [2] so the host can hard-fail post-frame.
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs (total, base, pairs_p).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

inline uint32_t round_up(uint32_t v, uint32_t m) {
    return ((v + m - 1) / m) * m;
}
}  // namespace

void kernel_main() {
    const uint32_t total_addr   = get_arg_val<uint32_t>(0);
    const uint32_t base_addr    = get_arg_val<uint32_t>(1);
    const uint32_t pairs_p_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_cores    = get_arg_val<uint32_t>(3);
    const uint32_t p_max        = get_arg_val<uint32_t>(4);  // 0 => no clamp

    constexpr auto total_args = TensorAccessorArgs<0>();
    constexpr auto base_args =
        TensorAccessorArgs<total_args.next_compile_time_args_offset()>();
    constexpr auto pairs_args =
        TensorAccessorArgs<base_args.next_compile_time_args_offset()>();

    const auto total_acc = TensorAccessor(total_args, total_addr, PAGE_BYTES);
    const auto base_acc  = TensorAccessor(base_args, base_addr, PAGE_BYTES);
    const auto pairs_acc = TensorAccessor(pairs_args, pairs_p_addr, PAGE_BYTES);

    constexpr uint32_t CB_IN  = 0;
    constexpr uint32_t CB_OUT = 1;
    const uint32_t in_l1  = get_write_ptr(CB_IN);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);
    auto* inp  = reinterpret_cast<volatile uint32_t*>(in_l1);
    auto* outp = reinterpret_cast<volatile uint32_t*>(out_l1);

    uint32_t acc = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        noc_async_read(get_noc_addr(c, total_acc), in_l1, PAGE_BYTES);
        noc_async_read_barrier();
        outp[0] = acc;
        noc_async_write(out_l1, get_noc_addr(c, base_acc), PAGE_BYTES);
        noc_async_write_barrier();
        acc += inp[0];
    }

    const uint32_t P_true = acc;
    // S5.3 host-free: clamp the PUBLISHED count to the static ceiling so K2 /
    // sort can never index past the p_max-sized pair buffers (no OOB / no memory
    // corruption). overflow=1 surfaces a too-small ceiling for a post-frame host
    // hard-fail; the unclamped count is preserved at [3] for stats/diagnostics.
    const uint32_t overflow = (p_max != 0 && P_true > p_max) ? 1u : 0u;
    const uint32_t P_pub = (p_max != 0 && P_true > p_max) ? p_max : P_true;
    const uint32_t P_pad = round_up(P_pub, ELEMS_PER_PAGE);
    outp[0] = P_pub;
    outp[1] = P_pad;
    outp[2] = overflow;
    outp[3] = P_true;
    noc_async_write(out_l1, get_noc_addr(0, pairs_acc), PAGE_BYTES);
    noc_async_write_barrier();
}
