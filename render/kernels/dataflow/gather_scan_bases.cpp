// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gather_visible SCAN — on-device exclusive prefix-sum of the per-core visible
// counts (GSPLAT_TT_PROJ_DEVICE_SCAN). Replaces the host D2H(counts) + host
// exclusive-scan loop + host proj_M write that sit between the gather count and
// scatter passes (gather_visible_device.cpp). Runs on a SINGLE core.
//
// The count pass wrote each core's visible quota into the first uint32 of its
// 64B page in counts[]. This kernel:
//   * exclusive-prefix-sums those num_cores counts into per-core base offsets,
//     writing base[c] into the first uint32 of base[c]'s 64B page and a per-core
//     is_last flag (1 == the last non-empty core, which zero-pads the tail) into
//     the second uint32, and
//   * publishes M = sum(counts) into the first uint32 of proj_M page 0.
// The scatter pass then reads its (base, is_last) from base[core_id] over NoC
// instead of from host-computed runtime args, so the host never sees the counts
// and the count->scan->scatter chain stays resident on the in-order CQ (one
// fewer full-device Finish; the only host read is the 1-page M for buffer
// sizing). Integer addition only -> byte-identical to the host scan.
//
// RUNTIME ARGS (all uint32)
//   0: counts_addr   per-core visible counts (input; count at word[0] of page c)
//   1: base_addr     per-core exclusive bases (output; [0]=base, [1]=is_last)
//   2: m_addr        proj_M (output; M at word[0] of page 0)
//   3: num_cores     number of core slots in counts/base
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs (counts, base, m), DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t PAGE_BYTES = 64;
}  // namespace

void kernel_main() {
    const uint32_t counts_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_addr   = get_arg_val<uint32_t>(1);
    const uint32_t m_addr      = get_arg_val<uint32_t>(2);
    const uint32_t num_cores   = get_arg_val<uint32_t>(3);

    constexpr auto counts_args = TensorAccessorArgs<0>();
    constexpr auto base_args =
        TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto m_args =
        TensorAccessorArgs<base_args.next_compile_time_args_offset()>();

    const auto counts_acc = TensorAccessor(counts_args, counts_addr, PAGE_BYTES);
    const auto base_acc    = TensorAccessor(base_args, base_addr, PAGE_BYTES);
    const auto m_acc       = TensorAccessor(m_args, m_addr, PAGE_BYTES);

    constexpr uint32_t CB_IN = 0;
    constexpr uint32_t CB_OUT = 1;
    const uint32_t in_l1  = get_write_ptr(CB_IN);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);
    auto* inp  = reinterpret_cast<volatile uint32_t*>(in_l1);
    auto* outp = reinterpret_cast<volatile uint32_t*>(out_l1);

    // Pass 1: total M + the last non-empty core (it owns the tail zero-pad).
    uint32_t M = 0;
    uint32_t last_nz = 0;
    bool any = false;
    for (uint32_t c = 0; c < num_cores; c++) {
        noc_async_read(get_noc_addr(c, counts_acc), in_l1, PAGE_BYTES);
        noc_async_read_barrier();
        const uint32_t cnt = inp[0];
        if (cnt > 0) { last_nz = c; any = true; }
        M += cnt;
    }

    // Pass 2: exclusive prefix -> base[c], plus the is_last flag.
    uint32_t acc = 0;
    for (uint32_t c = 0; c < num_cores; c++) {
        noc_async_read(get_noc_addr(c, counts_acc), in_l1, PAGE_BYTES);
        noc_async_read_barrier();
        const uint32_t cnt = inp[0];
        outp[0] = acc;
        outp[1] = (any && c == last_nz) ? 1u : 0u;
        noc_async_write(out_l1, get_noc_addr(c, base_acc), PAGE_BYTES);
        noc_async_write_barrier();
        acc += cnt;
    }

    outp[0] = M;
    noc_async_write(out_l1, get_noc_addr(0, m_acc), PAGE_BYTES);
    noc_async_write_barrier();
}
