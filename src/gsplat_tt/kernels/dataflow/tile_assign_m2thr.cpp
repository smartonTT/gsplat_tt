// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign K3 — per-Gaussian m2_thresh from resident opacity (device).
//
// Replaces the host bridge that computed m2_thresh from opacities[] and H2D'd
// the buffer. Reads proj_m_opacity (M-compact fp32 SoA, 64B pages) and writes
// buf_m2thr with the same bit pattern as the host loop in tile_assign_device.cpp:
//   op <= contrib_floor -> sentinel -1.0f
//   else m2t = -2*log(contrib_floor/op)
//
// RUNTIME ARGS
//   0: op_addr       proj_m_opacity DRAM base
//   1: m2thr_addr    output DRAM base
//   2: page_start    first 16-Gaussian page this core handles
//   3: page_count    pages this core handles
//   4: M             real Gaussian count
//   5: floor_bits    contrib_floor as uint32 bits
//
// COMPILE-TIME ARGS: 2 TensorAccessorArgs (op, m2thr).

#include <cstdint>
#include <cmath>

#include "api/dataflow/dataflow_api.h"

#pragma GCC optimize("no-fast-math", "fp-contract=off")

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline void f_to_bits(float f, volatile uint32_t* out) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    *out = b;
}

}  // namespace

void kernel_main() {
    const uint32_t op_addr     = get_arg_val<uint32_t>(0);
    const uint32_t m2thr_addr  = get_arg_val<uint32_t>(1);
    const uint32_t page_start  = get_arg_val<uint32_t>(2);
    const uint32_t page_count  = get_arg_val<uint32_t>(3);
    const uint32_t M           = get_arg_val<uint32_t>(4);
    const uint32_t floor_bits  = get_arg_val<uint32_t>(5);
    float contrib_floor;
    __builtin_memcpy(&contrib_floor, &floor_bits, 4);

    constexpr auto op_args     = TensorAccessorArgs<0>();
    constexpr auto m2thr_args  = TensorAccessorArgs<op_args.next_compile_time_args_offset()>();

    const auto op_acc    = TensorAccessor(op_args,    op_addr,    PAGE_BYTES);
    const auto m2thr_acc = TensorAccessor(m2thr_args, m2thr_addr, PAGE_BYTES);

    if (page_count == 0) {
        return;
    }

    constexpr uint32_t CB_OP  = 0;
    constexpr uint32_t CB_OUT = 1;

    const uint32_t op_l1  = get_write_ptr(CB_OP);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);
    auto* op_p  = reinterpret_cast<volatile uint32_t*>(op_l1);
    auto* out_p = reinterpret_cast<volatile uint32_t*>(out_l1);

    for (uint32_t pi = 0; pi < page_count; pi++) {
        const uint32_t pg = page_start + pi;
        noc_async_read(get_noc_addr(pg, op_acc), op_l1, PAGE_BYTES);
        noc_async_read_barrier();

        for (uint32_t ip = 0; ip < ELEMS_PER_PAGE; ip++) {
            const uint32_t m = pg * ELEMS_PER_PAGE + ip;
            float m2t = -1.0f;
            if (m < M) {
                const float op = bits_to_f(op_p[ip]);
                if (op > contrib_floor) {
                    m2t = -2.0f * std::log(contrib_floor / op);
                }
            }
            f_to_bits(m2t, &out_p[ip]);
        }

        noc_async_write(out_l1, get_noc_addr(pg, m2thr_acc), PAGE_BYTES);
        noc_async_write_barrier();
    }
}
