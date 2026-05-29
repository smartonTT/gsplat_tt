// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign K1 — per-Gaussian AABB / tiles_per_gaussian (Phase 1).
//
// Single data-movement kernel (one RISC core, scalar C++). Splits work OVER
// GAUSSIANS: each core handles a contiguous run of 16-Gaussian pages. For
// each Gaussian m it reproduces gsplat_cpu::tile_assign Phase 1 (BB rect in
// tile space) bit-exact and writes tiles_per_gaussian[m] = w*h. Gaussians with
// m >= M (page padding) write 0 so the host prefix-sum is unaffected.
//
// Buffers are SoA fp32/int32 with 64-byte pages (16 elements). Reading/writing
// whole 64B pages keeps the interleaved DRAM page stride aligned (a 48B layout
// previously caused a silent zero-row bug).
//
// RUNTIME ARGS
//   0: px_addr      DRAM base, fp32 SoA means_2d.x
//   1: py_addr      DRAM base, fp32 SoA means_2d.y
//   2: rx_addr      DRAM base, fp32 SoA radii.x
//   3: ry_addr      DRAM base, fp32 SoA radii.y
//   4: tpg_addr     DRAM base, int32 SoA tiles_per_gaussian (output)
//   5: page_start   first 16-Gaussian page this core handles
//   6: page_count   number of pages this core handles
//   7: M            real Gaussian count (entries >= M are padding)
//   8: tiles_x
//   9: tiles_y
//  10: tile_size
//
// COMPILE-TIME ARGS: 5 TensorAccessorArgs (px, py, rx, ry, tpg), DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;       // 16 elements
constexpr uint32_t ELEMS_PER_PAGE = 16;

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline int clampi(int v, int lo, int hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

}  // namespace

void kernel_main() {
    const uint32_t px_addr    = get_arg_val<uint32_t>(0);
    const uint32_t py_addr    = get_arg_val<uint32_t>(1);
    const uint32_t rx_addr    = get_arg_val<uint32_t>(2);
    const uint32_t ry_addr    = get_arg_val<uint32_t>(3);
    const uint32_t tpg_addr   = get_arg_val<uint32_t>(4);
    const uint32_t page_start = get_arg_val<uint32_t>(5);
    const uint32_t page_count = get_arg_val<uint32_t>(6);
    const uint32_t M          = get_arg_val<uint32_t>(7);
    const int tiles_x         = static_cast<int>(get_arg_val<uint32_t>(8));
    const int tiles_y         = static_cast<int>(get_arg_val<uint32_t>(9));
    const float tsf           = static_cast<float>(get_arg_val<uint32_t>(10));

    constexpr auto px_args  = TensorAccessorArgs<0>();
    constexpr auto py_args  = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto rx_args  = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto ry_args  = TensorAccessorArgs<rx_args.next_compile_time_args_offset()>();
    constexpr auto tpg_args = TensorAccessorArgs<ry_args.next_compile_time_args_offset()>();

    const auto px_acc  = TensorAccessor(px_args,  px_addr,  PAGE_BYTES);
    const auto py_acc  = TensorAccessor(py_args,  py_addr,  PAGE_BYTES);
    const auto rx_acc  = TensorAccessor(rx_args,  rx_addr,  PAGE_BYTES);
    const auto ry_acc  = TensorAccessor(ry_args,  ry_addr,  PAGE_BYTES);
    const auto tpg_acc = TensorAccessor(tpg_args, tpg_addr, PAGE_BYTES);

    if (page_count == 0) {
        return;
    }

    // Scratch CBs (declared in tile_assign_device.cpp): CB 0..4 each reserve
    // one 64B page in L1. get_write_ptr returns the L1 address we DMA in/out
    // of (no cb_reserve/push — these are fixed scratch regions).
    constexpr uint32_t CB_PX  = 0;
    constexpr uint32_t CB_PY  = 1;
    constexpr uint32_t CB_RX  = 2;
    constexpr uint32_t CB_RY  = 3;
    constexpr uint32_t CB_OUT = 4;

    const uint32_t px_l1  = get_write_ptr(CB_PX);
    const uint32_t py_l1  = get_write_ptr(CB_PY);
    const uint32_t rx_l1  = get_write_ptr(CB_RX);
    const uint32_t ry_l1  = get_write_ptr(CB_RY);
    const uint32_t out_l1 = get_write_ptr(CB_OUT);

    auto pxp = reinterpret_cast<volatile uint32_t*>(px_l1);
    auto pyp = reinterpret_cast<volatile uint32_t*>(py_l1);
    auto rxp = reinterpret_cast<volatile uint32_t*>(rx_l1);
    auto ryp = reinterpret_cast<volatile uint32_t*>(ry_l1);
    auto outp = reinterpret_cast<volatile int32_t*>(out_l1);

    for (uint32_t pg = 0; pg < page_count; pg++) {
        const uint32_t page = page_start + pg;
        const uint32_t g0 = page * ELEMS_PER_PAGE;

        noc_async_read(get_noc_addr(page, px_acc), px_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(page, py_acc), py_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(page, rx_acc), rx_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(page, ry_acc), ry_l1, PAGE_BYTES);
        noc_async_read_barrier();

        for (uint32_t i = 0; i < ELEMS_PER_PAGE; i++) {
            const uint32_t g = g0 + i;
            if (g >= M) {
                outp[i] = 0;
                continue;
            }
            const float px = bits_to_f(pxp[i]);
            const float py = bits_to_f(pyp[i]);
            const float rx = bits_to_f(rxp[i]);
            const float ry = bits_to_f(ryp[i]);

            const int min_x = clampi(static_cast<int>((px - rx) / tsf), 0, tiles_x - 1);
            const int max_x = clampi(static_cast<int>((px + rx) / tsf), 0, tiles_x - 1);
            const int min_y = clampi(static_cast<int>((py - ry) / tsf), 0, tiles_y - 1);
            const int max_y = clampi(static_cast<int>((py + ry) / tsf), 0, tiles_y - 1);
            const int w = max_x - min_x + 1;
            const int h = max_y - min_y + 1;
            outp[i] = w * h;
        }

        noc_async_write(out_l1, get_noc_addr(page, tpg_acc), PAGE_BYTES);
        noc_async_write_barrier();
    }
}
