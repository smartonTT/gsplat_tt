// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc WRITER kernel (BRISC / NoC0).
// Drains 9 fp32 output tile streams from the compute CBs and writes them
// to per-stream DRAM buffers.
//
//   CB  9 -> m2x   (mean_x_2d)
//   CB 10 -> m2y
//   CB 11 -> depth
//   CB 12 -> cc00
//   CB 13 -> cc01
//   CB 14 -> cc02
//   CB 15 -> cc11
//   CB 16 -> cc12
//   CB 17 -> cc22
//
// RUNTIME ARGS
//   0..8 : DRAM addresses for the 9 output streams (same order as above)
//   9    : chunk_start
//   10   : num_chunks
//
// COMPILE-TIME: 9 TensorAccessorArgs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t m2x_addr  = get_arg_val<uint32_t>(0);
    const uint32_t m2y_addr  = get_arg_val<uint32_t>(1);
    const uint32_t dep_addr  = get_arg_val<uint32_t>(2);
    const uint32_t cc00_addr = get_arg_val<uint32_t>(3);
    const uint32_t cc01_addr = get_arg_val<uint32_t>(4);
    const uint32_t cc02_addr = get_arg_val<uint32_t>(5);
    const uint32_t cc11_addr = get_arg_val<uint32_t>(6);
    const uint32_t cc12_addr = get_arg_val<uint32_t>(7);
    const uint32_t cc22_addr = get_arg_val<uint32_t>(8);
    const uint32_t chunk_start = get_arg_val<uint32_t>(9);
    const uint32_t num_chunks  = get_arg_val<uint32_t>(10);

    constexpr uint32_t CB_M2X  = 9;
    constexpr uint32_t CB_M2Y  = 10;
    constexpr uint32_t CB_DEP  = 11;
    constexpr uint32_t CB_CC00 = 12;
    constexpr uint32_t CB_CC01 = 13;
    constexpr uint32_t CB_CC02 = 14;
    constexpr uint32_t CB_CC11 = 15;
    constexpr uint32_t CB_CC12 = 16;
    constexpr uint32_t CB_CC22 = 17;

    const uint32_t tile_bytes = get_tile_size(CB_M2X);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto a8 = TensorAccessorArgs<a7.next_compile_time_args_offset()>();

    const auto acc_m2x  = TensorAccessor(a0, m2x_addr,  tile_bytes);
    const auto acc_m2y  = TensorAccessor(a1, m2y_addr,  tile_bytes);
    const auto acc_dep  = TensorAccessor(a2, dep_addr,  tile_bytes);
    const auto acc_cc00 = TensorAccessor(a3, cc00_addr, tile_bytes);
    const auto acc_cc01 = TensorAccessor(a4, cc01_addr, tile_bytes);
    const auto acc_cc02 = TensorAccessor(a5, cc02_addr, tile_bytes);
    const auto acc_cc11 = TensorAccessor(a6, cc11_addr, tile_bytes);
    const auto acc_cc12 = TensorAccessor(a7, cc12_addr, tile_bytes);
    const auto acc_cc22 = TensorAccessor(a8, cc22_addr, tile_bytes);

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t k = 0; k < num_chunks; k++) {
        const uint32_t tile_id = chunk_start + k;

        cb_wait_front(CB_M2X, 1);
        noc_async_write_tile(tile_id, acc_m2x, get_read_ptr(CB_M2X));
        cb_wait_front(CB_M2Y, 1);
        noc_async_write_tile(tile_id, acc_m2y, get_read_ptr(CB_M2Y));
        cb_wait_front(CB_DEP, 1);
        noc_async_write_tile(tile_id, acc_dep, get_read_ptr(CB_DEP));
        cb_wait_front(CB_CC00, 1);
        noc_async_write_tile(tile_id, acc_cc00, get_read_ptr(CB_CC00));
        cb_wait_front(CB_CC01, 1);
        noc_async_write_tile(tile_id, acc_cc01, get_read_ptr(CB_CC01));
        cb_wait_front(CB_CC02, 1);
        noc_async_write_tile(tile_id, acc_cc02, get_read_ptr(CB_CC02));
        cb_wait_front(CB_CC11, 1);
        noc_async_write_tile(tile_id, acc_cc11, get_read_ptr(CB_CC11));
        cb_wait_front(CB_CC12, 1);
        noc_async_write_tile(tile_id, acc_cc12, get_read_ptr(CB_CC12));
        cb_wait_front(CB_CC22, 1);
        noc_async_write_tile(tile_id, acc_cc22, get_read_ptr(CB_CC22));

        noc_async_write_barrier();

        cb_pop_front(CB_M2X, 1);
        cb_pop_front(CB_M2Y, 1);
        cb_pop_front(CB_DEP, 1);
        cb_pop_front(CB_CC00, 1);
        cb_pop_front(CB_CC01, 1);
        cb_pop_front(CB_CC02, 1);
        cb_pop_front(CB_CC11, 1);
        cb_pop_front(CB_CC12, 1);
        cb_pop_front(CB_CC22, 1);
    }
}
