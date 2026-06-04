// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc WRITER kernel (BRISC / NoC0) — amendment-002 tt-008c.
// Drains 8 fp32 output tile streams from the compute CBs and writes them to
// per-stream DRAM buffers. Replaces the 9-stream cov_cam-flavored writer; the
// host pfwc finisher now consumes cov2d + radii directly (no Jacobian on host).
//
//   CB  9 -> m2x       (mean_2d.x)
//   CB 10 -> m2y       (mean_2d.y)
//   CB 11 -> depth     (= tz)
//   CB 12 -> cov2d_a   (cov2d[0,0])
//   CB 13 -> cov2d_b   (cov2d[0,1])
//   CB 14 -> cov2d_c   (cov2d[1,1])
//   CB 15 -> radii_x   (ceil(k · sqrt(max(a, 0))))
//   CB 16 -> radii_y   (ceil(k · sqrt(max(c, 0))))
//
// RUNTIME ARGS
//   0..7 : DRAM addresses for the 8 output streams (same order as above)
//   8    : chunk_start
//   9    : num_chunks
//
// COMPILE-TIME: 8 TensorAccessorArgs.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t m2x_addr = get_arg_val<uint32_t>(0);
    const uint32_t m2y_addr = get_arg_val<uint32_t>(1);
    const uint32_t dep_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_addr   = get_arg_val<uint32_t>(3);
    const uint32_t b_addr   = get_arg_val<uint32_t>(4);
    const uint32_t c_addr   = get_arg_val<uint32_t>(5);
    const uint32_t rx_addr  = get_arg_val<uint32_t>(6);
    const uint32_t ry_addr  = get_arg_val<uint32_t>(7);
    const uint32_t chunk_start = get_arg_val<uint32_t>(8);
    const uint32_t num_chunks  = get_arg_val<uint32_t>(9);

    constexpr uint32_t CB_M2X = 9;
    constexpr uint32_t CB_M2Y = 10;
    constexpr uint32_t CB_DEP = 11;
    constexpr uint32_t CB_A   = 12;
    constexpr uint32_t CB_B   = 13;
    constexpr uint32_t CB_C   = 14;
    constexpr uint32_t CB_RX  = 15;
    constexpr uint32_t CB_RY  = 16;

    const uint32_t tile_bytes = get_tile_size(CB_M2X);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();

    const auto acc_m2x = TensorAccessor(a0, m2x_addr, tile_bytes);
    const auto acc_m2y = TensorAccessor(a1, m2y_addr, tile_bytes);
    const auto acc_dep = TensorAccessor(a2, dep_addr, tile_bytes);
    const auto acc_a   = TensorAccessor(a3, a_addr,   tile_bytes);
    const auto acc_b   = TensorAccessor(a4, b_addr,   tile_bytes);
    const auto acc_c   = TensorAccessor(a5, c_addr,   tile_bytes);
    const auto acc_rx  = TensorAccessor(a6, rx_addr,  tile_bytes);
    const auto acc_ry  = TensorAccessor(a7, ry_addr,  tile_bytes);

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
        cb_wait_front(CB_A, 1);
        noc_async_write_tile(tile_id, acc_a, get_read_ptr(CB_A));
        cb_wait_front(CB_B, 1);
        noc_async_write_tile(tile_id, acc_b, get_read_ptr(CB_B));
        cb_wait_front(CB_C, 1);
        noc_async_write_tile(tile_id, acc_c, get_read_ptr(CB_C));
        cb_wait_front(CB_RX, 1);
        noc_async_write_tile(tile_id, acc_rx, get_read_ptr(CB_RX));
        cb_wait_front(CB_RY, 1);
        noc_async_write_tile(tile_id, acc_ry, get_read_ptr(CB_RY));

        noc_async_write_barrier();

        cb_pop_front(CB_M2X, 1);
        cb_pop_front(CB_M2Y, 1);
        cb_pop_front(CB_DEP, 1);
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_B, 1);
        cb_pop_front(CB_C, 1);
        cb_pop_front(CB_RX, 1);
        cb_pop_front(CB_RY, 1);
    }
}
