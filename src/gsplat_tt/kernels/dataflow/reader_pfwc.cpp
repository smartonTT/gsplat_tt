// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc READER kernel (NCRISC / NoC1).
// Streams 9 fp32 SoA tile streams into the kernel's input CBs:
//   mcx, mcy, mcz, c00, c01, c02, c11, c12, c22.
//
// mcx/mcy/mcz live in DRAM buffers that gsplat_tt::device_state holds across
// frames (means_cam_x/y/z published by the project kernel, tt-005b). The
// cov3d 6-stream is uploaded once per scene and pinned (host-side cache).
//
// RUNTIME ARGS
//   0..8 : DRAM base addresses (mcx, mcy, mcz, c00, c01, c02, c11, c12, c22)
//   9    : chunk_start (this core's first tile index)
//   10   : num_chunks
//
// COMPILE-TIME ARGS: 9 TensorAccessorArgs, in the same order as runtime args 0..8.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t mcx_addr  = get_arg_val<uint32_t>(0);
    const uint32_t mcy_addr  = get_arg_val<uint32_t>(1);
    const uint32_t mcz_addr  = get_arg_val<uint32_t>(2);
    const uint32_t c00_addr  = get_arg_val<uint32_t>(3);
    const uint32_t c01_addr  = get_arg_val<uint32_t>(4);
    const uint32_t c02_addr  = get_arg_val<uint32_t>(5);
    const uint32_t c11_addr  = get_arg_val<uint32_t>(6);
    const uint32_t c12_addr  = get_arg_val<uint32_t>(7);
    const uint32_t c22_addr  = get_arg_val<uint32_t>(8);
    const uint32_t chunk_start = get_arg_val<uint32_t>(9);
    const uint32_t num_chunks  = get_arg_val<uint32_t>(10);

    constexpr uint32_t CB_MCX = 0;
    constexpr uint32_t CB_MCY = 1;
    constexpr uint32_t CB_MCZ = 2;
    constexpr uint32_t CB_C00 = 3;
    constexpr uint32_t CB_C01 = 4;
    constexpr uint32_t CB_C02 = 5;
    constexpr uint32_t CB_C11 = 6;
    constexpr uint32_t CB_C12 = 7;
    constexpr uint32_t CB_C22 = 8;

    const uint32_t tile_bytes = get_tile_size(CB_MCX);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto a8 = TensorAccessorArgs<a7.next_compile_time_args_offset()>();

    const auto acc_mcx = TensorAccessor(a0, mcx_addr, tile_bytes);
    const auto acc_mcy = TensorAccessor(a1, mcy_addr, tile_bytes);
    const auto acc_mcz = TensorAccessor(a2, mcz_addr, tile_bytes);
    const auto acc_c00 = TensorAccessor(a3, c00_addr, tile_bytes);
    const auto acc_c01 = TensorAccessor(a4, c01_addr, tile_bytes);
    const auto acc_c02 = TensorAccessor(a5, c02_addr, tile_bytes);
    const auto acc_c11 = TensorAccessor(a6, c11_addr, tile_bytes);
    const auto acc_c12 = TensorAccessor(a7, c12_addr, tile_bytes);
    const auto acc_c22 = TensorAccessor(a8, c22_addr, tile_bytes);

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t k = 0; k < num_chunks; k++) {
        const uint32_t tile_id = chunk_start + k;

        cb_reserve_back(CB_MCX, 1);
        noc_async_read_tile(tile_id, acc_mcx, get_write_ptr(CB_MCX));
        cb_reserve_back(CB_MCY, 1);
        noc_async_read_tile(tile_id, acc_mcy, get_write_ptr(CB_MCY));
        cb_reserve_back(CB_MCZ, 1);
        noc_async_read_tile(tile_id, acc_mcz, get_write_ptr(CB_MCZ));
        cb_reserve_back(CB_C00, 1);
        noc_async_read_tile(tile_id, acc_c00, get_write_ptr(CB_C00));
        cb_reserve_back(CB_C01, 1);
        noc_async_read_tile(tile_id, acc_c01, get_write_ptr(CB_C01));
        cb_reserve_back(CB_C02, 1);
        noc_async_read_tile(tile_id, acc_c02, get_write_ptr(CB_C02));
        cb_reserve_back(CB_C11, 1);
        noc_async_read_tile(tile_id, acc_c11, get_write_ptr(CB_C11));
        cb_reserve_back(CB_C12, 1);
        noc_async_read_tile(tile_id, acc_c12, get_write_ptr(CB_C12));
        cb_reserve_back(CB_C22, 1);
        noc_async_read_tile(tile_id, acc_c22, get_write_ptr(CB_C22));

        noc_async_read_barrier();

        cb_push_back(CB_MCX, 1);
        cb_push_back(CB_MCY, 1);
        cb_push_back(CB_MCZ, 1);
        cb_push_back(CB_C00, 1);
        cb_push_back(CB_C01, 1);
        cb_push_back(CB_C02, 1);
        cb_push_back(CB_C11, 1);
        cb_push_back(CB_C12, 1);
        cb_push_back(CB_C22, 1);
    }
}
