// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// project_means_cam WRITER kernel (BRISC / NoC0).
//
// For each chunk this core processed, drains 3 output tiles (mc_x, mc_y, mc_z)
// from CBs and writes them to DRAM at the appropriate tile indices.
//
// RUNTIME ARGS
//   0: mcx_addr     DRAM base of mc_x output tile buffer
//   1: mcy_addr     DRAM base of mc_y output tile buffer
//   2: mcz_addr     DRAM base of mc_z output tile buffer
//   3: chunk_start  this core's first tile index (global)
//   4: num_chunks   number of tiles this core writes
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs in order: mcx, mcy, mcz. All DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t mcx_addr    = get_arg_val<uint32_t>(0);
    uint32_t mcy_addr    = get_arg_val<uint32_t>(1);
    uint32_t mcz_addr    = get_arg_val<uint32_t>(2);
    uint32_t chunk_start = get_arg_val<uint32_t>(3);
    uint32_t num_chunks  = get_arg_val<uint32_t>(4);

    constexpr uint32_t CB_MCX = 3;
    constexpr uint32_t CB_MCY = 4;
    constexpr uint32_t CB_MCZ = 5;

    const uint32_t tile_bytes = get_tile_size(CB_MCX);  // 4096 bytes (fp32 32x32)

    constexpr auto mcx_args = TensorAccessorArgs<0>();
    constexpr auto mcy_args = TensorAccessorArgs<mcx_args.next_compile_time_args_offset()>();
    constexpr auto mcz_args = TensorAccessorArgs<mcy_args.next_compile_time_args_offset()>();

    const auto mcx = TensorAccessor(mcx_args, mcx_addr, tile_bytes);
    const auto mcy = TensorAccessor(mcy_args, mcy_addr, tile_bytes);
    const auto mcz = TensorAccessor(mcz_args, mcz_addr, tile_bytes);

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t k = 0; k < num_chunks; k++) {
        const uint32_t tile_id = chunk_start + k;

        cb_wait_front(CB_MCX, 1);
        noc_async_write_tile(tile_id, mcx, get_read_ptr(CB_MCX));
        cb_wait_front(CB_MCY, 1);
        noc_async_write_tile(tile_id, mcy, get_read_ptr(CB_MCY));
        cb_wait_front(CB_MCZ, 1);
        noc_async_write_tile(tile_id, mcz, get_read_ptr(CB_MCZ));
        noc_async_write_barrier();
        cb_pop_front(CB_MCX, 1);
        cb_pop_front(CB_MCY, 1);
        cb_pop_front(CB_MCZ, 1);
    }
}
