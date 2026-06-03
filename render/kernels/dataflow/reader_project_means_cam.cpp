// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// project_means_cam READER kernel (NCRISC / NoC1).
//
// Streams 3 SoA fp32 tile streams (mx, my, mz) from DRAM into the 3 input CBs
// for this core's slice of chunks. Each tile is 32x32x4 = 4096 bytes of fp32
// (1024 Gaussians per tile).
//
// RUNTIME ARGS
//   0: mx_addr      DRAM base of mx tile buffer (4 KB per tile)
//   1: my_addr      DRAM base of my tile buffer
//   2: mz_addr      DRAM base of mz tile buffer
//   3: chunk_start  this core's first tile index (global)
//   4: num_chunks   number of tiles this core processes
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs in order: mx, my, mz. All DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t mx_addr     = get_arg_val<uint32_t>(0);
    uint32_t my_addr     = get_arg_val<uint32_t>(1);
    uint32_t mz_addr     = get_arg_val<uint32_t>(2);
    uint32_t chunk_start = get_arg_val<uint32_t>(3);
    uint32_t num_chunks  = get_arg_val<uint32_t>(4);

    constexpr uint32_t CB_MX = 0;
    constexpr uint32_t CB_MY = 1;
    constexpr uint32_t CB_MZ = 2;

    const uint32_t tile_bytes = get_tile_size(CB_MX);  // 4096 bytes (fp32 32x32)

    constexpr auto mx_args = TensorAccessorArgs<0>();
    constexpr auto my_args = TensorAccessorArgs<mx_args.next_compile_time_args_offset()>();
    constexpr auto mz_args = TensorAccessorArgs<my_args.next_compile_time_args_offset()>();

    const auto mx = TensorAccessor(mx_args, mx_addr, tile_bytes);
    const auto my = TensorAccessor(my_args, my_addr, tile_bytes);
    const auto mz = TensorAccessor(mz_args, mz_addr, tile_bytes);

    if (num_chunks == 0) {
        return;
    }

    for (uint32_t k = 0; k < num_chunks; k++) {
        const uint32_t tile_id = chunk_start + k;

        cb_reserve_back(CB_MX, 1);
        noc_async_read_tile(tile_id, mx, get_write_ptr(CB_MX));
        cb_reserve_back(CB_MY, 1);
        noc_async_read_tile(tile_id, my, get_write_ptr(CB_MY));
        cb_reserve_back(CB_MZ, 1);
        noc_async_read_tile(tile_id, mz, get_write_ptr(CB_MZ));
        noc_async_read_barrier();
        cb_push_back(CB_MX, 1);
        cb_push_back(CB_MY, 1);
        cb_push_back(CB_MZ, 1);
    }
}
