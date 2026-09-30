// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Alpha-blend WRITER kernel (BRISC, NoC0; see DataMovementProcessor::RISCV_0
// in alpha_blend.cpp).
//
// ROLE
// ----
// The mirror of the reader on the output side. For each screen tile this
// core processed, the compute kernel pushes 3 fp32 32x32 tiles (R, G, B in
// that order) to CB_COLOR_OUT. We async-write them to consecutive DRAM
// pages at offsets `3*screen_tile + {0, 1, 2}` of the output buffer. The
// writer uses NoC0 while the reader uses NoC1 — bidirectional dual-NoC
// torus lets I/O overlap.
//
// TILE ASSIGNMENT (task #60)
// ---------------------------
// Tiles are claimed dynamically by the reader; it queues each claimed screen
// tile id in CB_TILE_Q (in blend order) and 0xFFFFFFFF at the end.
//
// PER-TILE WORK
// -------------
//   1. cb_wait_front(CB_COLOR_OUT, 3)   wait for compute's R/G/B push
//   2. async-write 3 channels to DRAM pages 3*screen_tile + {0, 1, 2}
//   3. cb_pop_front(CB_COLOR_OUT, 3)
//
// RUNTIME ARGS
//   0: out_addr           DRAM base of the (num_tiles, 3, 32, 32) fp32 output buffer
//   1-3: tile_ids_addr, lpt_meta_addr, core_index (unused since task #60)
//
// COMPILE-TIME ARGS: 2 TensorAccessorArgs in order: out, tile_ids.

void kernel_main() {
    uint32_t out_addr        = get_arg_val<uint32_t>(0);
    uint32_t tile_ids_addr   = get_arg_val<uint32_t>(1);
    const uint32_t lpt_meta_addr = get_arg_val<uint32_t>(2);
    const uint32_t core_index    = get_arg_val<uint32_t>(3);

    constexpr uint32_t CB_COLOR_OUT = 16;
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);
    constexpr uint32_t tile_ids_page_bytes = 64;

    constexpr auto out_args      = TensorAccessorArgs<0>();
    constexpr auto tile_ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();

    const auto out          = TensorAccessor(out_args,      out_addr,      tile_bytes);
    (void)tile_ids_addr; (void)lpt_meta_addr; (void)core_index;
    // Task #60: the reader claims tiles dynamically and queues each claimed
    // screen tile id here (CB_TILE_Q) before its data; 0xFFFFFFFF ends the stream.
    constexpr uint32_t CB_TILE_Q = 13;

    // Main per-tile loop: drain 3 R/G/B tiles compute pushed for this screen
    // tile and async-write them to their global slots in the output buffer.
    for (;;) {
        cb_wait_front(CB_TILE_Q, 1);
        const uint32_t screen_tile =
            reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_Q))[0];
        cb_pop_front(CB_TILE_Q, 1);
        if (screen_tile == 0xFFFFFFFFu) {
            break;
        }

        // Wait for compute's batch of 3 tiles (R, then G, then B) in order.
        // Note: CB_COLOR_OUT has depth 6 (multiple of 3) on the host side so
        // this 3-tile batch never straddles a CB wrap, which would break the
        // `read_ptr += tile_bytes` arithmetic below.
        cb_wait_front(CB_COLOR_OUT, 3);
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        for (uint32_t ch = 0; ch < 3; ch++) {
            // Output buffer layout: (num_tiles, 3, 32, 32) fp32. Tile-major
            // order with R/G/B interleaved per screen tile, so the global
            // page index for channel `ch` of `screen_tile` is `3*screen_tile + ch`.
            uint32_t out_tile_id = 3 * screen_tile + ch;
            noc_async_write_tile(out_tile_id, out, read_ptr);
            read_ptr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(CB_COLOR_OUT, 3);
    }
}
