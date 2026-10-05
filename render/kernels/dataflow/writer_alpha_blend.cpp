// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "img_pack_u8.h"

// Task #280: shifted ids / compile-time args in the fused mat+blend program.
#ifndef BLEND_CB_BASE
#define BLEND_CB_BASE 0
#endif
#ifndef BLEND_CTA_BASE
#define BLEND_CTA_BASE 0
#endif

// Alpha-blend WRITER kernel (BRISC, NoC0; see DataMovementProcessor::RISCV_0
// in alpha_blend.cpp).
//
// ROLE
// ----
// The mirror of the reader on the output side. For each screen tile this
// core processed, the compute kernel pushes 3 bf16 32x32 tiles (R, G, B in
// that order) to CB_COLOR_OUT. The writer packs them into the final 8-bit RGB
// image on device (task #61, img_pack_u8.h): bf16 -> u8 with the host's
// uint8(clip(x,0,1)*255) rule, microblock permutation undone, RGB interleaved
// into a 32-row x 96-byte staging block, then written as 32 row segments into
// the row-major (rows, pitch) u8 image buffer. The host reads the image back
// as-is: no bf16->fp32 assemble, half the D2H bytes. The writer uses NoC0
// while the reader uses NoC1.
//
// TILE ASSIGNMENT (task #60)
// ---------------------------
// Tiles are claimed dynamically by the reader; it queues each claimed screen
// tile id in CB_TILE_Q (in blend order) and 0xFFFFFFFF at the end.
//
// PER-TILE WORK
// -------------
//   1. cb_wait_front(CB_COLOR_OUT, 3)   wait for compute's R/G/B push
//   2. pack the 3 channels into a CB_IMG_U8 staging block, pop CB_COLOR_OUT
//   3. async-write the 32 rows to image rows ty*32 + i, byte offset tx*96
//
// RUNTIME ARGS
//   0: out_addr           DRAM base of the u8 image buffer (one page per image row)
//   1-3: tile_ids_addr, lpt_meta_addr, core_index (unused since task #60)
//   4: tiles_x
//   5: pitch              image buffer row pitch in bytes (page size)
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs in order: out, tile_ids, lpt_meta.

// Task #83: fine per-tile zones for floor attribution (host env
// GSPLAT_TT_BLEND_PROF=1; compiled out by default).
#if defined(BLEND_PROF) && BLEND_PROF
#define BLEND_PZ(name) DeviceZoneScopedN(name)
#else
#define BLEND_PZ(name) ((void)0)
#endif

void kernel_main() {
    uint32_t out_addr        = get_arg_val<uint32_t>(0);
    uint32_t tile_ids_addr   = get_arg_val<uint32_t>(1);
    const uint32_t lpt_meta_addr = get_arg_val<uint32_t>(2);
    const uint32_t core_index    = get_arg_val<uint32_t>(3);
    const uint32_t tiles_x       = get_arg_val<uint32_t>(4);
    const uint32_t pitch         = get_arg_val<uint32_t>(5);

    constexpr uint32_t CB_COLOR_OUT = BLEND_CB_BASE + 16;
    constexpr uint32_t CB_IMG_U8 = BLEND_CB_BASE + 8;  // 32 x 96 B RGB staging block
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);
    constexpr uint32_t tile_ids_page_bytes = 64;

    constexpr auto out_args      = TensorAccessorArgs<BLEND_CTA_BASE>();
    constexpr auto tile_ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();

    const auto out          = TensorAccessor(out_args,      out_addr,      pitch);
    (void)tile_ids_addr; (void)lpt_meta_addr; (void)core_index;
    // Task #60: the reader claims tiles dynamically and queues each claimed
    // screen tile id here (CB_TILE_Q) before its data; 0xFFFFFFFF ends the stream.
    constexpr uint32_t CB_TILE_Q = BLEND_CB_BASE + 13;

    // Main per-tile loop: pack the 3 R/G/B tiles compute pushed for this
    // screen tile into u8 image rows and write them to the image buffer.
    using img_pack_u8::ROW_BYTES;
    const uint32_t stage_addr = get_write_ptr(CB_IMG_U8);
    auto stage = reinterpret_cast<volatile uint8_t*>(stage_addr);
    for (;;) {
        cb_wait_front(CB_TILE_Q, 1);
        const uint32_t screen_tile =
            reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_Q))[0];
        cb_pop_front(CB_TILE_Q, 1);
        if (screen_tile == 0xFFFFFFFFu) {
            break;
        }
        const uint32_t ty = screen_tile / tiles_x;
        const uint32_t tx = screen_tile - ty * tiles_x;

        // CB_COLOR_OUT has depth 6 (multiple of 3) on the host side so this
        // 3-tile batch never straddles a CB wrap.
        {
            BLEND_PZ("wr_wait_out");
            cb_wait_front(CB_COLOR_OUT, 3);
        }
        uint32_t read_ptr = get_read_ptr(CB_COLOR_OUT);
        // The previous tile's rows must have left the staging block (they had
        // a whole blend tile to drain, so this does not wait in practice).
        noc_async_writes_flushed();
        BLEND_PZ("wr_pack");
        for (uint32_t ch = 0; ch < 3; ch++) {
            img_pack_u8::pack_channel(
                reinterpret_cast<volatile uint32_t*>(read_ptr), stage, ch);
            read_ptr += tile_bytes;
        }
        cb_pop_front(CB_COLOR_OUT, 3);

        const uint32_t row0 = ty * 32u;
        const uint32_t col_off = tx * ROW_BYTES;
        for (uint32_t i = 0; i < 32u; i++) {
            noc_async_write(stage_addr + i * ROW_BYTES,
                            out.get_noc_addr(row0 + i, col_off), ROW_BYTES);
        }
    }
    noc_async_write_barrier();
}
