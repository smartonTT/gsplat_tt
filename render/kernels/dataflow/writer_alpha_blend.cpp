// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "img_pack_u8.h"
#include "tile_window.h"

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
//   6-8 (OUT_PINNED only, task #367): pcie_xy_enc, addr_lo, addr_hi of a pinned,
//        NoC-mapped host image buffer with the same (rows, pitch) layout; the
//        rows go straight to host memory over PCIe instead of to out_addr.
//
//   9-10 (BLEND_TRAIN only, task #708): train_addr, image_tiles_x. Per screen
//        tile compute also pushes 4 fp32 tiles (R, G, B, T; microblock layout)
//        to CB_TRAIN_OUT; they go as one 16 KB page to page ty * image_tiles_x + tx
//        of the train buffer.
//
// COMPILE-TIME ARGS: 3 TensorAccessorArgs in order: out, tile_ids, lpt_meta
// (+ train with BLEND_TRAIN).

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
    // Task #623: arg 4 may carry a tile window (tile_window::pack); the image
    // stays full size (pitch), the window's tiles land at their global origin.
    const tile_window::Origin win = tile_window::decode(get_arg_val<uint32_t>(4));
    const uint32_t tiles_x       = win.w;
    const uint32_t pitch         = get_arg_val<uint32_t>(5);

    constexpr uint32_t CB_COLOR_OUT = BLEND_CB_BASE + 16;
    constexpr uint32_t CB_IMG_U8 = BLEND_CB_BASE + 8;  // 32 x 96 B RGB staging block
    const uint32_t tile_bytes = get_tile_size(CB_COLOR_OUT);
    constexpr uint32_t tile_ids_page_bytes = 64;

    constexpr auto out_args      = TensorAccessorArgs<BLEND_CTA_BASE>();
    constexpr auto tile_ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();

    const auto out          = TensorAccessor(out_args,      out_addr,      pitch);
#if defined(BLEND_TRAIN) && BLEND_TRAIN
    constexpr uint32_t CB_TRAIN_OUT = BLEND_CB_BASE + 17;
    constexpr uint32_t TRAIN_PAGE_BYTES = 4u * 32u * 32u * 4u;
    constexpr auto train_args = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    const auto train = TensorAccessor(train_args, get_arg_val<uint32_t>(9), TRAIN_PAGE_BYTES);
    const uint32_t image_tiles_x = get_arg_val<uint32_t>(10);
#endif
#if defined(OUT_PINNED) && OUT_PINNED
    const uint32_t pin_xy = get_arg_val<uint32_t>(6);
    const uint64_t pin_base = (static_cast<uint64_t>(get_arg_val<uint32_t>(8)) << 32) |
                              get_arg_val<uint32_t>(7);
    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
#endif
    (void)tile_ids_addr; (void)lpt_meta_addr; (void)core_index;
    // Task #60: the reader claims tiles dynamically and queues each claimed
    // screen tile id here (CB_TILE_Q) before its data; 0xFFFFFFFF ends the stream.
    constexpr uint32_t CB_TILE_Q = BLEND_CB_BASE + 13;

    // Main per-tile loop: pack the 3 R/G/B tiles compute pushed for this
    // screen tile into u8 image rows and write them to the image buffer.
    using img_pack_u8::ROW_BYTES;
    const uint32_t stage_addr = get_write_ptr(CB_IMG_U8);
    auto stage = reinterpret_cast<volatile uint8_t*>(stage_addr);
    // One screen tile: pack the 3 R/G/B tiles compute pushed for it into u8
    // image rows and write them to the image buffer.
    auto write_tile = [&](uint32_t screen_tile) {
        const uint32_t ty0 = screen_tile / tiles_x;
        const uint32_t tx = screen_tile - ty0 * tiles_x + win.x0;
        const uint32_t ty = ty0 + win.y0;

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
#if defined(BLEND_TRAIN) && BLEND_TRAIN
        cb_wait_front(CB_TRAIN_OUT, 4);
        noc_async_write(get_read_ptr(CB_TRAIN_OUT), train.get_noc_addr(ty * image_tiles_x + tx),
                        TRAIN_PAGE_BYTES);
        noc_async_writes_flushed();
        cb_pop_front(CB_TRAIN_OUT, 4);
#endif

        const uint32_t row0 = ty * 32u;
        const uint32_t col_off = tx * ROW_BYTES;
#if defined(OUT_PINNED) && OUT_PINNED
#if defined(BLEND_WRITER_PRE)
        // Task #542: the pre-phase's other NoC writes reuse write_cmd_buf.
        noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
#endif
        uint64_t dst = pin_base + static_cast<uint64_t>(row0) * pitch + col_off;
        for (uint32_t i = 0; i < 32u; i++) {
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                noc_index, stage_addr + i * ROW_BYTES, pin_xy, dst, ROW_BYTES, 1);
            dst += pitch;
        }
#else
        for (uint32_t i = 0; i < 32u; i++) {
            noc_async_write(stage_addr + i * ROW_BYTES,
                            out.get_noc_addr(row0 + i, col_off), ROW_BYTES);
        }
#endif
    };
    bool ended = false;
#if defined(BLEND_WRITER_PRE)
    // Task #542 (B4a): BLEND_WRITER_PRE(poll) runs other work on this RISC
    // first and calls poll() often; poll writes a tile only when its id and
    // all 3 colour tiles are already there, so it never blocks.
    auto poll = [&]() {
        if (ended || !cb_pages_available_at_front(CB_TILE_Q, 1)) return;
        const uint32_t screen_tile =
            reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_Q))[0];
        if (screen_tile == 0xFFFFFFFFu) {
            cb_pop_front(CB_TILE_Q, 1);
            ended = true;
            return;
        }
        if (!cb_pages_available_at_front(CB_COLOR_OUT, 3)) return;
        cb_pop_front(CB_TILE_Q, 1);
        write_tile(screen_tile);
    };
    BLEND_WRITER_PRE(poll);
#endif
#if defined(BLEND_TRAIN) && BLEND_TRAIN && defined(TRAIN_OPC_WRITER) && TRAIN_OPC_WRITER
    // Task #730: the reader queues {slab L1 address, record count} per bulk
    // slab on CB_OPC_REQ; this RISC gathers each record's fp32 [op, r, g, b]
    // (rec[7] = scene index, 64 B DRAM pages of 4 entries) into CB_TRAIN_OPC at
    // g * 16, the same bytes the reader's #717 gather wrote. The slab slot stays
    // valid until compute pops it, which it does only after this push.
    constexpr uint32_t CB_TRAIN_OPC = BLEND_CB_BASE + 9;
    constexpr uint32_t CB_OPC_REQ = BLEND_CB_BASE + 14;
    constexpr uint32_t OPC_SLOT = 4096u;                 // BULK_REC_SLOT (bucket fit 8192)
    constexpr uint32_t OPC_STAGE = 128u * 1024u;         // DRAM page staging
#ifndef TRAIN_OPC_BATCH
#define TRAIN_OPC_BATCH 64u
#endif
    static_assert(TRAIN_OPC_BATCH >= 1u && TRAIN_OPC_BATCH * 64u <= 128u * 1024u, "staging overflows the slot");
    const InterleavedAddrGen<true> opc_gen{get_arg_val<uint32_t>(11), 64u};
    auto serve = [&]() -> bool {
        if (!cb_pages_available_at_front(CB_OPC_REQ, 1) ||
            !cb_pages_reservable_at_back(CB_TRAIN_OPC, OPC_SLOT)) {
            return false;
        }
        invalidate_l1_cache();
        auto req = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_OPC_REQ));
        auto recs = reinterpret_cast<volatile uint32_t*>(req[0]);
        const uint32_t n_rec = req[1];
        cb_reserve_back(CB_TRAIN_OPC, OPC_SLOT);
        const uint32_t side = get_write_ptr(CB_TRAIN_OPC);
        auto dst = reinterpret_cast<volatile uint32_t*>(side);
        auto stg = reinterpret_cast<volatile uint32_t*>(side + OPC_STAGE);
        uint32_t g0 = 0;
        while (g0 < n_rec) {
            const uint32_t g1 = (n_rec - g0 > TRAIN_OPC_BATCH) ? g0 + TRAIN_OPC_BATCH : n_rec;
            for (uint32_t g = g0; g < g1; ++g) {
                noc_async_read(get_noc_addr(recs[g * 8u + 7u] >> 2, opc_gen),
                               side + OPC_STAGE + (g - g0) * 64u, 64u);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t g = g0; g < g1; ++g) {
                const uint32_t s = (g - g0) * 16u + (recs[g * 8u + 7u] & 3u) * 4u;
                dst[g * 4u + 0u] = stg[s + 0u];
                dst[g * 4u + 1u] = stg[s + 1u];
                dst[g * 4u + 2u] = stg[s + 2u];
                dst[g * 4u + 3u] = stg[s + 3u];
            }
            g0 = g1;
        }
        asm volatile("fence" ::: "memory");
        cb_push_back(CB_TRAIN_OPC, OPC_SLOT);
        cb_pop_front(CB_OPC_REQ, 1);
        return true;
    };
    // Never block on one queue: a tile's colour waits on gathers still queued.
    // Write a tile only when its id and all 3 colour tiles are there.
    while (!ended) {
        if (serve()) {
            continue;
        }
        if (!cb_pages_available_at_front(CB_TILE_Q, 1)) {
            continue;
        }
        const uint32_t screen_tile =
            reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_Q))[0];
        if (screen_tile == 0xFFFFFFFFu) {
            cb_pop_front(CB_TILE_Q, 1);
            ended = true;
            break;
        }
        if (!cb_pages_available_at_front(CB_COLOR_OUT, 3)) {
            continue;
        }
        cb_pop_front(CB_TILE_Q, 1);
        write_tile(screen_tile);
    }
#endif
    while (!ended) {
        cb_wait_front(CB_TILE_Q, 1);
        const uint32_t screen_tile =
            reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TILE_Q))[0];
        cb_pop_front(CB_TILE_Q, 1);
        if (screen_tile == 0xFFFFFFFFu) {
            break;
        }
        write_tile(screen_tile);
    }
    noc_async_write_barrier();
}
