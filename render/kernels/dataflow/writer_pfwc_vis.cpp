// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc WRITER for lever 2 (task #99, GSPLAT_TT_SFPU_VIS), BRISC / NoC0.
// writer_pfwc.cpp plus the two SFPU word tiles. Per 1024-gaussian tile it
//   * resolves RECHECK words (an inf/NaN input) with the soft-float predicate
//     and K1 rectangle (vis_tile::exact_words; the opacity tile is read from
//     DRAM only then),
//   * builds the 1024-bit visibility mask (the proj_count layout) and counts
//     the visible gaussians and their pairs (vis_tile::classify_tile),
//   * writes the 10 output tiles and the mask page,
// and at the end writes the [visible, pairs] counts of its tiles into the
// dense per-tile counts buffer (1 KB pages, partial page writes at matching
// L1 / DRAM offsets, like the gather scatter's boundary pages). The writer was
// idle behind the SFPU, so the classify loop runs in its shadow.
//
// RUNTIME ARGS
//   0..7  : DRAM bases m2x, m2y, depth, a, b, c, rx, ry (tile pages)
//   8, 9  : DRAM bases tpg word, aabb word (tile pages)
//   10    : visibility mask base (128 B page per tile)
//   11    : per-tile counts base (1 KB pages)
//   12    : scene opacity base (tile pages, RECHECK only)
//   13    : chunk_start   14: num_chunks   15: N
//   16..20: k_near, min_opacity, image width, image height, max_radius (fp32 bits)
//   21    : tiles_x   22: tiles_y   23: tile_size (power of two)
//
// COMPILE-TIME ARGS: 13 TensorAccessorArgs in runtime-arg order 0..12.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "vis_tile.h"

void kernel_main() {
    uint32_t addr[13];
    for (uint32_t k = 0; k < 13; k++) addr[k] = get_arg_val<uint32_t>(k);
    const uint32_t chunk_start = get_arg_val<uint32_t>(13);
    const uint32_t num_chunks = get_arg_val<uint32_t>(14);
    const uint32_t N = get_arg_val<uint32_t>(15);
    vis_tile::Params prm;
    prm.k_near = get_arg_val<uint32_t>(16);
    prm.min_opacity = get_arg_val<uint32_t>(17);
    prm.img_w = get_arg_val<uint32_t>(18);
    prm.img_h = get_arg_val<uint32_t>(19);
    prm.max_radius = get_arg_val<uint32_t>(20);
    prm.tiles_x = get_arg_val<uint32_t>(21);
    prm.tiles_y = get_arg_val<uint32_t>(22);
    const uint32_t tile_size = get_arg_val<uint32_t>(23);
    prm.tile_shift = dm_fp32::pow2_shift(tile_size);
    prm.inv_tile = 1.0f / static_cast<float>(tile_size);

    constexpr uint32_t CB_M2X = 9, CB_M2Y = 10, CB_DEP = 11, CB_A = 12, CB_B = 13, CB_C = 14,
                       CB_RX = 15, CB_RY = 16, CB_TPG = 35, CB_AABB = 36;
    constexpr uint32_t CB_MASK = 37;    // 128 B mask staging
    constexpr uint32_t CB_COUNTS = 38;  // counts staging (+64 B alignment slack)
    constexpr uint32_t CB_OPT = 39;     // opacity tile for RECHECK
    constexpr uint32_t OUT_CB[10] = {CB_M2X, CB_M2Y, CB_DEP, CB_A,  CB_B,
                                     CB_C,   CB_RX,  CB_RY,  CB_TPG, CB_AABB};
    constexpr uint32_t MASK_BYTES = vis_tile::MASK_WORDS * 4;
    constexpr uint32_t CPAGE = vis_tile::COUNTS_PAGE_BYTES;
    constexpr uint32_t CTILES = vis_tile::COUNTS_TILES_PER_PAGE;

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
    constexpr auto a9 = TensorAccessorArgs<a8.next_compile_time_args_offset()>();
    constexpr auto a10 = TensorAccessorArgs<a9.next_compile_time_args_offset()>();
    constexpr auto a11 = TensorAccessorArgs<a10.next_compile_time_args_offset()>();
    constexpr auto a12 = TensorAccessorArgs<a11.next_compile_time_args_offset()>();

    const auto acc_m2x = TensorAccessor(a0, addr[0], tile_bytes);
    const auto acc_m2y = TensorAccessor(a1, addr[1], tile_bytes);
    const auto acc_dep = TensorAccessor(a2, addr[2], tile_bytes);
    const auto acc_a = TensorAccessor(a3, addr[3], tile_bytes);
    const auto acc_b = TensorAccessor(a4, addr[4], tile_bytes);
    const auto acc_c = TensorAccessor(a5, addr[5], tile_bytes);
    const auto acc_rx = TensorAccessor(a6, addr[6], tile_bytes);
    const auto acc_ry = TensorAccessor(a7, addr[7], tile_bytes);
    const auto acc_tpg = TensorAccessor(a8, addr[8], tile_bytes);
    const auto acc_aabb = TensorAccessor(a9, addr[9], tile_bytes);
    const auto acc_mask = TensorAccessor(a10, addr[10], MASK_BYTES);
    const auto acc_cnt = TensorAccessor(a11, addr[11], CPAGE);
    const auto acc_op = TensorAccessor(a12, addr[12], tile_bytes);

    if (num_chunks == 0) {
        return;
    }

    const uint32_t l1_mask = get_write_ptr(CB_MASK);
    const uint32_t l1_op = get_write_ptr(CB_OPT);
    // Counts staging: tile t's [visible, pairs] at byte t * 8 - page0 * CPAGE, so
    // every partial page write has the same offset mod 64 in L1 and DRAM.
    const uint32_t l1_cnt = (get_write_ptr(CB_COUNTS) + 63u) & ~63u;
    const uint32_t page0 = chunk_start / CTILES;
    auto cnt = reinterpret_cast<volatile uint32_t*>(l1_cnt);
    auto mw = reinterpret_cast<volatile uint32_t*>(l1_mask);
    auto opw = reinterpret_cast<volatile uint32_t*>(l1_op);

    for (uint32_t k = 0; k < num_chunks; k++) {
        const uint32_t t = chunk_start + k;
        for (uint32_t o = 0; o < 10; o++) cb_wait_front(OUT_CB[o], 1);

        auto p_m2x = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2X));
        auto p_m2y = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2Y));
        auto p_dep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_DEP));
        auto p_rx = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RX));
        auto p_ry = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RY));
        auto p_tpg = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TPG));
        auto p_aabb = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_AABB));

        const uint32_t tbase = t * vis_tile::TILE_ELEMS;
        const uint32_t n_el = (tbase >= N) ? 0u
            : (N - tbase < vis_tile::TILE_ELEMS ? N - tbase : vis_tile::TILE_ELEMS);
        bool op_loaded = false;
        auto get_inputs = [&](uint32_t il, uint32_t& tz, uint32_t& op, uint32_t& mx,
                              uint32_t& my, uint32_t& rx, uint32_t& ry) {
            if (!op_loaded) {
                noc_async_read_tile(t, acc_op, l1_op);
                noc_async_read_barrier();
                op_loaded = true;
            }
            tz = p_dep[il];
            op = opw[il];
            mx = p_m2x[il];
            my = p_m2y[il];
            rx = p_rx[il];
            ry = p_ry[il];
        };
        uint32_t vc = 0, pc = 0;
        vis_tile::classify_tile(p_tpg, p_aabb, n_el, mw, prm, get_inputs, &vc, &pc);
        const uint32_t co = (t - page0 * CTILES) * vis_tile::COUNT_WORDS;
        cnt[co + 0] = vc;
        cnt[co + 1] = pc;

        noc_async_write(l1_mask, get_noc_addr(t, acc_mask), MASK_BYTES);
        noc_async_write_tile(t, acc_m2x, get_read_ptr(CB_M2X));
        noc_async_write_tile(t, acc_m2y, get_read_ptr(CB_M2Y));
        noc_async_write_tile(t, acc_dep, get_read_ptr(CB_DEP));
        noc_async_write_tile(t, acc_a, get_read_ptr(CB_A));
        noc_async_write_tile(t, acc_b, get_read_ptr(CB_B));
        noc_async_write_tile(t, acc_c, get_read_ptr(CB_C));
        noc_async_write_tile(t, acc_rx, get_read_ptr(CB_RX));
        noc_async_write_tile(t, acc_ry, get_read_ptr(CB_RY));
        noc_async_write_tile(t, acc_tpg, get_read_ptr(CB_TPG));
        noc_async_write_tile(t, acc_aabb, get_read_ptr(CB_AABB));
        noc_async_write_barrier();

        for (uint32_t o = 0; o < 10; o++) cb_pop_front(OUT_CB[o], 1);
    }

    // Counts of tiles [chunk_start, chunk_start + num_chunks), page by page.
    const uint32_t t_end = chunk_start + num_chunks;
    for (uint32_t p = page0; p * CTILES < t_end; p++) {
        const uint32_t lo = (p * CTILES > chunk_start) ? p * CTILES : chunk_start;
        const uint32_t hi = ((p + 1) * CTILES < t_end) ? (p + 1) * CTILES : t_end;
        const uint32_t off = (lo - p * CTILES) * vis_tile::COUNT_WORDS * 4;
        noc_async_write(l1_cnt + (p - page0) * CPAGE + off, get_noc_addr(p, acc_cnt) + off,
                        (hi - lo) * vis_tile::COUNT_WORDS * 4);
    }
    noc_async_write_barrier();
}
