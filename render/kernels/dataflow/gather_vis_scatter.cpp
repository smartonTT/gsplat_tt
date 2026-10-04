// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gather_visible SCATTER for lever 2 (task #99, GSPLAT_TT_SFPU_VIS). Runs on
// BRISC and NCRISC of every core, one (core, mover) slot each. Differences from
// gather_visible_scatter.cpp:
//   * no count pass: the visibility mask and the per-slot parameters come from
//     the pfwc writer and gather_vis_scan.cpp. A slot owns a contiguous range
//     [q0, q0 + qn) of the compaction sequence (vis_tile::SeqMap), cut by the
//     scan at about equal cost, so the scatter is balanced while the compact
//     order stays the legacy one;
//   * only the streams the default chain reads are written: proj_m_depth
//     (sort) and the AoS blend record (sort / materialize), plus the two
//     tile_assign inputs that replace ta_gauss_aabb and the TA scans:
//     proj_m_offs (exclusive pair offset; the is_last slot pads its last page
//     with P) and proj_m_aabb (packed min_x / min_y / w). With check != 0 it
//     also writes proj_m_px/py/rx/ry for the legacy K1 cross-check;
//   * a page is handed to the NoC with noc_async_writes_flushed (L1 reusable)
//     instead of a full write barrier; one barrier at the end.
// Boundary pages shared with the neighbouring slot get disjoint partial writes
// at matching L1 / DRAM offsets, as in the legacy scatter.
//
// RUNTIME ARGS
//   0..9  : inputs m2x, m2y, depth, a, b, c (pfwc, conic), opacity, col_r,
//           col_g, col_b (scene)   (tile pages)
//   10, 11: tpg word, aabb word (pfwc, tile pages)
//   12, 13: rx, ry (pfwc, tile pages; read when check != 0)
//   14..17: outputs depth, blendrec (64 B record per page), offs, aabb
//   18..21: outputs px, py, rx, ry (written when check != 0)
//   22    : visibility mask base (128 B page per tile)
//   23    : slot pages base (64 B)
//   24: N   25: num_cores (tile stride)   26: num_tiles   27: slot   28: mover
//   29: check
//   30, 31: scene_puboc01 / scene_puboc23 (tile pages; EMIT_PUBOC): record
//       words 10 / 11 precomputed per scene (task #122, CBs 24 / 25). 0 = the
//       scene has a NaN: the visible lanes are packed here (to_unorm16).
//
// COMPILE-TIME ARGS: 24 TensorAccessorArgs in runtime-arg order 0..23 (the
// puboc tiles reuse the opacity accessor's, all are DRAM interleaved).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"

#ifndef EMIT_PUBOC
#define EMIT_PUBOC 0
#endif
#include "vis_tile.h"

namespace {
constexpr uint32_t TILE_BYTES = 4096;
constexpr uint32_t PAGE_ELEMS = 16;
constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t REC_WORDS = 16;
constexpr uint32_t MASK_BYTES = vis_tile::MASK_WORDS * 4;
constexpr uint32_t N_IN = 14;   // input tile streams
constexpr uint32_t N_ACC = 24;  // accessors
}  // namespace

void kernel_main() {
    DeviceZoneScopedN("proj_scatter");
    uint32_t addr[N_ACC];
    for (uint32_t k = 0; k < N_ACC; k++) addr[k] = get_arg_val<uint32_t>(k);
    const uint32_t N = get_arg_val<uint32_t>(24);
    const uint32_t num_cores = get_arg_val<uint32_t>(25);
    const uint32_t num_tiles = get_arg_val<uint32_t>(26);
    const uint32_t slot_id = get_arg_val<uint32_t>(27);
    const uint32_t mover = get_arg_val<uint32_t>(28);
    const bool check = get_arg_val<uint32_t>(29) != 0;
    const uint32_t pub01 = get_arg_val<uint32_t>(30);
    const uint32_t pub23 = get_arg_val<uint32_t>(31);
    (void)N;

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
    constexpr auto a13 = TensorAccessorArgs<a12.next_compile_time_args_offset()>();
    constexpr auto a14 = TensorAccessorArgs<a13.next_compile_time_args_offset()>();
    constexpr auto a15 = TensorAccessorArgs<a14.next_compile_time_args_offset()>();
    constexpr auto a16 = TensorAccessorArgs<a15.next_compile_time_args_offset()>();
    constexpr auto a17 = TensorAccessorArgs<a16.next_compile_time_args_offset()>();
    constexpr auto a18 = TensorAccessorArgs<a17.next_compile_time_args_offset()>();
    constexpr auto a19 = TensorAccessorArgs<a18.next_compile_time_args_offset()>();
    constexpr auto a20 = TensorAccessorArgs<a19.next_compile_time_args_offset()>();
    constexpr auto a21 = TensorAccessorArgs<a20.next_compile_time_args_offset()>();
    constexpr auto a22 = TensorAccessorArgs<a21.next_compile_time_args_offset()>();
    constexpr auto a23 = TensorAccessorArgs<a22.next_compile_time_args_offset()>();

    const auto i_m2x = TensorAccessor(a0, addr[0], TILE_BYTES);
    const auto i_m2y = TensorAccessor(a1, addr[1], TILE_BYTES);
    const auto i_dep = TensorAccessor(a2, addr[2], TILE_BYTES);
    const auto i_a = TensorAccessor(a3, addr[3], TILE_BYTES);
    const auto i_b = TensorAccessor(a4, addr[4], TILE_BYTES);
    const auto i_c = TensorAccessor(a5, addr[5], TILE_BYTES);
    const auto i_op = TensorAccessor(a6, addr[6], TILE_BYTES);
    const auto i_cr = TensorAccessor(a7, addr[7], TILE_BYTES);
    const auto i_cg = TensorAccessor(a8, addr[8], TILE_BYTES);
    const auto i_cb = TensorAccessor(a9, addr[9], TILE_BYTES);
    const auto i_tpg = TensorAccessor(a10, addr[10], TILE_BYTES);
    const auto i_aabb = TensorAccessor(a11, addr[11], TILE_BYTES);
    const auto i_rx = TensorAccessor(a12, addr[12], TILE_BYTES);
    const auto i_ry = TensorAccessor(a13, addr[13], TILE_BYTES);
    const auto o_dep = TensorAccessor(a14, addr[14], PAGE_BYTES);
    const auto o_rec = TensorAccessor(a15, addr[15], PAGE_BYTES);
    const auto o_offs = TensorAccessor(a16, addr[16], PAGE_BYTES);
    const auto o_aabb = TensorAccessor(a17, addr[17], PAGE_BYTES);
    const auto o_px = TensorAccessor(a18, addr[18], PAGE_BYTES);
    const auto o_py = TensorAccessor(a19, addr[19], PAGE_BYTES);
    const auto o_rx = TensorAccessor(a20, addr[20], PAGE_BYTES);
    const auto o_ry = TensorAccessor(a21, addr[21], PAGE_BYTES);
    const auto acc_mask = TensorAccessor(a22, addr[22], MASK_BYTES);
    const auto acc_slot = TensorAccessor(a23, addr[23], PAGE_BYTES);
    const auto i_q01 = TensorAccessor(a6, pub01, TILE_BYTES);
    const auto i_q23 = TensorAccessor(a6, pub23, TILE_BYTES);

    // L1: every CB holds one copy per mover; mover m uses the m-th.
    uint32_t l1_in[N_IN];
    for (uint32_t k = 0; k < N_IN; k++) l1_in[k] = get_write_ptr(k) + mover * TILE_BYTES;
    constexpr uint32_t CB_ODEP = 14, CB_OOFFS = 15, CB_OAABB = 16, CB_OPX = 17, CB_OPY = 18,
                       CB_ORX = 19, CB_ORY = 20, CB_OREC = 21, CB_MASK = 22, CB_SLOT = 23;
    const uint32_t l1_odep = get_write_ptr(CB_ODEP) + mover * PAGE_BYTES;
    const uint32_t l1_ooffs = get_write_ptr(CB_OOFFS) + mover * PAGE_BYTES;
    const uint32_t l1_oaabb = get_write_ptr(CB_OAABB) + mover * PAGE_BYTES;
    const uint32_t l1_opx = get_write_ptr(CB_OPX) + mover * PAGE_BYTES;
    const uint32_t l1_opy = get_write_ptr(CB_OPY) + mover * PAGE_BYTES;
    const uint32_t l1_orx = get_write_ptr(CB_ORX) + mover * PAGE_BYTES;
    const uint32_t l1_ory = get_write_ptr(CB_ORY) + mover * PAGE_BYTES;
    const uint32_t l1_orec = get_write_ptr(CB_OREC) + mover * PAGE_ELEMS * PAGE_BYTES;
    const uint32_t l1_mask = get_write_ptr(CB_MASK) + mover * MASK_BYTES;
    const uint32_t l1_slot = get_write_ptr(CB_SLOT) + mover * PAGE_BYTES;
    constexpr uint32_t CB_Q01 = 24, CB_Q23 = 25;
    const uint32_t l1_q01 = get_write_ptr(CB_Q01) + mover * TILE_BYTES;
    const uint32_t l1_q23 = get_write_ptr(CB_Q23) + mover * TILE_BYTES;

    auto in = [&](uint32_t k) { return reinterpret_cast<volatile uint32_t*>(l1_in[k]); };
    auto p_m2x = in(0), p_m2y = in(1), p_dep = in(2), p_a = in(3), p_b = in(4), p_c = in(5);
    auto p_op = in(6), p_cr = in(7), p_cg = in(8), p_cb = in(9), p_tpg = in(10), p_aabb = in(11);
    auto p_rx = in(12), p_ry = in(13);
    auto q01 = reinterpret_cast<volatile uint32_t*>(l1_q01);
    auto q23 = reinterpret_cast<volatile uint32_t*>(l1_q23);
    auto w_dep = reinterpret_cast<volatile uint32_t*>(l1_odep);
    auto w_offs = reinterpret_cast<volatile uint32_t*>(l1_ooffs);
    auto w_aabb = reinterpret_cast<volatile uint32_t*>(l1_oaabb);
    auto w_px = reinterpret_cast<volatile uint32_t*>(l1_opx);
    auto w_py = reinterpret_cast<volatile uint32_t*>(l1_opy);
    auto w_rx = reinterpret_cast<volatile uint32_t*>(l1_orx);
    auto w_ry = reinterpret_cast<volatile uint32_t*>(l1_ory);
    auto w_rec = reinterpret_cast<volatile uint32_t*>(l1_orec);
    auto mk = reinterpret_cast<volatile uint32_t*>(l1_mask);
    auto sp = reinterpret_cast<volatile uint32_t*>(l1_slot);

    noc_async_read(get_noc_addr(slot_id, acc_slot), l1_slot, PAGE_BYTES);
    noc_async_read_barrier();
    uint32_t g = sp[vis_tile::S_BASE];
    const uint32_t is_last = sp[vis_tile::S_IS_LAST];
    const uint32_t q0 = sp[vis_tile::S_Q0];
    const uint32_t qn = sp[vis_tile::S_QN];
    uint32_t pr = sp[vis_tile::S_PBASE];

    uint32_t cur_page = g / PAGE_ELEMS;
    uint32_t slot = g % PAGE_ELEMS;
    uint32_t flush_lo = slot;

    auto flush_page = [&](uint32_t page, uint32_t lo, uint32_t hi) {
        const uint32_t off = lo * 4, sz = (hi - lo) * 4;
        noc_async_write(l1_odep + off, get_noc_addr(page, o_dep) + off, sz);
        noc_async_write(l1_ooffs + off, get_noc_addr(page, o_offs) + off, sz);
        noc_async_write(l1_oaabb + off, get_noc_addr(page, o_aabb) + off, sz);
        if (check) {
            noc_async_write(l1_opx + off, get_noc_addr(page, o_px) + off, sz);
            noc_async_write(l1_opy + off, get_noc_addr(page, o_py) + off, sz);
            noc_async_write(l1_orx + off, get_noc_addr(page, o_rx) + off, sz);
            noc_async_write(l1_ory + off, get_noc_addr(page, o_ry) + off, sz);
        }
        // One 64 B record page per compact gaussian (page index == g).
        for (uint32_t s = lo; s < hi; ++s)
            noc_async_write(l1_orec + s * PAGE_BYTES, get_noc_addr(page * PAGE_ELEMS + s, o_rec),
                            PAGE_BYTES);
        noc_async_writes_flushed();  // staging reusable; completion at the end
    };

    // Record words 9..15 are always zero (gather_visible_scatter.cpp).
    for (uint32_t s = 0; s < PAGE_ELEMS; ++s)
        for (uint32_t w = 9; w < REC_WORDS; ++w) w_rec[s * REC_WORDS + w] = 0;

    vis_tile::SeqMap sm;
    sm.init(num_tiles, num_cores);
    uint32_t c = 0, k = 0;
    if (qn) sm.locate(q0, &c, &k);
    const uint32_t n_read = check ? N_IN : N_IN - 2;
    for (uint32_t i = 0; i < qn; i++) {
        const uint32_t t = c + k * num_cores;
        if (++k == sm.count(c)) {
            c++;
            k = 0;
        }
        noc_async_read(get_noc_addr(t, acc_mask), l1_mask, MASK_BYTES);
        noc_async_read_barrier();
        uint32_t mbits[vis_tile::MASK_WORDS];
        uint32_t any = 0;
        for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++) {
            mbits[w] = mk[w];
            any |= mbits[w];
        }
        if (any == 0) continue;
        noc_async_read_tile(t, i_m2x, l1_in[0]);
        noc_async_read_tile(t, i_m2y, l1_in[1]);
        noc_async_read_tile(t, i_dep, l1_in[2]);
        noc_async_read_tile(t, i_a, l1_in[3]);
        noc_async_read_tile(t, i_b, l1_in[4]);
        noc_async_read_tile(t, i_c, l1_in[5]);
        noc_async_read_tile(t, i_op, l1_in[6]);
        noc_async_read_tile(t, i_cr, l1_in[7]);
        noc_async_read_tile(t, i_cg, l1_in[8]);
        noc_async_read_tile(t, i_cb, l1_in[9]);
        noc_async_read_tile(t, i_tpg, l1_in[10]);
        noc_async_read_tile(t, i_aabb, l1_in[11]);
        if (n_read > 12) {
            noc_async_read_tile(t, i_rx, l1_in[12]);
            noc_async_read_tile(t, i_ry, l1_in[13]);
        }
#if EMIT_PUBOC
        if (pub01 != 0) {
            noc_async_read_tile(t, i_q01, l1_q01);
            noc_async_read_tile(t, i_q23, l1_q23);
        }
#endif
        noc_async_read_barrier();
#if EMIT_PUBOC
        if (pub01 == 0)  // NaN scene: pack the visible lanes here
            for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++)
                for (uint32_t bits = mbits[w]; bits != 0; bits &= bits - 1) {
                    const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                    q01[il] = sort_bin_fp32::to_unorm16(p_op[il]) |
                              (sort_bin_fp32::to_unorm16(p_cr[il]) << 16);
                    q23[il] = sort_bin_fp32::to_unorm16(p_cg[il]) |
                              (sort_bin_fp32::to_unorm16(p_cb[il]) << 16);
                }
#endif
        for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++)
            for (uint32_t bits = mbits[w]; bits != 0; bits &= bits - 1) {
                const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                const uint32_t mx = p_m2x[il], my = p_m2y[il];
                w_dep[slot] = p_dep[il];
                w_offs[slot] = pr;
                w_aabb[slot] = p_aabb[il] & vis_tile::PAYLOAD;
                pr += p_tpg[il] & vis_tile::PAYLOAD;
                volatile uint32_t* r = w_rec + slot * REC_WORDS;
                r[0] = p_a[il];
                r[1] = p_b[il];
                r[2] = p_c[il];
                r[3] = mx;
                r[4] = my;
                r[5] = p_op[il];
                r[6] = p_cr[il];
                r[7] = p_cg[il];
                r[8] = p_cb[il];
#if EMIT_PUBOC
                // Task #100: publish the emit's per-gaussian pack (op/color
                // UNORM16 at words 10, 11, as sort_bin did) and the depth key
                // (word 12) so the emit only copies them. Task #122: the packs
                // come per scene from the host (or the NaN pre-pass above).
                r[10] = q01[il];
                r[11] = q23[il];
                r[12] = p_dep[il];
#endif
                if (check) {
                    w_px[slot] = mx;
                    w_py[slot] = my;
                    w_rx[slot] = p_rx[il];
                    w_ry[slot] = p_ry[il];
                }
                slot++;
                g++;
                if (slot == PAGE_ELEMS) {
                    flush_page(cur_page, flush_lo, PAGE_ELEMS);
                    slot = 0;
                    flush_lo = 0;
                    cur_page++;
                }
            }
    }

    // Tail: the slot holding compact element M - 1 pads its last page (depth,
    // aabb and the record with 0, offs with P) so [M, M_pad) is defined.
    uint32_t hi = slot;
    if (is_last && slot != 0) {
        for (uint32_t s = slot; s < PAGE_ELEMS; s++) {
            w_dep[s] = 0;
            w_offs[s] = pr;
            w_aabb[s] = 0;
            w_px[s] = 0;
            w_py[s] = 0;
            w_rx[s] = 0;
            w_ry[s] = 0;
            for (uint32_t w = 0; w < REC_WORDS; ++w) w_rec[s * REC_WORDS + w] = 0;
        }
        hi = PAGE_ELEMS;
    }
    if (hi > flush_lo) flush_page(cur_page, flush_lo, hi);
    noc_async_write_barrier();
}
