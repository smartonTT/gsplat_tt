// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc WRITER for lever B (task #125, GSPLAT_TT_PFWC_FUSE=1), BRISC / NoC0.
// writer_pfwc_vis.cpp with the gather compaction fused in. Tiles are dealt
// strided (tile chunk_start + k * stride), so the core's visible gaussians, in
// tile then lane order, are exactly the legacy compaction sequence of core c
// (vis_tile::SeqMap). Per 1024-gaussian tile it
//   * classifies the tile (vis_tile::classify_tile: RECHECK words, mask, counts),
//   * writes every visible gaussian straight to the compact streams at storage
//     index seg_base + j (pfwc_fuse.h): proj_m_depth, the 64 B blend record
//     (gather_vis_scatter.cpp layout), proj_m_offs (SEGMENT-LOCAL exclusive pair
//     offset) and proj_m_aabb,
// and at the end writes its counts page [visible, pairs] for the segment K2
// (tile_assign_scatter_seg.cpp). None of the 10 pfwc tiles, the mask or the
// per-tile counts go to DRAM, and the gather scan / scatter do not run.
//
// RUNTIME ARGS
//   0..3  : scene opacity, col_r, col_g, col_b (tile pages)
//   4..7  : outputs proj_m_depth, proj_m_blendrec (64 B record per page),
//           proj_m_offs, proj_m_aabb (64 B pages)
//   8     : counts table (64 B page per pfwc core)
//   9: chunk_start (= c)   10: num_chunks   11: stride (= num_cores)   12: N
//   13..17: k_near, min_opacity, image width, image height, max_radius (fp32 bits)
//   18: tiles_x   19: tiles_y   20: tile_size (power of two)
//   21: seg_base   22: core index c (counts page)
//   23, 24: scene_puboc01 / scene_puboc23 (tile pages; EMIT_PUBOC): record
//       words 10 / 11 precomputed per scene (task #122). 0 = the scene has a
//       NaN: the writer packs the visible lanes itself (to_unorm16).
//   25..: PFWC_TILE_LIST only (task #169 chunk cull): the tile ids (padded)
//
// COMPILE-TIME ARGS: 9 TensorAccessorArgs in runtime-arg order 0..8 (the
// puboc tiles reuse the opacity accessor's, all are DRAM interleaved).
// FUSE_ABL (define, GSPLAT_TT_FUSE_ABL; targeted profiling only, writes wrong
// records): 1 = skip the record NoC writes, 2 = skip the record field copies,
// 4 = skip the opacity / color / puboc tile reads. Compile-time because
// run-time bits in the copy loop cost 0.4-0.6 ms (task #122).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwc_fuse.h"
#include "sort_bin_fp32.h"
#include "vis_tile.h"

#ifndef FUSE_ABL
#define FUSE_ABL 0
#endif

void kernel_main() {
    uint32_t addr[9];
    for (uint32_t k = 0; k < 9; k++) addr[k] = get_arg_val<uint32_t>(k);
    const uint32_t chunk_start = get_arg_val<uint32_t>(9);
    const uint32_t num_chunks = get_arg_val<uint32_t>(10);
    const uint32_t stride = get_arg_val<uint32_t>(11);
    const uint32_t N = get_arg_val<uint32_t>(12);
    vis_tile::Params prm;
    prm.k_near = get_arg_val<uint32_t>(13);
    prm.min_opacity = get_arg_val<uint32_t>(14);
    prm.img_w = get_arg_val<uint32_t>(15);
    prm.img_h = get_arg_val<uint32_t>(16);
    prm.max_radius = get_arg_val<uint32_t>(17);
    prm.tiles_x = get_arg_val<uint32_t>(18);
    prm.tiles_y = get_arg_val<uint32_t>(19);
    const uint32_t tile_size = get_arg_val<uint32_t>(20);
    prm.tile_shift = dm_fp32::pow2_shift(tile_size);
    prm.inv_tile = 1.0f / static_cast<float>(tile_size);
    const uint32_t seg_base = get_arg_val<uint32_t>(21);
    const uint32_t core = get_arg_val<uint32_t>(22);
    const uint32_t pub01 = get_arg_val<uint32_t>(23);
    const uint32_t pub23 = get_arg_val<uint32_t>(24);
    constexpr uint32_t ABL = FUSE_ABL;

    constexpr uint32_t CB_M2X = 9, CB_M2Y = 10, CB_DEP = 11, CB_A = 12, CB_B = 13, CB_C = 14,
                       CB_RX = 15, CB_RY = 16, CB_TPG = 35, CB_AABB = 36;
    constexpr uint32_t CB_MASK = 37;  // 128 B mask (L1 only)
    constexpr uint32_t CB_OPT = 39;   // opacity tile
    constexpr uint32_t CB_FUSE = 40;  // color tiles + record / page staging
    constexpr uint32_t IN_CB[10] = {CB_M2X, CB_M2Y, CB_DEP, CB_A,  CB_B,
                                    CB_C,   CB_RX,  CB_RY,  CB_TPG, CB_AABB};
    constexpr uint32_t PW = pfwc_fuse::PAGE_WORDS;
    constexpr uint32_t PB = PW * 4;

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

    const auto i_op = TensorAccessor(a0, addr[0], tile_bytes);
    const auto i_cr = TensorAccessor(a1, addr[1], tile_bytes);
    const auto i_cg = TensorAccessor(a2, addr[2], tile_bytes);
    const auto i_cb = TensorAccessor(a3, addr[3], tile_bytes);
    const auto i_q01 = TensorAccessor(a0, pub01, tile_bytes);
    const auto i_q23 = TensorAccessor(a0, pub23, tile_bytes);
    const auto o_dep = TensorAccessor(a4, addr[4], PB);
    const auto o_rec = TensorAccessor(a5, addr[5], PB);
    const auto o_offs = TensorAccessor(a6, addr[6], PB);
    const auto o_aabb = TensorAccessor(a7, addr[7], PB);
    const auto o_cnt = TensorAccessor(a8, addr[8], PB);

    const uint32_t l1_mask = get_write_ptr(CB_MASK);
    const uint32_t l1_op = get_write_ptr(CB_OPT);
    const uint32_t l1_f = (get_write_ptr(CB_FUSE) + 63u) & ~63u;
    const uint32_t l1_cr = l1_f, l1_cg = l1_cr + tile_bytes, l1_cb = l1_cg + tile_bytes;
    const uint32_t l1_q01 = l1_cb + tile_bytes, l1_q23 = l1_q01 + tile_bytes;
    // Record staging: RL records per DRAM bank, NB_MAX banks (see flush_rec).
    constexpr uint32_t RL = 16, NB_MAX = 8;
    const uint32_t l1_rec = l1_q23 + tile_bytes;
    const uint32_t l1_dep = l1_rec + NB_MAX * RL * PB;  // one page each
    const uint32_t l1_offs = l1_dep + PB;
    const uint32_t l1_aabb = l1_offs + PB;
    const uint32_t l1_cnt = l1_aabb + PB;

    auto mw = reinterpret_cast<volatile uint32_t*>(l1_mask);
    auto opw = reinterpret_cast<volatile uint32_t*>(l1_op);
    auto p_cr = reinterpret_cast<volatile uint32_t*>(l1_cr);
    auto p_cg = reinterpret_cast<volatile uint32_t*>(l1_cg);
    auto p_cb = reinterpret_cast<volatile uint32_t*>(l1_cb);
    [[maybe_unused]] auto q01 = reinterpret_cast<volatile uint32_t*>(l1_q01);
    [[maybe_unused]] auto q23 = reinterpret_cast<volatile uint32_t*>(l1_q23);
    auto w_rec = reinterpret_cast<volatile uint32_t*>(l1_rec);
    auto w_dep = reinterpret_cast<volatile uint32_t*>(l1_dep);
    auto w_offs = reinterpret_cast<volatile uint32_t*>(l1_offs);
    auto w_aabb = reinterpret_cast<volatile uint32_t*>(l1_aabb);
    auto w_cnt = reinterpret_cast<volatile uint32_t*>(l1_cnt);

    // Record words 9..15 are zero (gather_vis_scatter.cpp); EMIT_PUBOC sets 10..12.
    for (uint32_t s = 0; s < NB_MAX * RL; ++s)
        for (uint32_t w = 9; w < PW; ++w) w_rec[s * PW + w] = 0;

    // Records are one 64 B page each and page g sits on DRAM bank g % nb at
    // offset g / nb, so pages g, g + nb, ... are contiguous within a bank. The
    // records are staged bank-major in groups of RL * nb pages (group base G0 a
    // multiple of RL * nb) and each group goes out as one write per bank of up
    // to RL pages instead of one write per record (task #122). nb is read off
    // the accessor; if that fails, every record is written on its own.
    uint32_t nb = 1;
    {
        const uint64_t a = get_noc_addr(seg_base, o_rec);
        while (nb <= NB_MAX && get_noc_addr(seg_base + nb, o_rec) != a + PB) nb++;
    }
    const bool per_page = nb > NB_MAX;
    if (per_page) nb = NB_MAX;  // staging layout only
    const uint32_t GS = RL * nb;
    uint32_t G0 = seg_base / GS * GS;  // current group
    uint32_t gs = seg_base;            // first staged record of the group
    uint32_t rb = (seg_base - G0) % nb, rl = (seg_base - G0) / nb;
    // Stage records [gs, ge) of group G0.
    auto flush_rec = [&](uint32_t ge) {
        if (!(ABL & 1u)) {
            const uint32_t d = gs - G0, e = ge - G0;
            for (uint32_t b = 0; b < nb; ++b) {
                uint32_t lo = 0, hi = RL;
                if (d != 0 || e != GS) {
                    lo = d > b ? (d - b + nb - 1) / nb : 0;
                    hi = e > b ? (e - b + nb - 1) / nb : 0;
                }
                if (hi <= lo) continue;
                if (per_page) {
                    for (uint32_t l = lo; l < hi; ++l)
                        noc_async_write(l1_rec + (b * RL + l) * PB,
                                        get_noc_addr(G0 + b + l * nb, o_rec), PB);
                } else {
                    noc_async_write(l1_rec + (b * RL + lo) * PB,
                                    get_noc_addr(G0 + b + lo * nb, o_rec), (hi - lo) * PB);
                }
            }
        }
        noc_async_writes_flushed();  // staging reusable; completion at the end
    };

    uint32_t page = seg_base / PW;  // seg_base is 1024-aligned
    uint32_t slot = 0, m = 0, pr = 0;
    auto flush = [&]() {
        noc_async_write(l1_dep, get_noc_addr(page, o_dep), PB);
        noc_async_write(l1_offs, get_noc_addr(page, o_offs), PB);
        noc_async_write(l1_aabb, get_noc_addr(page, o_aabb), PB);
        noc_async_writes_flushed();  // staging reusable; completion at the end
    };

    for (uint32_t k = 0; k < num_chunks; k++) {
#ifdef PFWC_TILE_LIST
        const uint32_t t = get_arg_val<uint32_t>(25 + k);  // task #169 survivors
#else
        const uint32_t t = chunk_start + k * stride;
#endif
        // The opacity / color tiles stream in while the SFPU tile lands.
        if (!(ABL & 4u)) {
            noc_async_read_tile(t, i_op, l1_op);
            noc_async_read_tile(t, i_cr, l1_cr);
            noc_async_read_tile(t, i_cg, l1_cg);
            noc_async_read_tile(t, i_cb, l1_cb);
#if EMIT_PUBOC
            if (pub01 != 0) {
                noc_async_read_tile(t, i_q01, l1_q01);
                noc_async_read_tile(t, i_q23, l1_q23);
            }
#endif
        }
        for (uint32_t o = 0; o < 10; o++) cb_wait_front(IN_CB[o], 1);
        noc_async_read_barrier();

        auto p_m2x = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2X));
        auto p_m2y = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2Y));
        auto p_dep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_DEP));
        auto p_a = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_A));
        auto p_b = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_B));
        auto p_c = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_C));
        auto p_rx = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RX));
        auto p_ry = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RY));
        auto p_tpg = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TPG));
        auto p_aabb = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_AABB));

        const uint32_t tbase = t * vis_tile::TILE_ELEMS;
        const uint32_t n_el = (tbase >= N) ? 0u
            : (N - tbase < vis_tile::TILE_ELEMS ? N - tbase : vis_tile::TILE_ELEMS);
        auto get_inputs = [&](uint32_t il, uint32_t& tz, uint32_t& op, uint32_t& mx,
                              uint32_t& my, uint32_t& rx, uint32_t& ry) {
            tz = p_dep[il];
            op = opw[il];
            mx = p_m2x[il];
            my = p_m2y[il];
            rx = p_rx[il];
            ry = p_ry[il];
        };
        uint32_t vc = 0, pc = 0;
        vis_tile::classify_tile(p_tpg, p_aabb, n_el, mw, prm, get_inputs, &vc, &pc);
#if EMIT_PUBOC
        if (pub01 == 0)  // NaN scene: pack the visible lanes here
            for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++)
                for (uint32_t bits = mw[w]; bits != 0; bits &= bits - 1) {
                    const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                    q01[il] = sort_bin_fp32::to_unorm16(opw[il]) |
                              (sort_bin_fp32::to_unorm16(p_cr[il]) << 16);
                    q23[il] = sort_bin_fp32::to_unorm16(p_cg[il]) |
                              (sort_bin_fp32::to_unorm16(p_cb[il]) << 16);
                }
#endif

        for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++)
            for (uint32_t bits = mw[w]; bits != 0; bits &= bits - 1) {
                const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                // Loads grouped ahead of their stores so the L1 load latency
                // overlaps (a load-store pair per word stalls on every word).
                const uint32_t dep = p_dep[il], aabb = p_aabb[il], tpg = p_tpg[il];
                w_dep[slot] = dep;
                w_offs[slot] = pr;
                w_aabb[slot] = aabb & vis_tile::PAYLOAD;
                pr += tpg & vis_tile::PAYLOAD;
                volatile uint32_t* r = w_rec + (rb * RL + rl) * PW;
                if (!(ABL & 2u)) {
                    {
                        const uint32_t a = p_a[il], b = p_b[il], cc = p_c[il];
                        const uint32_t mx = p_m2x[il], my = p_m2y[il];
                        r[0] = a;
                        r[1] = b;
                        r[2] = cc;
                        r[3] = mx;
                        r[4] = my;
                    }
                    {
                        const uint32_t op = opw[il], cr = p_cr[il], cg = p_cg[il], cb = p_cb[il];
#if EMIT_PUBOC
                        const uint32_t u01 = q01[il], u23 = q23[il];
#endif
                        r[5] = op;
                        r[6] = cr;
                        r[7] = cg;
                        r[8] = cb;
#if EMIT_PUBOC
                        r[10] = u01;
                        r[11] = u23;
                        r[12] = dep;
#endif
                    }
                }
                m++;
                if (++rb == nb) {
                    rb = 0;
                    if (++rl == RL) {
                        flush_rec(G0 + GS);
                        G0 += GS;
                        gs = G0;
                        rl = 0;
                    }
                }
                if (++slot == PW) {
                    flush();
                    slot = 0;
                    page++;
                }
            }

        for (uint32_t o = 0; o < 10; o++) cb_pop_front(IN_CB[o], 1);
    }

    // Partial last page: depth / aabb 0 and offs = segment pairs past the last
    // gaussian, like the legacy tail (records past m are never read).
    if (slot != 0) {
        for (uint32_t s = slot; s < PW; s++) {
            w_dep[s] = 0;
            w_offs[s] = pr;
            w_aabb[s] = 0;
        }
        flush();
    }
    // Partial last record group.
    if (G0 + rl * nb + rb > gs) flush_rec(G0 + rl * nb + rb);
    for (uint32_t w = 0; w < PW; w++) w_cnt[w] = 0;
    w_cnt[pfwc_fuse::T_M] = m;
    w_cnt[pfwc_fuse::T_P] = pr;
    noc_async_write(l1_cnt, get_noc_addr(core, o_cnt), PB);
    noc_async_write_barrier();
}
