// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Lever B (task #125, GSPLAT_TT_PFWC_FUSE=1): visible-gaussian compaction fused
// into the pfwc writer, and the TA pair scatter run from per-core segments.
// Shared by writer_pfwc_fuse.cpp, tile_assign_scatter_seg.cpp and the host
// model test (tests/unit/test_pfwc_fuse.cpp). Header-only, no device API.
//
// Layout. pfwc deals tiles strided over C cores (core c owns tiles c, c + C,
// ...), the order of the legacy compaction (vis_tile::SeqMap). Core c writes
// its visible gaussians, in that order, to its own SEGMENT of the compact
// streams starting at storage index seg_base(c) = SeqMap.first(c) * 1024 (a
// core can not have more visible gaussians than its tiles hold). Segments are
// 1024-aligned, so no 64 B page is shared between cores and no scan is needed
// before the write. The storage index s is the gaussian id downstream (sort
// reads proj_m_depth / blendrec by gid, both padded_n long); the map from the
// legacy dense id g to s is monotone, so the pair order and the image are the
// legacy ones. proj_m_offs holds the SEGMENT-LOCAL exclusive pair offset; the
// segment K2 adds the segment's pair base.
//
// Counts table: one 64 B page per pfwc core, [visible m_c, pairs p_c, 0...].
#pragma once

#include <cstdint>

#include "vis_tile.h"

namespace pfwc_fuse {

constexpr uint32_t PAGE_WORDS = 16;  // 64 B
constexpr uint32_t T_M = 0;          // visible gaussians of the segment
constexpr uint32_t T_P = 1;          // pairs of the segment
constexpr uint32_t T_PB = 2;         // exclusive pair base of the segment
constexpr uint32_t T_SB = 3;         // storage base of the segment
constexpr uint32_t MAX_SEG = 128;    // table capacity (pfwc cores)

inline uint32_t seg_base(uint32_t num_tiles, uint32_t num_cores, uint32_t c) {
    vis_tile::SeqMap sm;
    sm.init(num_tiles, num_cores);
    return sm.first(c) * vis_tile::TILE_ELEMS;
}

// tab: nseg pages with T_M / T_P filled from the counts table. Fills T_PB and
// T_SB and returns the totals.
inline void seg_table(volatile uint32_t* tab, uint32_t nseg, uint32_t num_tiles, uint32_t* M,
                      uint32_t* P) {
    vis_tile::SeqMap sm;
    sm.init(num_tiles, nseg);
    uint32_t m = 0, p = 0;
    for (uint32_t c = 0; c < nseg; c++) {
        volatile uint32_t* t = tab + c * PAGE_WORDS;
        t[T_PB] = p;
        t[T_SB] = sm.first(c) * vis_tile::TILE_ELEMS;
        m += t[T_M];
        p += t[T_P];
    }
    *M = m;
    *P = p;
}

// Page range [*start, *start + *count) of mover `mover` on K2 core k: the
// (P_pub + 15) / 16 pages split contiguously over the cores (tile_assign
// split_pages), then a core's range split BRISC (mover 0) first by permille
// when dual (mover0_pages).
inline void k2_range(uint32_t P_pub, uint32_t ncores, uint32_t k, uint32_t mover, uint32_t dual,
                     uint32_t permille, uint32_t* start, uint32_t* count) {
    const uint32_t pages = (P_pub + PAGE_WORDS - 1) / PAGE_WORDS;
    const uint32_t base = pages / ncores, rem = pages % ncores;
    const uint32_t s = k * base + (k < rem ? k : rem);
    const uint32_t n = base + (k < rem ? 1u : 0u);
    const uint32_t n0 =
        dual ? static_cast<uint32_t>((static_cast<uint64_t>(n) * permille) / 1000u) : 0u;
    if (mover == 0) {
        *start = s;
        *count = n0;
    } else {
        *start = s + n0;
        *count = n - n0;
    }
}

// Pairs [p_start, p_end) of the gaussian-major pair list, legacy K2 order.
// read_lofs(s): segment-local exclusive offset of storage index s;
// read_box(s): packed rectangle (vis_tile::aabb_pack) of s;
// out(p, gid, tid) for every p (gid = tid = 0 for p >= P_pub).
template <class ReadLofs, class ReadBox, class Out>
inline void emit_pairs(const volatile uint32_t* tab, uint32_t nseg, uint32_t P_pub,
                       uint32_t tiles_x, uint32_t p_start, uint32_t p_end, ReadLofs&& read_lofs,
                       ReadBox&& read_box, Out&& out) {
    uint32_t c = 0, j = 0, m = 0, sb = 0, pb = 0, pc = 0;
    uint32_t off0 = 0, off1 = 0, minx = 0, miny = 0, w = 1;
    auto load = [&]() {
        const uint32_t s = sb + j;
        off0 = pb + read_lofs(s);
        off1 = pb + (j + 1 < m ? read_lofs(s + 1) : pc);
        const uint32_t box = read_box(s);
        minx = vis_tile::aabb_min_x(box);
        miny = vis_tile::aabb_min_y(box);
        w = vis_tile::aabb_w(box);
    };
    auto set_seg = [&](uint32_t cc) {
        const volatile uint32_t* t = tab + cc * PAGE_WORDS;
        c = cc;
        m = t[T_M];
        pc = t[T_P];
        pb = t[T_PB];
        sb = t[T_SB];
    };
    if (p_start < P_pub) {
        // Last segment with gaussians whose pair base is <= p_start.
        uint32_t cs = 0;
        for (uint32_t cc = 0; cc < nseg; cc++) {
            const volatile uint32_t* t = tab + cc * PAGE_WORDS;
            if (t[T_M] != 0 && t[T_PB] <= p_start) cs = cc;
        }
        set_seg(cs);
        // Largest j with lofs[sb + j] <= p_start - pb (lofs[sb] == 0).
        uint32_t lo = 0, hi = m - 1;
        const uint32_t key = p_start - pb;
        while (lo < hi) {
            const uint32_t mid = (lo + hi + 1) >> 1;
            if (read_lofs(sb + mid) <= key)
                lo = mid;
            else
                hi = mid - 1;
        }
        j = lo;
        load();
    }
    for (uint32_t p = p_start; p < p_end; p++) {
        uint32_t gid = 0, tid = 0;
        if (p < P_pub) {
            while (p >= off1) {
                if (++j == m) {
                    do set_seg(c + 1);
                    while (m == 0);
                    j = 0;
                }
                load();
            }
            const uint32_t local = p - off0;
            const uint32_t dy = local / w;
            const uint32_t dx = local - dy * w;
            gid = sb + j;
            tid = (miny + dy) * tiles_x + minx + dx;
        }
        out(p, gid, tid);
    }
}

}  // namespace pfwc_fuse
