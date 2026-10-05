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

// Task #170 fold: page range [pages * acc_lo / tot, pages * acc_hi / tot) of
// a mover whose running speed sums before and after it are acc_lo and acc_hi
// (sum of all = tot). Equals sort_split::speed_bounds (task #174), so the K2
// counts exactly the one-launch sort's emit range of that mover.
inline void k2_range_speed(uint32_t P_pub, uint32_t acc_lo, uint32_t acc_hi, uint32_t tot,
                           uint32_t* start, uint32_t* count) {
    const uint64_t pages = (P_pub + PAGE_WORDS - 1) / PAGE_WORDS;
    const uint32_t lo = static_cast<uint32_t>(pages * acc_lo / tot);
    const uint32_t hi = static_cast<uint32_t>(pages * acc_hi / tot);
    *start = lo;
    *count = hi - lo;
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

// Task #170 (pair stage diet; GSPLAT_TT_K2_DIET=0 is the kill switch): the pair
// pages [pg0, pg0 + npg) of emit_pairs (same gid / tid / padding), without one
// blocking 64 B read per lofs / box page, one divide per pair and one flushed
// write per pair page:
//   - the lofs and box pages stream in storage order (the segments' pages,
//     empty segments skipped) through a ring of 2 x RA_HALF slots; entering a
//     half waits for its reads and refills the half just left;
//   - pair pages stage in 2 x OUT_HALF slots and are written when full; only
//     entering a half waits for the staged writes to leave it;
//   - the rectangle is walked row by row (one divide, at the start gaussian);
//   - COUNT: cnt[tid]++ for every pair p < P_pub of the range (the caller
//     zeroes cnt): the sort's per-tile count of these pages.
// The start page comes from a 16-way search over the start segment's lofs
// pages (about 4 read round trips, not log2(m) dependent ones).
//
// Io (device: tile_assign_scatter_seg.cpp; host model: test_pfwc_fuse.cpp):
//   issue_lofs(page, slot) / issue(page, slot)   lofs (+ box) page -> ring slot
//   wait_reads()                                 every issued read landed
//   lofs_slot(slot), box_slot(slot)              const uint32_t*, 16 words
//   gid_slot(o), tid_slot(o)                     uint32_t*, staging page o
//   write_page(page, o)                          staging page o -> pair page
//   writes_flushed()                             issued writes left the staging
constexpr uint32_t RA_HALF = 8;
constexpr uint32_t RA_SLOTS = 2 * RA_HALF;
constexpr uint32_t OUT_HALF = 8;
constexpr uint32_t OUT_SLOTS = 2 * OUT_HALF;

// Start of the stream for pair p (< P_pub): the last segment with gaussians
// whose pair base is <= p, and the last page (segment-local) of that segment
// whose first lofs is <= p - pair base (lofs[sb] == 0 <= key, lofs is
// non-decreasing). Each round reads up to RA_SLOTS probe pages of (lo, hi] at
// once (io.issue_lofs into ring slots 0.., then io.wait_reads).
template <class Io>
inline void diet_start(const volatile uint32_t* tab, uint32_t nseg, uint32_t p, Io& io,
                       uint32_t* c_out, uint32_t* lo_out) {
    uint32_t c = 0;
    for (uint32_t cc = 0; cc < nseg; cc++) {
        const volatile uint32_t* t = tab + cc * PAGE_WORDS;
        if (t[T_M] != 0 && t[T_PB] <= p) c = cc;
    }
    const volatile uint32_t* t = tab + c * PAGE_WORDS;
    const uint32_t m = t[T_M];
    const uint32_t key = p - t[T_PB];
    const uint32_t sp = t[T_SB] / PAGE_WORDS;
    uint32_t lo = 0, hi = (m - 1) / PAGE_WORDS;
    while (lo < hi) {
        const uint32_t d = hi - lo;
        const uint32_t n = d < RA_SLOTS ? d : RA_SLOTS;
        for (uint32_t i = 0; i < n; i++) io.issue_lofs(sp + lo + 1 + (i * d) / n, i);
        io.wait_reads();
        uint32_t i = 0;
        while (i < n && io.lofs_slot(i)[0] <= key) i++;
        // Probes [0, i) are <= key: the page is in [q(i-1), q(i) - 1].
        const uint32_t nlo = (i == 0) ? lo : lo + 1 + ((i - 1) * d) / n;
        if (i < n) hi = lo + (i * d) / n;  // q(i) - 1
        lo = nlo;
    }
    *c_out = c;
    *lo_out = lo;
}

// emit_pairs_diet from a known stream start (c0, lo0) = diet_start(pg0 * 16)
// (unused when pg0 * 16 >= P_pub: padding only).
template <bool COUNT, class Io>
inline void emit_pairs_diet_from(const volatile uint32_t* tab, uint32_t nseg, uint32_t P_pub,
                                 uint32_t tiles_x, uint32_t pg0, uint32_t npg, uint32_t c0,
                                 uint32_t lo0, Io& io, uint32_t* cnt) {
    if (npg == 0) return;
    const uint32_t p_end = (pg0 + npg) * PAGE_WORDS;
    const uint32_t p_real = p_end < P_pub ? p_end : P_pub;
    uint32_t p = pg0 * PAGE_WORDS;
    uint32_t out_page = pg0, oslot = 0, oi = 0;
    uint32_t* gp = io.gid_slot(0);
    uint32_t* tp = io.tid_slot(0);
    auto next_out = [&]() {
        io.write_page(out_page++, oslot);
        oi = 0;
        if (++oslot == OUT_SLOTS) oslot = 0;
        if ((oslot & (OUT_HALF - 1)) == 0) io.writes_flushed();
        gp = io.gid_slot(oslot);
        tp = io.tid_slot(oslot);
    };
    if (p < p_real) {
        uint32_t c = c0;
        uint32_t m = 0, pc = 0, pb = 0, sb = 0;
        auto set_seg = [&](uint32_t cc) {
            const volatile uint32_t* t = tab + cc * PAGE_WORDS;
            c = cc;
            m = t[T_M];
            pc = t[T_P];
            pb = t[T_PB];
            sb = t[T_SB];
        };
        set_seg(c);
        const uint32_t key = p - pb;
        const uint32_t sp = sb / PAGE_WORDS;
        const uint32_t lo = lo0;
        // Read-ahead stream: pages [iq, iq_end] of segment ic, then the pages
        // of the next segments with gaussians.
        uint32_t ic = c, iq = sp + lo, iq_end = sp + (m - 1) / PAGE_WORDS;
        bool imore = true;
        auto refill = [&](uint32_t half) {
            for (uint32_t k = 0; k < RA_HALF && imore; k++) {
                io.issue(iq, half * RA_HALF + k);
                if (iq != iq_end) {
                    iq++;
                    continue;
                }
                do {
                    if (++ic == nseg) {
                        imore = false;
                        break;
                    }
                } while (tab[ic * PAGE_WORDS + T_M] == 0);
                if (imore) {
                    const uint32_t s0 = tab[ic * PAGE_WORDS + T_SB];
                    iq = s0 / PAGE_WORDS;
                    iq_end = (s0 + tab[ic * PAGE_WORDS + T_M] - 1) / PAGE_WORDS;
                }
            }
        };
        refill(0);
        io.wait_reads();
        refill(1);
        uint32_t slot = 0;
        const uint32_t* lp = io.lofs_slot(0);
        const uint32_t* bp = io.box_slot(0);
        auto advance = [&]() {
            if (++slot == RA_SLOTS) slot = 0;
            if ((slot & (RA_HALF - 1)) == 0) {
                io.wait_reads();
                refill((slot / RA_HALF) ^ 1u);  // the half just left
            }
            lp = io.lofs_slot(slot);
            bp = io.box_slot(slot);
        };
        // Start gaussian: the last word of the start page with lofs <= key.
        uint32_t j = lo * PAGE_WORDS, word = 0;
        {
            const uint32_t wn = (m - j < PAGE_WORDS) ? m - j : PAGE_WORDS;
            while (word + 1 < wn && lp[word + 1] <= key) word++;
            j += word;
        }
        uint32_t off0 = pb + lp[word];
        for (;;) {
            // Gaussian (c, j) sits at (slot, word); its box is read before
            // the stream may move past its page.
            const uint32_t box = bp[word];
            uint32_t off1;
            if (j + 1 < m) {
                if (++word == PAGE_WORDS) {
                    word = 0;
                    advance();
                }
                off1 = pb + lp[word];
            } else {
                off1 = pb + pc;
            }
            if (off1 > p) {
                const uint32_t e = off1 < p_real ? off1 : p_real;
                const uint32_t w = vis_tile::aabb_w(box);
                uint32_t dx = p - off0, dy = 0;  // non-zero at the start gaussian only
                if (dx >= w) {
                    dy = dx / w;
                    dx -= dy * w;
                }
                uint32_t row = (vis_tile::aabb_min_y(box) + dy) * tiles_x + vis_tile::aabb_min_x(box);
                const uint32_t gid = sb + j;
                uint32_t n = e - p;
                p = e;
                while (n != 0) {
                    uint32_t run = w - dx;
                    if (run > n) run = n;
                    n -= run;
                    uint32_t t = row + dx;
                    do {
                        gp[oi] = gid;
                        tp[oi] = t;
                        if constexpr (COUNT) cnt[t]++;
                        t++;
                        if (++oi == PAGE_WORDS) next_out();
                    } while (--run != 0);
                    dx = 0;
                    row += tiles_x;
                }
                if (p >= p_real) break;
            }
            off0 = off1;
            if (++j == m) {
                // The next segment with gaussians: its first page is the
                // stream's next page. p < P_pub, so one exists (the bound only
                // keeps a bad table from hanging the core).
                uint32_t cn = c + 1;
                while (cn < nseg && tab[cn * PAGE_WORDS + T_M] == 0) cn++;
                if (cn >= nseg) break;
                set_seg(cn);
                j = 0;
                word = 0;
                advance();
                off0 = pb + lp[0];
            }
        }
        io.wait_reads();  // read-ahead past the range end lands before exit
    }
    // Padding (gid = tid = 0) to the end of the last page.
    while (p < p_end) {
        gp[oi] = 0;
        tp[oi] = 0;
        p++;
        if (++oi == PAGE_WORDS) next_out();
    }
}

template <bool COUNT, class Io>
inline void emit_pairs_diet(const volatile uint32_t* tab, uint32_t nseg, uint32_t P_pub,
                            uint32_t tiles_x, uint32_t pg0, uint32_t npg, Io& io, uint32_t* cnt) {
    if (npg == 0) return;
    const uint32_t p = pg0 * PAGE_WORDS;
    uint32_t c = 0, lo = 0;
    if (p < P_pub) diet_start(tab, nseg, p, io, &c, &lo);
    emit_pairs_diet_from<COUNT>(tab, nseg, P_pub, tiles_x, pg0, npg, c, lo, io, cnt);
}

// Task #274 (GSPLAT_TT_K2_TRISC=1): the core's TRISCs run emit_pairs_diet_from
// over L1 windows the mover read for them (no NoC on a TRISC). A mover range
// [pg0, pg0 + npg) splits into job 0 (n0 pages), job 1 (n1 pages), then the
// mover's own pages; pm0 / pm1 are permille of npg, each capped at cap pages.
inline void k2_jobs(uint32_t npg, uint32_t pm0, uint32_t pm1, uint32_t cap, uint32_t* n0,
                    uint32_t* n1) {
    uint32_t a = npg * pm0 / 1000u, b = npg * pm1 / 1000u;
    if (a > cap) a = cap;
    if (b > cap) b = cap;
    if (a + b > npg) b = npg - a;
    *n0 = a;
    *n1 = b;
}

// The stream pages emit_pairs_diet_from reads from (c, lo) until the part's
// end: through page lo_n + 1 (segment-local, clipped to the segment) of
// segment c_n, the start (diet_start) of the next part, or with to_end to the
// end of the stream. visit(k, storage page) for the k-th; returns the count,
// or cap + 1 as soon as it would pass cap (nothing visited past cap).
template <class Visit>
inline uint32_t diet_window(const volatile uint32_t* tab, uint32_t nseg, uint32_t c, uint32_t lo,
                            bool to_end, uint32_t c_n, uint32_t lo_n, uint32_t cap,
                            Visit&& visit) {
    uint32_t ic = c;
    uint32_t sb = tab[ic * PAGE_WORDS + T_SB];
    uint32_t iq = sb / PAGE_WORDS + lo;
    uint32_t iq_end = (sb + tab[ic * PAGE_WORDS + T_M] - 1) / PAGE_WORDS;
    uint32_t n = 0;
    for (;;) {
        if (n == cap) return cap + 1u;
        visit(n++, iq);
        if (!to_end && ic == c_n) {
            const uint32_t sp = sb / PAGE_WORDS;
            const uint32_t stop = sp + lo_n + 1u < iq_end ? sp + lo_n + 1u : iq_end;
            if (iq >= stop) return n;
        }
        if (iq != iq_end) {
            iq++;
            continue;
        }
        do {
            if (++ic == nseg) return n;
        } while (tab[ic * PAGE_WORDS + T_M] == 0);
        if (!to_end && ic > c_n) return n;
        sb = tab[ic * PAGE_WORDS + T_SB];
        iq = sb / PAGE_WORDS;
        iq_end = (sb + tab[ic * PAGE_WORDS + T_M] - 1) / PAGE_WORDS;
    }
}

// emit_pairs_diet_from's Io over a diet_window in L1: the k-th issued stream
// page is window page k (past the window: the poison page, never consumed);
// pair pages go to consecutive pages of gid / tid. No reads to wait for, no
// writes to flush.
struct WinIo {
    const uint32_t* lofs;
    const uint32_t* box;
    const uint32_t* poison;
    uint32_t n_in;
    uint32_t* gid;
    uint32_t* tid;
    uint32_t k = 0, out_k = 0;
    const uint32_t* sl[RA_SLOTS] = {};
    const uint32_t* sb[RA_SLOTS] = {};
    // Task #291: window pages landed so far (the mover fills the window in
    // chunks while the TRISC runs); nullptr = the whole window is in place.
    const volatile uint32_t* fill = nullptr;
    void issue_lofs(uint32_t, uint32_t) {}
    void issue(uint32_t, uint32_t slot) {
        sl[slot] = k < n_in ? lofs + k * PAGE_WORDS : poison;
        sb[slot] = k < n_in ? box + k * PAGE_WORDS : poison;
        k++;
    }
    void wait_reads() {
        if (fill == nullptr) return;
        const uint32_t need = k < n_in ? k : n_in;
        while (*fill < need) {
#if defined(COMPILE_FOR_TRISC)
            asm volatile("fence" ::: "memory");
#endif
        }
#if defined(COMPILE_FOR_TRISC)
        asm volatile("fence" ::: "memory");
#endif
    }
    const uint32_t* lofs_slot(uint32_t s) const { return sl[s]; }
    const uint32_t* box_slot(uint32_t s) const { return sb[s]; }
    uint32_t* gid_slot(uint32_t) { return gid + out_k * PAGE_WORDS; }
    uint32_t* tid_slot(uint32_t) { return tid + out_k * PAGE_WORDS; }
    void write_page(uint32_t, uint32_t) { out_k++; }
    void writes_flushed() {}
};

}  // namespace pfwc_fuse
