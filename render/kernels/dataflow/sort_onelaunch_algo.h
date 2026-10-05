// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Plain-C++ pieces of the one-launch sort v2 (task #124, lever A), shared by
// the kernels and tests/unit/test_sort_onelaunch_v2.cpp:
//
//  - ring_*: sort_bin_onelaunch.cpp's per-tile record runs (OL_RING). A run
//    is R slots aligned to the bucket slot (c & (R-1)), so a full run is one
//    R*32 B write inside one 2 KB record page; a mover's first and last run of
//    a tile are clipped to its own cursors.
//  - select_*: sort_subchunk_materialize.cpp's big-tile items (OL_MAT_SELECT).
//    An item needs only the stable depth ranks [lo, hi) of its tile. Key
//    histograms over 1024 linear bins narrow the key ranges of ranks lo and
//    hi-1; only the keys from the first to the second range (in index order)
//    are radix-sorted. All
//    equal keys fall in it together, so they are a contiguous slice of the
//    tile's stable order starting at rank `base`, and ranks [lo, hi) are
//    sorted[lo - base, hi - base): the same ids as sorting the whole tile.
#pragma once

#include <cstdint>

#include "sort_radix_tile_algo.h"

namespace sort_ol {

// First slot of the run that ends at cursor `last` (inclusive), clipped to
// this mover's first cursor of the tile.
inline uint32_t ring_run_start(uint32_t first, uint32_t last, uint32_t R) {
    const uint32_t grp = last & ~(R - 1u);
    return grp > first ? grp : first;
}

// End of the emit: true if the tile's final run (cursors [first, end), slots
// past cap dropped) is partial and still in L1; *last = its last cursor. A
// full final run (last & (R-1) == R-1) was written in the loop.
inline bool ring_drain(uint32_t first, uint32_t end, uint32_t cap, uint32_t R, uint32_t* last) {
    const uint32_t en = end < cap ? end : cap;
    if (en <= first) return false;
    *last = en - 1u;
    return (*last & (R - 1u)) != R - 1u;
}

constexpr uint32_t SELECT_BINS = 1024;
constexpr uint32_t SELECT_LEVELS = 3;
static_assert(SELECT_BINS <= sort_radix_tile::HIST_ENTRIES, "bins live in the radix histogram");

// The candidates of ranks [lo, hi): every key in [klo, khi], `m` of them, the
// first of stable rank `base` (exactly `base` keys are below klo).
struct Select {
    uint32_t klo;
    uint32_t khi;
    uint32_t base;
    uint32_t m;
};

// A key range holding rank r: keys [klo, khi], `below` keys under klo, `cnt`
// keys inside (below <= r < below + cnt).
struct Edge {
    uint32_t klo, khi, below, cnt;
};

// Histogram the keys of e over 1024 linear bins and narrow lo_e / hi_e (both
// inside e's range) to the bins of their ranks.
inline void select_refine(const uint32_t* k, uint32_t n, const Edge e, uint32_t r_lo, uint32_t r_hi,
                          Edge* lo_e, Edge* hi_e, sort_radix_tile::hist_t* hist) {
    const uint32_t span = e.khi - e.klo;
    uint32_t shift = 0;
    while ((span >> shift) >= SELECT_BINS) shift++;
    for (uint32_t b = 0; b < SELECT_BINS; b++) hist[b] = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t d = k[i] - e.klo;  // unsigned: keys below klo wrap past span
        if (d <= span) hist[d >> shift]++;
    }
    bool want_lo = lo_e != nullptr, want_hi = hi_e != nullptr;
    uint32_t cum = e.below;
    for (uint32_t b = 0; b < SELECT_BINS && (want_lo || want_hi); b++) {
        const uint32_t c = hist[b];
        if (c == 0u) continue;
        // Bin keys [klo + (b << shift), + (1 << shift) - 1], clipped to khi.
        // (b + 1) << shift wraps only for the last bin at shift 22, where the
        // clip keeps khi.
        const uint32_t top = ((b + 1u) << shift) - 1u;
        const Edge bin{e.klo + (b << shift), (top < span) ? e.klo + top : e.khi, cum, c};
        if (want_lo && cum + c > r_lo) {
            *lo_e = bin;
            want_lo = false;
        }
        if (want_hi && cum + c > r_hi) {
            *hi_e = bin;
            want_hi = false;
        }
        cum += c;
    }
}

// Up to SELECT_LEVELS rounds narrowing the key ranges holding ranks lo and
// hi - 1 (one histogram pass while they share a range, else one each), until
// at most 2 (hi - lo) keys lie between them or both ranges are single keys.
// Separate edges matter when outliers or a gap sit between the two ranks.
// 0 <= lo < hi <= n <= MAX_N.
inline Select select_bins(const uint32_t* k, uint32_t n, uint32_t lo, uint32_t hi,
                          sort_radix_tile::hist_t* hist, uint32_t levels = SELECT_LEVELS) {
    Edge el{k[0], k[0], 0u, n};
    for (uint32_t i = 1; i < n; i++) {
        const uint32_t x = k[i];
        el.klo = x < el.klo ? x : el.klo;
        el.khi = x > el.khi ? x : el.khi;
    }
    Edge eh = el;
    auto result = [&]() { return Select{el.klo, eh.khi, el.below, eh.below + eh.cnt - el.below}; };
    for (uint32_t lvl = 0; lvl < levels; lvl++) {
        if (result().m <= 2u * (hi - lo)) break;
        const bool lo_done = el.klo == el.khi, hi_done = eh.klo == eh.khi;
        if (lo_done && hi_done) break;
        if (el.klo == eh.klo && el.khi == eh.khi) {
            const Edge e = el;
            select_refine(k, n, e, lo, hi - 1u, &el, &eh, hist);
        } else {
            if (!lo_done) select_refine(k, n, el, lo, 0u, &el, nullptr, hist);
            if (!hi_done) select_refine(k, n, eh, 0u, hi - 1u, nullptr, &eh, hist);
        }
    }
    return result();
}

// The candidates (key, index) in index order; returns their count (== s.m).
inline uint32_t select_collect(const uint32_t* k, uint32_t n, const Select& s, uint32_t* ck,
                               uint32_t* cv) {
    const uint32_t span = s.khi - s.klo;
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (k[i] - s.klo <= span) {  // unsigned: k[i] in [klo, khi]
            ck[m] = k[i];
            cv[m] = i;
            m++;
        }
    }
    return m;
}

// Indices of the stable ranks [lo, hi) of k[0..n) into out[0..hi-lo). ck, cv,
// ck2, cv2 need s.m entries each (<= n). out may alias k: k is not read after
// the collect.
inline void select_ranks(const uint32_t* k, uint32_t n, uint32_t lo, uint32_t hi, uint32_t* ck,
                         uint32_t* cv, uint32_t* ck2, uint32_t* cv2, uint32_t* out,
                         sort_radix_tile::hist_t* hist) {
    const Select s = select_bins(k, n, lo, hi, hist);
    const uint32_t m = select_collect(k, n, s, ck, cv);
    const uint32_t* res = sort_radix_tile::sort_pairs(ck, cv, ck2, cv2, m, hist) ? cv2 : cv;
    for (uint32_t i = lo; i < hi; i++) out[i - lo] = res[i - s.base];
}

// Task #284: a grown tile bucket (after an overflow) holds up to BIG_MAX_N
// records, more than the radix sorts (MAX_N); the u16 bin counts stay exact
// below 65536. One more level than select_ranks: after 4 rounds of 1024 bins
// every edge range is a single key. The collect then drops the edge keys'
// ties outside [lo, hi) (equal keys keep index order, so their ranks are
// known), leaving ~hi - lo candidates even on tied keys. ck, cv, ck2, cv2
// hold `cap` (<= MAX_N) entries each. Returns false (out partly written) if
// the candidates do not fit them.
constexpr uint32_t BIG_MAX_N = 65472;
constexpr uint32_t BIG_LEVELS = SELECT_LEVELS + 1u;
static_assert(BIG_MAX_N < 65536u, "u16 bin counts");
inline bool select_ranks_big(const uint32_t* k, uint32_t n, uint32_t lo, uint32_t hi, uint32_t* ck,
                             uint32_t* cv, uint32_t* ck2, uint32_t* cv2, uint32_t cap,
                             uint32_t* out, sort_radix_tile::hist_t* hist) {
    const Select s = select_bins(k, n, lo, hi, hist, BIG_LEVELS);
    const uint32_t span = s.khi - s.klo;
    // Ranks of the khi ties start at b_hi (keys below khi).
    uint32_t b_hi = s.base;
    if (s.khi != s.klo) {
        for (uint32_t i = 0; i < n; i++) b_hi += (k[i] - s.klo < span) ? 1u : 0u;
    }
    const uint32_t skip_lo = lo - s.base;
    uint32_t t_lo = 0, t_hi = 0, m = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t x = k[i];
        if (x - s.klo > span) continue;  // unsigned: outside [klo, khi]
        if (x == s.klo) {
            const uint32_t t = t_lo++;
            if (t < skip_lo || s.base + t >= hi) continue;  // rank base + t
        } else if (x == s.khi) {
            if (b_hi + t_hi++ >= hi) continue;
        }
        if (m == cap) return false;
        ck[m] = x;
        cv[m] = i;
        m++;
    }
    if (m > sort_radix_tile::MAX_N) return false;
    const uint32_t base = s.base + (t_lo < skip_lo ? t_lo : skip_lo);  // rank of ck[first]
    const uint32_t* res = sort_radix_tile::sort_pairs(ck, cv, ck2, cv2, m, hist) ? cv2 : cv;
    for (uint32_t i = lo; i < hi; i++) out[i - lo] = res[i - base];
    return true;
}

}  // namespace sort_ol
