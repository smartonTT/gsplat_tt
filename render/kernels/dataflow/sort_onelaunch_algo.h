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
//    histograms over 1024 linear bins find a key range [klo, khi] holding ranks
//    lo..hi-1; only the keys in it (in index order) are radix-sorted. All
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

// Up to SELECT_LEVELS histogram passes over k[0..n), each narrowing [klo, khi]
// to the bins of ranks lo and hi-1 among 1024 linear bins of the current range
// (a later level only helps when outliers stretch the range, so most keys share
// a few bins). Stops once m <= 2 (hi - lo) or the bins are single keys.
// 0 <= lo < hi <= n <= MAX_N.
inline Select select_bins(const uint32_t* k, uint32_t n, uint32_t lo, uint32_t hi,
                          sort_radix_tile::hist_t* hist) {
    Select s{k[0], k[0], 0u, n};
    for (uint32_t i = 1; i < n; i++) {
        const uint32_t x = k[i];
        s.klo = x < s.klo ? x : s.klo;
        s.khi = x > s.khi ? x : s.khi;
    }
    for (uint32_t lvl = 0; lvl < SELECT_LEVELS; lvl++) {
        const uint32_t span = s.khi - s.klo;
        uint32_t shift = 0;
        while ((span >> shift) >= SELECT_BINS) shift++;
        for (uint32_t b = 0; b < SELECT_BINS; b++) hist[b] = 0;
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t d = k[i] - s.klo;  // unsigned: keys below klo wrap past span
            if (d <= span) hist[d >> shift]++;
        }
        uint32_t cum = s.base, base = s.base, b_lo = 0, b_hi = 0;
        bool have_lo = false;
        for (uint32_t b = 0; b < SELECT_BINS; b++) {
            const uint32_t c = hist[b];
            if (!have_lo && cum + c > lo) {
                have_lo = true;
                b_lo = b;
                base = cum;
            }
            cum += c;
            if (cum >= hi) {
                b_hi = b;
                break;
            }
        }
        const uint32_t klo = s.klo + (b_lo << shift);
        // Last key of bin b_hi; wraps to 0xFFFFFFFF (>= span) only for the
        // last of 1024 bins at shift 22, where khi stays.
        const uint32_t top = ((b_hi + 1u) << shift) - 1u;
        s.khi = (top < span) ? s.klo + top : s.khi;
        s.klo = klo;
        s.base = base;
        s.m = cum - base;
        if (shift == 0u || s.m <= 2u * (hi - lo)) break;
    }
    return s;
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

}  // namespace sort_ol
