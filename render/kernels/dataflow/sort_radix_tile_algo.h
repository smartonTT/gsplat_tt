// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Adaptive stable LSD radix sort of (key, id) pairs for sort_radix_tile.cpp
// (R11). Plain C++ so tests/unit/test_sort_radix_tile.cpp can check it against
// std::stable_sort on the host.
//
// The old kernel always ran 4 x 8-bit passes, each re-reading the keys for its
// histogram. Here, per tile:
//   1. one scan finds kmin / kmax; all keys equal -> input order is the answer;
//   2. keys are sorted as r = key - kmin, which needs only B = bitlen(kmax-kmin)
//      bits (order-preserving since key >= kmin);
//   3. the pass count P (1..4) and digit width d = ceil(B/P) are picked per
//      tile from n and B with a small cost model (P << d <= HIST_ENTRIES);
//   4. one scan fills all P histograms, then P scatter passes; the last pass
//      writes ids only (the keys are not an output).
// Every choice is a STABLE sort by key, so the permutation (and the output) is
// identical to the old fixed 4 x 8-bit radix / n <= 16 insertion sort.
#pragma once

#include <cstdint>

namespace sort_radix_tile {

// Histograms live in the RISC's 8 KB local memory (kernel stack), not L1: a
// bucket counter is loaded right after the store to the same bucket, and on
// L1 that read-after-write costs ~7% of the radix time (measured, task #26).
// uint16 counts suffice since n <= 32768 (MAX_TILE_ENTRIES). 1536 entries =
// 3 KB covers 3 passes x 9-bit digits (27 key bits) or 4 x 8.
using hist_t = uint16_t;
#ifndef SORT_RADIX_HIST_ENTRIES
#define SORT_RADIX_HIST_ENTRIES 1536
#endif
constexpr uint32_t HIST_ENTRIES = SORT_RADIX_HIST_ENTRIES;
constexpr uint32_t MAX_PASSES = 4;           // 4 x 8 >= 32 bits always fits
constexpr uint32_t MAX_N = 32768;            // hist_t range
static_assert(HIST_ENTRIES >= MAX_PASSES * 256u, "4 x 8-bit plan must fit");

// Relative costs per pass: each bucket is cleared + prefix-summed, each element
// is histogrammed + scattered.
constexpr uint32_t BUCKET_COST = 1;
constexpr uint32_t ELEM_COST = 8;

constexpr uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

struct Plan {
    uint32_t passes;
    uint32_t bits;
};

// Pick (P, d) for n (<= MAX_N) elements whose relative keys need B (1..32) bits.
inline Plan choose_plan(uint32_t n, uint32_t B) {
    Plan best{0u, 0u};
    uint32_t best_cost = 0xFFFFFFFFu;
    for (uint32_t p = 1; p <= MAX_PASSES; p++) {
        const uint32_t d = ceil_div(B, p);
        if (d > 15u || (p << d) > HIST_ENTRIES) continue;  // d guard: no 1 << 32
        const uint32_t cost = p * ((BUCKET_COST << d) + ELEM_COST * n);
        if (cost < best_cost) {
            best_cost = cost;
            best.passes = p;
            best.bits = d;
        }
    }
    return best;
}

inline uint32_t bit_length(uint32_t x) {
    uint32_t b = 0;
    while (x) {
        x >>= 1;
        b++;
    }
    return b;
}

template <uint32_t P>
inline void hist_add(hist_t* hist, uint32_t r, uint32_t d, uint32_t mask) {
    const uint32_t R = mask + 1u;
    hist[r & mask]++;
    if (P > 1) hist[R + ((r >> d) & mask)]++;
    if (P > 2) hist[2u * R + ((r >> (2u * d)) & mask)]++;
    if (P > 3) hist[3u * R + ((r >> (3u * d)) & mask)]++;
}

// Loops below load UNROLL keys/ids up front so the ~8-cycle L1 load-to-use
// latency overlaps; the bucket updates stay in element order (stability).
constexpr uint32_t UNROLL = 4;

template <uint32_t P>
inline void fill_hist(const uint32_t* k, uint32_t n, uint32_t kmin, uint32_t d,
                      uint32_t mask, hist_t* hist) {
    uint32_t i = 0;
    for (; i + UNROLL <= n; i += UNROLL) {
        uint32_t r[UNROLL];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) r[u] = k[i + u] - kmin;
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) hist_add<P>(hist, r[u], d, mask);
    }
    for (; i < n; i++) hist_add<P>(hist, k[i] - kmin, d, mask);
}

// Stable sort of the pairs (k[i], v[i]), i < n, by k. (k2, v2) is ping-pong
// scratch of n entries; hist holds HIST_ENTRIES counters. Returns true when the
// sorted ids end up in v2, false when they are in v. Keys are clobbered.
inline bool sort_pairs(uint32_t* k, uint32_t* v, uint32_t* k2, uint32_t* v2,
                       uint32_t n, hist_t* hist) {
    if (n <= 16u) {
        for (uint32_t i = 1; i < n; i++) {
            const uint32_t kk = k[i];
            const uint32_t vv = v[i];
            uint32_t j = i;
            while (j > 0 && k[j - 1] > kk) {
                k[j] = k[j - 1];
                v[j] = v[j - 1];
                --j;
            }
            k[j] = kk;
            v[j] = vv;
        }
        return false;
    }

    uint32_t kmin = k[0];
    uint32_t kmax = k[0];
    for (uint32_t i = 1; i < n; i++) {
        const uint32_t x = k[i];
        kmin = x < kmin ? x : kmin;
        kmax = x > kmax ? x : kmax;
    }
    if (kmin == kmax) return false;  // all keys equal: stable order == input

    const Plan pl = choose_plan(n, bit_length(kmax - kmin));
    const uint32_t P = pl.passes;
    const uint32_t d = pl.bits;
    const uint32_t R = 1u << d;
    const uint32_t mask = R - 1u;

    for (uint32_t j = 0; j < P * R; j++) hist[j] = 0;
    switch (P) {
        case 1: fill_hist<1>(k, n, kmin, d, mask, hist); break;
        case 2: fill_hist<2>(k, n, kmin, d, mask, hist); break;
        case 3: fill_hist<3>(k, n, kmin, d, mask, hist); break;
        default: fill_hist<4>(k, n, kmin, d, mask, hist); break;
    }
    for (uint32_t p = 0; p < P; p++) {
        hist_t* h = hist + p * R;
        uint32_t sum = 0;
        for (uint32_t b = 0; b < R; b++) {
            const uint32_t c = h[b];
            h[b] = static_cast<hist_t>(sum);
            sum += c;
        }
    }

    uint32_t* ik = k;
    uint32_t* iv = v;
    uint32_t* ok = k2;
    uint32_t* ov = v2;
    for (uint32_t p = 0; p < P; p++) {
        hist_t* h = hist + p * R;
        const uint32_t sh = p * d;
        const uint32_t sub = (p == 0) ? kmin : 0u;  // pass 0 stores r = k - kmin
        uint32_t i = 0;
        if (p + 1u == P) {
            for (; i + UNROLL <= n; i += UNROLL) {
                uint32_t r[UNROLL], vv[UNROLL];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) {
                    r[u] = ik[i + u] - sub;
                    vv[u] = iv[i + u];
                }
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) ov[h[(r[u] >> sh) & mask]++] = vv[u];
            }
            for (; i < n; i++) ov[h[((ik[i] - sub) >> sh) & mask]++] = iv[i];
        } else {
            for (; i + UNROLL <= n; i += UNROLL) {
                uint32_t r[UNROLL], vv[UNROLL];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) {
                    r[u] = ik[i + u] - sub;
                    vv[u] = iv[i + u];
                }
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) {
                    const uint32_t pos = h[(r[u] >> sh) & mask]++;
                    ok[pos] = r[u];
                    ov[pos] = vv[u];
                }
            }
            for (; i < n; i++) {
                const uint32_t r = ik[i] - sub;
                const uint32_t pos = h[(r >> sh) & mask]++;
                ok[pos] = r;
                ov[pos] = iv[i];
            }
        }
        uint32_t* t = ik; ik = ok; ok = t;
        t = iv; iv = ov; ov = t;
    }
    return (P & 1u) != 0u;
}

}  // namespace sort_radix_tile
