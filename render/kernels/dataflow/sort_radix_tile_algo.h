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

namespace sort_radix_tile {

// Task #418: SORT_RECS_PACKED=1 (host GSPLAT_TT_SORT_PACKED, default on) sorts
// the record ids with one u32 per element after the first pass: w = (r >> d)
// << ib | id, ib = bitlen(n - 1), whenever the key bits left after pass 0 fit
// next to the id (B - d + ib <= 32; per-tile depth ranges almost always do).
// Pass 0 reads the gathered keys and takes the id from the loop index, the last
// pass writes ids straight into v, and kmin / kmax come from the gather loop.
// Per element that halves the L1 traffic of sort_pairs (no id array, no
// key + id pair moves, no odd-pass copy back). Still a stable LSD radix with
// the same plan, so the permutation is the same.
#ifndef SORT_RECS_PACKED
#define SORT_RECS_PACKED 1
#endif

// Ids of the stable order of k[0, n) (n > 16, kmin < kmax) into out; w2 is n u32
// of scratch, k is clobbered. Requires P == 1 or B - d + ib <= 32.
inline void sort_ids_packed(uint32_t* k, uint32_t* out, uint32_t* w2, uint32_t n,
                            uint32_t kmin, const Plan& pl, uint32_t ib, hist_t* hist) {
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
    uint32_t i = 0;
    if (P == 1) {
        for (; i + UNROLL <= n; i += UNROLL) {
            uint32_t r[UNROLL];
#pragma GCC unroll 4
            for (uint32_t u = 0; u < UNROLL; u++) r[u] = k[i + u] - kmin;
#pragma GCC unroll 4
            for (uint32_t u = 0; u < UNROLL; u++) out[hist[r[u] & mask]++] = i + u;
        }
        for (; i < n; i++) out[hist[(k[i] - kmin) & mask]++] = i;
        return;
    }
    // Pass 0: keys -> packed words in w2.
    for (; i + UNROLL <= n; i += UNROLL) {
        uint32_t r[UNROLL];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) r[u] = k[i + u] - kmin;
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) w2[hist[r[u] & mask]++] = ((r[u] >> d) << ib) | (i + u);
    }
    for (; i < n; i++) {
        const uint32_t r = k[i] - kmin;
        w2[hist[r & mask]++] = ((r >> d) << ib) | i;
    }
    const uint32_t idmask = (1u << ib) - 1u;
    uint32_t* src = w2;
    uint32_t* dst = k;
    for (uint32_t p = 1; p < P; p++) {
        hist_t* h = hist + p * R;
        const uint32_t sh = ib + (p - 1u) * d;
        i = 0;
        if (p + 1u == P) {
            for (; i + UNROLL <= n; i += UNROLL) {
                uint32_t w[UNROLL];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) w[u] = src[i + u];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) out[h[(w[u] >> sh) & mask]++] = w[u] & idmask;
            }
            for (; i < n; i++) out[h[(src[i] >> sh) & mask]++] = src[i] & idmask;
        } else {
            for (; i + UNROLL <= n; i += UNROLL) {
                uint32_t w[UNROLL];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) w[u] = src[i + u];
#pragma GCC unroll 4
                for (uint32_t u = 0; u < UNROLL; u++) dst[h[(w[u] >> sh) & mask]++] = w[u];
            }
            for (; i < n; i++) dst[h[(src[i] >> sh) & mask]++] = src[i];
            uint32_t* t = src; src = dst; dst = t;
        }
    }
}

// Stable depth order of n 32 B records (8 u32 words, record i at words
// [8i, 8i+8), depth key at word 3) for sort_subchunk_materialize.cpp. Returns
// the sorted record indices, always in v. k, v, k2, v2 hold n entries each.
// Same permutation as the old fixed 4 x 8-bit radix / n <= 16 insertion sort
// (both are stable sorts by the key), with the keys gathered once instead of
// re-read through the index on every pass.
inline uint32_t* sort_record_ids(const volatile uint32_t* recs, uint32_t n, uint32_t* k,
                                 uint32_t* v, uint32_t* k2, uint32_t* v2, hist_t* hist) {
    uint32_t i = 0;
#if SORT_RECS_PACKED
    if (n > 16u) {
        uint32_t kmin = 0xFFFFFFFFu, kmax = 0u;
        for (; i + UNROLL <= n; i += UNROLL) {
            uint32_t kk[UNROLL];
#pragma GCC unroll 4
            for (uint32_t u = 0; u < UNROLL; u++) kk[u] = recs[(i + u) * 8u + 3u];
#pragma GCC unroll 4
            for (uint32_t u = 0; u < UNROLL; u++) {
                k[i + u] = kk[u];
                kmin = kk[u] < kmin ? kk[u] : kmin;
                kmax = kk[u] > kmax ? kk[u] : kmax;
            }
        }
        for (; i < n; i++) {
            const uint32_t x = recs[i * 8u + 3u];
            k[i] = x;
            kmin = x < kmin ? x : kmin;
            kmax = x > kmax ? x : kmax;
        }
        if (kmin == kmax) {
            for (i = 0; i < n; i++) v[i] = i;
            return v;
        }
        const uint32_t B = bit_length(kmax - kmin);
        const Plan pl = choose_plan(n, B);
        const uint32_t ib = bit_length(n - 1u);
        if (pl.passes == 1u || B - pl.bits + ib <= 32u) {
            sort_ids_packed(k, v, k2, n, kmin, pl, ib, hist);
            return v;
        }
        for (i = 0; i < n; i++) v[i] = i;  // wide key range: the pair sort below
        if (sort_pairs(k, v, k2, v2, n, hist)) {
            for (i = 0; i < n; i++) v[i] = v2[i];
        }
        return v;
    }
#endif
    for (; i + UNROLL <= n; i += UNROLL) {
        uint32_t kk[UNROLL];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) kk[u] = recs[(i + u) * 8u + 3u];
#pragma GCC unroll 4
        for (uint32_t u = 0; u < UNROLL; u++) {
            k[i + u] = kk[u];
            v[i + u] = i + u;
        }
    }
    for (; i < n; i++) {
        k[i] = recs[i * 8u + 3u];
        v[i] = i;
    }
    if (sort_pairs(k, v, k2, v2, n, hist)) {
        for (i = 0; i < n; i++) v[i] = v2[i];
    }
    return v;
}

}  // namespace sort_radix_tile
