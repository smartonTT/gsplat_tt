// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort_onelaunch_layout.h — host model of the task #106 one-launch device sort
// (render/kernels/dataflow/sort_bin_onelaunch.cpp, GSPLAT_TT_SORT_ONELAUNCH=1).
// Header-only and free of tt-metal types so tests/unit/test_sort_onelaunch.cpp
// can check the device prefix and the bucket order against the legacy layout;
// sort_device.cpp uses check_prefix for GSPLAT_TT_SORT_ONELAUNCH_CHECK=1.
//
// Rows are `stride` u32 per core (num_tiles rounded up to 16). Core c's pair
// pages [lo, hi) are split at mid: mover 0 (BRISC) emits [lo, mid), mover 1
// (NCRISC) [mid, hi). Tile t's records go to slots t * tile_cap + cursor, the
// cursor starting at base[c][t] (mover 0) or base[c][t] + h0[c][t] (mover 1),
// so each bucket holds the tile's kept pairs in ascending pair order — the
// order the legacy prefix layout feeds its stable radix.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_tt::sort_onelaunch {

inline constexpr uint32_t kElemsPerPage = 16;  // u32 per 64 B page
inline constexpr uint32_t kDropped = 0xFFFFFFFFu;
// Records per tile bucket (== sort_device.cpp MAX_TILE_ENTRIES, the legacy
// per-tile limit). The materialize's big path keeps keys + 3 tile_cap u32
// arrays in its 512 KB bucket CB, so tile_cap <= 43690.
inline constexpr uint32_t kTileCap = 32768;
// Task #284: the bucket after a frame overflowed kTileCap (sort_device.cpp
// grows it once and keeps it). == sort_onelaunch_algo.h BIG_MAX_N: the
// materialize selects the ranks of tiles over kTileCap instead of sorting
// all their keys. A multiple of the 64-record DRAM page.
inline constexpr uint32_t kTileCapBig = 65472;
static_assert(kTileCapBig % 64u == 0u, "whole record pages");
// Task #365: the bucket is interleaved over the DRAM banks by 2 KB record page
// (64 records), so tile t's page k sits in bank (t * pages + k) % nbanks. kTileCap
// is 512 pages: with the p150's 8 banks page k of every tile lands in the same
// bank, and the movers early in the prefix order (bases near 0) all write bank 0.
// bucket_tile_cap pads the tile stride to the next page count coprime with the
// bank count (8 banks: 513 pages); with 7 banks (p100a) 512 stays. pad_pages >= 0
// forces that many extra pages instead (GSPLAT_TT_OL_TILE_PAD; 0 = pre-#365).
inline constexpr uint32_t gcd_u32(uint32_t a, uint32_t b) {
    while (b != 0u) {
        const uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}
inline constexpr uint32_t bucket_tile_cap(uint32_t cap, uint32_t nbanks, int pad_pages) {
    uint32_t pages = cap / 64u;
    if (pad_pages >= 0) return (pages + static_cast<uint32_t>(pad_pages)) * 64u;
    while (nbanks > 1u && gcd_u32(pages, nbanks) != 1u) pages++;
    return pages * 64u;
}
static_assert(bucket_tile_cap(kTileCap, 7u, -1) == kTileCap, "p100a: stride unchanged");
static_assert(bucket_tile_cap(kTileCap, 8u, -1) == kTileCap + 64u, "p150: 513 pages");
static_assert(bucket_tile_cap(kTileCap, 8u, -1) <= 43690u, "materialize big-path limit");

struct CoreSplit {
    uint32_t lo = 0, mid = 0, hi = 0;  // pair pages: mover 0 [lo, mid), mover 1 [mid, hi)
};

// Count pass: h0 = mover 0's counts, h = the core's counts (h0 + h1), rows of
// `stride` u32. A pair is kept iff i < P and keep[i] != 0.
struct Counts {
    std::vector<uint32_t> h0, h;
};

inline Counts count_pass(const std::vector<int32_t>& tids, const std::vector<int32_t>& keep,
                         uint32_t P, const std::vector<CoreSplit>& split, uint32_t stride) {
    const uint32_t n = static_cast<uint32_t>(split.size());
    Counts out;
    out.h0.assign(static_cast<std::size_t>(n) * stride, 0u);
    out.h.assign(static_cast<std::size_t>(n) * stride, 0u);
    for (uint32_t c = 0; c < n; ++c) {
        for (uint32_t pg = split[c].lo; pg < split[c].hi; ++pg) {
            for (uint32_t j = 0; j < kElemsPerPage; ++j) {
                const uint32_t i = pg * kElemsPerPage + j;
                if (i >= P) break;
                if (keep[i] == 0) continue;
                const std::size_t at = static_cast<std::size_t>(c) * stride +
                                       static_cast<uint32_t>(tids[i]);
                out.h[at]++;
                if (pg < split[c].mid) out.h0[at]++;
            }
        }
    }
    return out;
}

// The kernel's column prefix: the owner of page p (cores p, p + n, ...) walks
// the cores in order for each of the page's 16 tiles. base = exclusive prefix
// of h over cores; totals[t] = records of tile t; totals[stride + t] = sum of
// the cores' ceil16(h) (== the legacy tile_pad, the LPT cost).
struct Prefix {
    std::vector<uint32_t> base;    // num_cores * stride
    std::vector<uint32_t> totals;  // 2 * stride
};

inline Prefix device_prefix(const std::vector<uint32_t>& h, uint32_t num_cores, uint32_t stride) {
    Prefix out;
    out.base.assign(static_cast<std::size_t>(num_cores) * stride, 0u);
    out.totals.assign(static_cast<std::size_t>(stride) * 2u, 0u);
    const uint32_t row_pages = stride / kElemsPerPage;
    for (uint32_t owner = 0; owner < num_cores; ++owner) {
        for (uint32_t p = owner; p < row_pages; p += num_cores) {
            for (uint32_t j = 0; j < kElemsPerPage; ++j) {
                const uint32_t t = p * kElemsPerPage + j;
                uint32_t acc = 0, pad = 0;
                for (uint32_t r = 0; r < num_cores; ++r) {
                    const uint32_t v = h[static_cast<std::size_t>(r) * stride + t];
                    out.base[static_cast<std::size_t>(r) * stride + t] = acc;
                    acc += v;
                    pad += (v + 15u) & ~15u;
                }
                out.totals[t] = acc;
                out.totals[stride + t] = pad;
            }
        }
    }
    return out;
}

// Task #198 (GSPLAT_TT_MAT_CQ1): device_prefix's totals from the fold K2's
// count rows (row 2c + mover, stride words each; a core's h is its two rows'
// sum), so the host has them before the one-launch sort ends. tot holds at
// least 2 * stride words: [0, stride) per-tile counts, [stride, 2 stride) the
// per-core counts padded to 16, summed.
inline void totals_from_k2_rows(const std::vector<uint32_t>& krow, uint32_t num_cores,
                                uint32_t stride, std::vector<uint32_t>& tot) {
    std::fill(tot.begin(), tot.begin() + 2u * static_cast<std::size_t>(stride), 0u);
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t* r0 = krow.data() + 2u * static_cast<std::size_t>(c) * stride;
        const uint32_t* r1 = r0 + stride;
        for (uint32_t t = 0; t < stride; ++t) {
            const uint32_t h = r0[t] + r1[t];
            tot[t] += h;
            tot[stride + t] += (h + 15u) & ~15u;
        }
    }
}

// Emit: the bucket slot of every pair (kDropped if not kept or past tile_cap).
inline std::vector<uint32_t> emit_slots(const std::vector<int32_t>& tids,
                                        const std::vector<int32_t>& keep, uint32_t P,
                                        const std::vector<CoreSplit>& split, const Counts& cnt,
                                        const Prefix& pfx, uint32_t stride, uint32_t tile_cap) {
    std::vector<uint32_t> slot(tids.size(), kDropped);
    const uint32_t n = static_cast<uint32_t>(split.size());
    for (uint32_t c = 0; c < n; ++c) {
        const std::size_t row = static_cast<std::size_t>(c) * stride;
        for (uint32_t mover = 0; mover < 2; ++mover) {
            std::vector<uint32_t> cur(stride);
            for (uint32_t t = 0; t < stride; ++t) {
                cur[t] = pfx.base[row + t] + (mover == 1 ? cnt.h0[row + t] : 0u);
            }
            const uint32_t pg0 = mover == 0 ? split[c].lo : split[c].mid;
            const uint32_t pg1 = mover == 0 ? split[c].mid : split[c].hi;
            for (uint32_t pg = pg0; pg < pg1; ++pg) {
                for (uint32_t j = 0; j < kElemsPerPage; ++j) {
                    const uint32_t i = pg * kElemsPerPage + j;
                    if (i >= P) break;
                    if (keep[i] == 0) continue;
                    const uint32_t t = static_cast<uint32_t>(tids[i]);
                    const uint32_t k = cur[t]++;
                    slot[i] = (k < tile_cap) ? t * tile_cap + k : kDropped;
                }
            }
        }
    }
    return slot;
}

// GSPLAT_TT_SORT_ONELAUNCH_CHECK: the device's count rows (crow), base rows
// (brow) and totals against the prefix recomputed from crow. Returns the
// number of tiles with any mismatch.
inline uint32_t check_prefix(const std::vector<uint32_t>& crow, const std::vector<uint32_t>& brow,
                             const std::vector<uint32_t>& tot, uint32_t num_cores, uint32_t stride,
                             uint32_t num_tiles) {
    const Prefix ref = device_prefix(crow, num_cores, stride);
    uint32_t bad = 0;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        bool ok = tot[t] == ref.totals[t] && tot[stride + t] == ref.totals[stride + t];
        for (uint32_t c = 0; ok && c < num_cores; ++c) {
            const std::size_t at = static_cast<std::size_t>(c) * stride + t;
            ok = brow[at] == ref.base[at];
        }
        if (!ok) bad++;
    }
    return bad;
}

}  // namespace gsplat_tt::sort_onelaunch
