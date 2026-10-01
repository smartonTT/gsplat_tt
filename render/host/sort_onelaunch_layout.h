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

#include <cstdint>
#include <vector>

namespace gsplat_tt::sort_onelaunch {

inline constexpr uint32_t kElemsPerPage = 16;  // u32 per 64 B page
inline constexpr uint32_t kDropped = 0xFFFFFFFFu;

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
