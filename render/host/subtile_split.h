// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// subtile_split.h — task #719 (B1, docs/training-plan/t710-lowres-blend/SIZING.md):
// host model of the adaptive heavy-tile split. Header-only and tt-metal free
// (tests/unit/test_subtile_split.cpp).
//
// A 32x32 tile whose estimated cost passes total / num_cores becomes four 16x16
// sub-tiles; a quarter whose own estimate still passes it becomes four 8x8
// sub-tiles. Sub-tile ids follow the screen tiles: [num_tiles, num_ids()). A split
// tile's own list is empty; each sub-tile has its own exact list.
//
// Lossless by construction: a sub-tile is its parent tile plus a microblock mask.
// Records stay parent-relative (the sort packs them at the parent's origin), so
// every float the blend sees is the one it sees today. A record goes to sub-tile
// s iff its microblock box (the cull's [mx_lo, mx_hi] x [my_lo, my_hi]) meets s's
// mask, so every record with a mask bit in s is in s's list, in the parent's
// order (the bucket keeps ascending pair order and the radix is stable).
//
// Microblock m = (my << 2) | mx: my 0..7 (4-pixel rows), mx 0..3 (8-pixel
// columns), as in src/gsplat_cpu/microblock_cull.cpp and the device mat cull.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "../kernels/dataflow/subtile_route.h"
#include "../kernels/dataflow/subtile_unit.h"

namespace gsplat_tt::subtile {

inline constexpr uint32_t kFullMask = 0xFFFFFFFFu;
inline constexpr uint32_t kNone = 0xFFFFFFFFu;

// Microblock mask of the 16x16 quarter q (qx = q & 1, qy = q >> 1): mx in
// {2qx, 2qx+1}, my in [4qy, 4qy+4).
inline constexpr uint32_t quarter_mask(uint32_t q) {
    uint32_t m = 0;
    const uint32_t qx = q & 1u, qy = q >> 1;
    for (uint32_t my = 4u * qy; my < 4u * qy + 4u; ++my)
        for (uint32_t mx = 2u * qx; mx < 2u * qx + 2u; ++mx) m |= 1u << ((my << 2) | mx);
    return m;
}

// Microblock mask of the 8x8 sub-tile (ex, ey), ex, ey in 0..3: mx = ex,
// my in {2ey, 2ey+1}.
inline constexpr uint32_t eighth_mask(uint32_t ex, uint32_t ey) {
    return (1u << (((2u * ey) << 2) | ex)) | (1u << (((2u * ey + 1u) << 2) | ex));
}

// Mask of the microblock box [mx_lo, mx_hi] x [my_lo, my_hi] (inclusive, clamped
// to the tile by the caller).
inline uint32_t box_mask(int mx_lo, int mx_hi, int my_lo, int my_hi) {
    uint32_t m = 0;
    for (int my = my_lo; my <= my_hi; ++my)
        for (int mx = mx_lo; mx <= mx_hi; ++mx) m |= 1u << ((my << 2) | mx);
    return m;
}

struct Plan {
    uint32_t num_tiles = 0;
    std::vector<uint32_t> first_id;  // per tile: kNone if unsplit, else its first sub id
    std::vector<uint32_t> n_sub;     // per tile: 0, 4 .. 16 sub-tiles
    std::vector<uint32_t> parent;    // per sub id - num_tiles
    std::vector<uint32_t> mask;      // per sub id - num_tiles: microblock mask
    std::vector<uint32_t> code;      // plan_units only: per sub id - num_tiles, B1' unit code
    uint32_t num_ids() const { return num_tiles + static_cast<uint32_t>(parent.size()); }
    uint32_t parent_of(uint32_t id) const { return id < num_tiles ? id : parent[id - num_tiles]; }
    uint32_t mask_of(uint32_t id) const { return id < num_tiles ? kFullMask : mask[id - num_tiles]; }
};

// tile_cost: per-tile estimate (e.g. the previous view's microblock touches or
// pair counts). quarter_cost: optional per-tile 16x16 quarter estimates (from a
// previous split); without it a quarter is a quarter of its tile. The plan only
// moves records between lists, so a stale estimate costs balance, never output.
inline Plan plan_split(const std::vector<double>& tile_cost,
                       const std::vector<std::array<double, 4>>* quarter_cost,
                       uint32_t num_cores) {
    Plan p;
    p.num_tiles = static_cast<uint32_t>(tile_cost.size());
    p.first_id.assign(p.num_tiles, kNone);
    p.n_sub.assign(p.num_tiles, 0);
    double total = 0.0;
    for (double c : tile_cost) total += c;
    const double thr = num_cores != 0u ? total / num_cores : 0.0;
    if (thr <= 0.0) return p;
    for (uint32_t t = 0; t < p.num_tiles; ++t) {
        if (!(tile_cost[t] > thr)) continue;
        p.first_id[t] = p.num_ids();
        for (uint32_t q = 0; q < 4u; ++q) {
            const double qc = quarter_cost ? (*quarter_cost)[t][q] : tile_cost[t] / 4.0;
            if (qc > thr) {
                const uint32_t qx = q & 1u, qy = q >> 1;
                for (uint32_t ey = 2u * qy; ey < 2u * qy + 2u; ++ey)
                    for (uint32_t ex = 2u * qx; ex < 2u * qx + 2u; ++ex) {
                        p.parent.push_back(t);
                        p.mask.push_back(eighth_mask(ex, ey));
                    }
            } else {
                p.parent.push_back(t);
                p.mask.push_back(quarter_mask(q));
            }
        }
        p.n_sub[t] = p.num_ids() - p.first_id[t];
    }
    return p;
}

// #744 (B1 route, PLAN.md §7.3): the same plan in B1' unit codes, the shapes the
// blend already writes. A tile splits when its cost passes total / num_cores
// (balance) or cap (its bucket would overflow): 16 bands (16x4) when it passes
// 4 x that or cap, else 4 quarters; sub k is unit subtile_unit::split_code(n, k).
// Heaviest tiles first while num_ids() stays <= max_ids (a band split that does
// not fit falls back to quarters); ids are then assigned in tile order.
inline Plan plan_units(const std::vector<double>& tile_cost, uint32_t num_cores, uint32_t max_ids,
                       double cap) {
    Plan p;
    p.num_tiles = static_cast<uint32_t>(tile_cost.size());
    p.first_id.assign(p.num_tiles, kNone);
    p.n_sub.assign(p.num_tiles, 0);
    double total = 0.0;
    for (double c : tile_cost) total += c;
    const double thr = num_cores != 0u ? total / num_cores : 0.0;
    if (thr <= 0.0) return p;
    std::vector<uint32_t> order(p.num_tiles);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return tile_cost[a] > tile_cost[b]; });
    uint32_t budget = max_ids > p.num_tiles ? max_ids - p.num_tiles : 0u;
    for (const uint32_t t : order) {
        const double c = tile_cost[t];
        if (!(c > thr) && !(c > cap)) break;
        uint32_t n = (c > 4.0 * thr || c > cap) ? 16u : 4u;
        if (n > budget) n = 4u;
        if (n > budget) break;
        p.n_sub[t] = n;
        budget -= n;
    }
    for (uint32_t t = 0; t < p.num_tiles; ++t) {
        if (p.n_sub[t] == 0u) continue;
        p.first_id[t] = p.num_ids();
        for (uint32_t k = 0; k < p.n_sub[t]; ++k) {
            const uint32_t code = subtile_unit::split_code(p.n_sub[t], k);
            p.parent.push_back(t);
            p.mask.push_back(subtile_unit::mask(code));
            p.code.push_back(code);
        }
    }
    return p;
}

// The device plan table (sort_bin_onelaunch.cpp SORT_B1_ROUTE): per screen tile
// subtile_route::plan_word(first_id, n_sub), 0 if unsplit.
inline std::vector<uint32_t> plan_words(const Plan& p) {
    std::vector<uint32_t> w(p.num_tiles, 0u);
    for (uint32_t t = 0; t < p.num_tiles; ++t)
        if (p.first_id[t] != kNone) w[t] = subtile_route::plan_word(p.first_id[t], p.n_sub[t]);
    return w;
}

// The ids a record of tile t with microblock box mask `bm` is emitted to: t
// itself when t is unsplit, else every sub-tile of t whose mask meets bm. Host
// model of the device emit.
template <typename F>
inline void for_each_target(const Plan& p, uint32_t t, uint32_t bm, F&& f) {
    if (p.first_id[t] == kNone) {
        f(t);
        return;
    }
    for (uint32_t k = 0; k < p.n_sub[t]; ++k) {
        const uint32_t id = p.first_id[t] + k;
        if (p.mask[id - p.num_tiles] & bm) f(id);
    }
}

}  // namespace gsplat_tt::subtile
