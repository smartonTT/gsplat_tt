// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort_mover_split.h — host-side work splits for the sort tail kernels
// (sort_subchunk_materialize, sort_radix_tile). Header-only and free of
// tt-metal types so tests/unit/test_sort_tail_dual_mover.cpp can check them.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "config.h"

namespace gsplat_tt::sort_split {

// Record capacity of BRISC's materialize buffers (L1: 72 B / record).
inline constexpr uint32_t kMatMover0Cap = 6144;
// Over-cap gather subchunks are split into parts of this many records (the
// second work-item word is sc | part << 8); one whole 8192-record subchunk was
// a single ~8x-weighted item that set the busiest core's time by itself.
// == sort_subchunk_materialize.cpp GATHER_PART_RECS.
inline constexpr uint32_t kGatherPartRecs = 2048;

// iter 130: materialize work-item assignment — balance at (tile, subchunk)
// granularity. iter-130 MEASURED the dominant materialize cost as the OVERFLOW
// gather (24.6 ms/view busiest-core vs the in-budget permute's 1.7 ms), and the
// shared per-tile count-LPT overloads cores owning big overflow tiles (max 27.1
// vs the 17.0 ms balanced floor). Each (tile, sc) item is independent and writes
// byte-identical output regardless of which core runs it (in-budget reads
// buf_l1_recs by tile / writes payload by (tile,sc); gather reads sorted_ids +
// blendrec by global id / writes payload by (tile,sc)). So greedily LPT-balance
// all items, weighting gather subchunks GATHER_WEIGHT x their record count.
//
// Dual mover (task #35): with movers == 2 the items are balanced over 2 slots
// per core, slot 2c = NCRISC, 2c+1 = BRISC. BRISC's L1 buffers hold only
// m0_cap records, so a whole-tile item of more records may only go to an
// NCRISC slot; gather items use no big buffer and go anywhere. Items stay
// independent, so the output is byte-identical to the single-mover pass.
struct MatWorkAssignment {
    std::vector<uint32_t> flat;             // 2 u32 / item: {tile_id, sc | part << 8}
    std::vector<uint32_t> per_core_offset;  // in ITEMS, per slot
    std::vector<uint32_t> per_core_count;   // in ITEMS, per slot
    uint32_t max_items_per_core = 0;
    uint32_t movers = 1;
};

inline MatWorkAssignment build_mat_worklist(
    const std::vector<int64_t>& counts,
    uint32_t num_tiles,
    uint32_t num_cores,
    uint32_t bucket_fit,
    uint32_t movers = 1,
    uint32_t m0_cap = 0) {
    constexpr uint64_t GATHER_WEIGHT = 8;  // gather ~8-10x an in-budget record
    // iter-138: overflow tiles within the L1 cap are pre-packed at emit; the
    // materialize path reads the WHOLE tile coalesced + L1-radix-permutes it in a
    // SINGLE work item (sc==0, processes every subchunk internally) — like the
    // in-budget permute, ~1x per record (NOT the GATHER_WEIGHT random gather).
    const uint32_t ov_cap = render_config::kOverflowL1Cap;
    struct Item { uint32_t tile; uint32_t sc; uint64_t cost; bool big; };
    std::vector<Item> items;
    items.reserve(static_cast<std::size_t>(num_tiles) + 256u);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t cnt = static_cast<uint32_t>(counts[t]);
        if (cnt == 0u) continue;
        const bool inbudget = (cnt <= bucket_fit);
        const bool prepack_ov = (cnt > bucket_fit && cnt <= ov_cap);
        if (inbudget || prepack_ov) {
            // ONE whole-tile item: coalesced bucket read + L1 depth permute.
            items.push_back({t, 0u, static_cast<uint64_t>(cnt), cnt > m0_cap});
            continue;
        }
        // Over-cap overflow tile: legacy per-subchunk blendrec gather.
        const uint32_t num_sc = (cnt + bucket_fit - 1u) / bucket_fit;
        for (uint32_t sc = 0; sc < num_sc; ++sc) {
            const uint32_t sc_off = sc * bucket_fit;
            const uint32_t l_sub = (sc_off >= cnt) ? 0u
                : ((cnt - sc_off > bucket_fit) ? bucket_fit : (cnt - sc_off));
            for (uint32_t p0 = 0, part = 0; p0 < l_sub; p0 += kGatherPartRecs, ++part) {
                const uint32_t recs = std::min(kGatherPartRecs, l_sub - p0);
                items.push_back({t, sc | (part << 8),
                                 static_cast<uint64_t>(recs) * GATHER_WEIGHT, false});
            }
        }
    }
    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.cost > b.cost; });
    const uint32_t slots = num_cores * movers;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> per_core(slots);
    std::vector<uint64_t> load(slots, 0);
    for (const auto& it : items) {
        // Least-loaded eligible slot, ties -> lowest index (single mover: the
        // same choice as std::min_element over cores).
        const uint32_t step = (movers == 2 && it.big) ? 2u : 1u;
        uint32_t c = 0;
        for (uint32_t k = step; k < slots; k += step) {
            if (load[k] < load[c]) c = k;
        }
        per_core[c].emplace_back(it.tile, it.sc);
        load[c] += it.cost;
    }
    if (std::getenv("GSPLAT_TT_MAT_STATS") != nullptr) {
        uint64_t tot = 0, big = 0, gather = 0, mx[2] = {0, 0};
        for (const auto& it : items) {
            tot += it.cost;
            if (it.big) big += it.cost;
            if (it.cost > 0 && !it.big && counts[it.tile] > ov_cap) gather += it.cost;
        }
        for (uint32_t k = 0; k < slots; ++k) {
            const uint32_t m = (movers == 2) ? (k & 1u) : 0u;
            mx[m] = std::max(mx[m], load[k]);
        }
        std::fprintf(stderr,
                     "[MAT_STATS] items=%zu slots=%u cost_total=%llu big=%llu "
                     "gather=%llu max_item=%llu mean_slot=%llu "
                     "max_ncrisc=%llu max_brisc=%llu\n",
                     items.size(), slots, (unsigned long long)tot,
                     (unsigned long long)big, (unsigned long long)gather,
                     (unsigned long long)(items.empty() ? 0 : items[0].cost),
                     (unsigned long long)(tot / std::max(slots, 1u)),
                     (unsigned long long)mx[0], (unsigned long long)mx[1]);
    }
    MatWorkAssignment a;
    a.movers = movers;
    a.per_core_offset.assign(slots, 0);
    a.per_core_count.assign(slots, 0);
    for (uint32_t c = 0; c < slots; ++c) {
        a.per_core_offset[c] = static_cast<uint32_t>(a.flat.size() / 2u);
        a.per_core_count[c] = static_cast<uint32_t>(per_core[c].size());
        a.max_items_per_core = std::max(a.max_items_per_core, a.per_core_count[c]);
        for (const auto& pr : per_core[c]) {
            a.flat.push_back(pr.first);
            a.flat.push_back(pr.second);
        }
    }
    return a;
}

// Dual-mover radix split of one core's LPT slice (descending cost): the k
// minimizing max(cost of the first k tiles, cost of the rest). Cost model of
// sort_radix_tile per tile: n <= 16 is an insertion sort, else 4 radix passes
// whose 256-bucket clear + prefix is worth ~256 elements.
inline uint32_t radix_split_point(
    const std::vector<uint32_t>& tile_ids, uint32_t start, uint32_t count,
    const std::vector<int64_t>& counts) {
    auto cost = [&](uint32_t i) -> uint64_t {
        const uint64_t n = static_cast<uint64_t>(counts[tile_ids[start + i]]);
        return (n <= 16u) ? 32u : n + 256u;
    };
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i) total += cost(i);
    uint32_t best_k = count;
    uint64_t best = total;
    uint64_t pre = 0;
    for (uint32_t k = 0; k <= count; ++k) {
        const uint64_t m = std::max(pre, total - pre);
        if (m < best) {
            best = m;
            best_k = k;
        }
        if (k < count) pre += cost(k);
    }
    return best_k;
}

}  // namespace gsplat_tt::sort_split
