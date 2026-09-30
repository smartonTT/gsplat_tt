// Task #24 (R8): host helpers of the atomic fixed-capacity bucket layout
// (sort_bin_atomic.cpp). Header-only so the unit test can use them.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace gsplat_tt::sort_atomic {

// Canonical record order of one tile: the cores' chunks in core order, from
// the chunk table entries (count << 16 | first slot) of cores 0..n-1 for the
// tile. Mirrors the key/slot fill of sort_subchunk_materialize.cpp.
inline std::vector<uint32_t> canonical_order(const std::vector<uint32_t>& entries, uint32_t n) {
    std::vector<uint32_t> idx;
    idx.reserve(n);
    for (const uint32_t e : entries) {
        const uint32_t len = e >> 16;
        const uint32_t base = e & 0xFFFFu;
        for (uint32_t k = 0; k < len && idx.size() < n; ++k) idx.push_back(base + k);
    }
    return idx;
}

// Materialize work: every non-empty tile is ONE whole-tile item (sc 0), since
// its full record set is in its bucket. LPT over record counts; tiles of more
// than ov_cap records extract keys and fill each subchunk from chunked reads
// (about twice the per-record cost). Returns the tile list of each core.
inline std::vector<std::vector<uint32_t>> lpt_whole_tiles(const std::vector<int64_t>& counts,
                                                          uint32_t num_tiles, uint32_t num_cores,
                                                          uint32_t ov_cap) {
    struct Item { uint32_t tile; uint64_t cost; };
    std::vector<Item> items;
    items.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const uint32_t cnt = static_cast<uint32_t>(counts[t]);
        if (cnt == 0u) continue;
        items.push_back({t, static_cast<uint64_t>(cnt) * (cnt > ov_cap ? 2u : 1u)});
    }
    std::stable_sort(items.begin(), items.end(),
                     [](const Item& a, const Item& b) { return a.cost > b.cost; });
    std::vector<std::vector<uint32_t>> per_core(num_cores);
    std::vector<uint64_t> load(num_cores, 0);
    for (const auto& it : items) {
        const uint32_t c = static_cast<uint32_t>(
            std::distance(load.begin(), std::min_element(load.begin(), load.end())));
        per_core[c].push_back(it.tile);
        load[c] += it.cost;
    }
    return per_core;
}

}  // namespace gsplat_tt::sort_atomic
