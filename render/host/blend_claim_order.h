// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// blend_claim_order.h — task #188 (docs/blend-tail-t183): blend per-core tile
// lists dealt round-robin over the non-empty tiles in record-count-descending
// order (ties -> higher tile id): core c gets ranks c, c+n, c+2n, ... The blend
// reader's rank interleave (claim i -> rank i/n, core i%n) then claims tiles in
// exact global descending order. With late claim each core takes the next tile
// only when it has a free ring slot, so no LPT plan is needed.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace gsplat_tt::blend_order {

inline void desc_round_robin(
    const std::vector<int64_t>& counts, uint32_t num_tiles, uint32_t num_cores,
    std::vector<uint32_t>& flat_tile_ids, std::vector<uint32_t>& per_core_offset,
    std::vector<uint32_t>& per_core_count) {
    std::vector<std::pair<uint32_t, uint32_t>> cost_id;
    cost_id.reserve(num_tiles);
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t c = static_cast<uint32_t>(counts[t]);
        if (c > 0) cost_id.emplace_back(c, t);
    }
    std::sort(cost_id.begin(), cost_id.end(), std::greater<>());

    const uint32_t m = static_cast<uint32_t>(cost_id.size());
    flat_tile_ids.clear();
    flat_tile_ids.reserve(m);
    per_core_offset.assign(num_cores, 0);
    per_core_count.assign(num_cores, 0);
    for (uint32_t c = 0; c < num_cores; c++) {
        per_core_offset[c] = static_cast<uint32_t>(flat_tile_ids.size());
        for (uint32_t r = c; r < m; r += num_cores) flat_tile_ids.push_back(cost_id[r].second);
        per_core_count[c] = static_cast<uint32_t>(flat_tile_ids.size()) - per_core_offset[c];
    }
}

}  // namespace gsplat_tt::blend_order
