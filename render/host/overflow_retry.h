// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// overflow_retry.h — task #270: a tile over the sort bucket capacity
// (sort_onelaunch::kTileCap = 32768 records) no longer fails the frame.
// render_view re-renders the view with a coarser contribution floor until the
// largest tile fits. The floor applies to every pair of the frame (pfwc
// precull, tile_assign cull, bucket cull), so the result is a uniform,
// seam-free accuracy step for that frame only; views that fit never take this
// path, so the default bench is unchanged.
//
// Model: a Gaussian reaches a tile while 2 ln(o / floor) >= d^2, so a tile's
// record count grows roughly with L = ln(1 / floor). The next floor scales L
// by 0.9 * cap / max_n, and at least halves the floor so the ladder ends
// quickly. Header-only and tt-metal free (tests/unit/test_overflow_retry.cpp).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gsplat_tt::overflow_retry {

// Retries per view. Each retry at least doubles the floor, so 8 retries take
// the bench floor 1/255 past 1/2; no real tile overflows that.
inline constexpr int kMaxRetries = 8;
// Largest floor the ladder reaches.
inline constexpr float kMaxFloor = 0.5f;

// The next contribution floor after a frame whose largest tile held `max_n`
// records against the bucket capacity `cap` (max_n > cap).
inline float next_floor(float floor, uint32_t max_n, uint32_t cap) {
    floor = std::clamp(floor, 1.0f / 1048576.0f, kMaxFloor);
    const double L = std::log(1.0 / static_cast<double>(floor));
    const double ratio = 0.9 * static_cast<double>(cap) / static_cast<double>(std::max(max_n, cap));
    const double by_model = std::exp(-L * ratio);
    const double next = std::max(by_model, 2.0 * static_cast<double>(floor));
    return static_cast<float>(std::min(next, static_cast<double>(kMaxFloor)));
}

// Task #284: before any coarser floor, the sort bucket grows from `cap` to
// `big` records (same floor, full quality). Returns the grown capacity, or 0
// when growing does not help: already grown, the tile is over `big`, or the
// largest tile `max_n` is over `limit` (GSPLAT_TT_TEST_TILE_CAP), which a
// bigger bucket does not lift.
inline uint32_t grown_tile_cap(uint32_t cap, uint32_t big, uint32_t max_n, uint32_t limit) {
    if (cap >= big || max_n > big || max_n > limit) return 0u;
    return big;
}

}  // namespace gsplat_tt::overflow_retry
