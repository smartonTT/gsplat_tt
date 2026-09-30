// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gather_visible's visibility predicate on fp32 bits (data movers have no FPU).
// Header-only so tests/unit/test_gather_visible_mask.cpp can check it on host.
#pragma once

#include <cstdint>

#include "dm_fp32.h"

namespace gather_pred {

// The visibility test on fp32 bits, with no libgcc soft-float call (BRISC has
// no FPU): same result for every input, incl. NaN/inf/-0, as
//   !(tz <= k_near || op < min_opacity) &&
//   mx + rx > 0 && mx - rx < img_w && my + ry > 0 && my - ry < img_h &&
//   rx > 0 && ry > 0 && rx <= max_radius && ry <= max_radius
// A rounded fp32 sum is > 0 iff the exact sum is, so `mx + rx > 0` is exactly
// `-rx < mx`; the two subtractions go through dm_fp32::sub_lt (checked by
// tests/unit/test_dm_fp32.cpp). Cheap compares first to skip the subtractions.
inline bool visible_bits(uint32_t tz, uint32_t op, uint32_t mx, uint32_t my, uint32_t rx,
                         uint32_t ry, uint32_t k_near, uint32_t min_opacity, uint32_t img_w,
                         uint32_t img_h, uint32_t max_radius) {
    using namespace dm_fp32;
    if (le(tz, k_near) || lt(op, min_opacity)) return false;
    return lt(0u, rx) && lt(0u, ry) && le(rx, max_radius) && le(ry, max_radius) &&
           lt(rx ^ SIGN, mx) && lt(ry ^ SIGN, my) && sub_lt(mx, rx, img_w) &&
           sub_lt(my, ry, img_h);
}

// visible_bits for element il of the staged tiles, loading each stream only
// when the predicate still needs it (most Gaussians fail on depth or on the
// image bounds). Returns exactly visible_bits(...).
inline bool visible_at(volatile uint32_t* p_dep, volatile uint32_t* p_op,
                       volatile uint32_t* p_m2x, volatile uint32_t* p_m2y,
                       volatile uint32_t* p_rx, volatile uint32_t* p_ry, uint32_t il,
                       uint32_t k_near, uint32_t min_opacity, uint32_t img_w,
                       uint32_t img_h, uint32_t max_radius) {
    using namespace dm_fp32;
    if (le(p_dep[il], k_near)) return false;
    if (lt(p_op[il], min_opacity)) return false;
    const uint32_t rx = p_rx[il];
    if (!(lt(0u, rx) && le(rx, max_radius))) return false;
    const uint32_t ry = p_ry[il];
    if (!(lt(0u, ry) && le(ry, max_radius))) return false;
    const uint32_t mx = p_m2x[il];
    if (!lt(rx ^ SIGN, mx)) return false;
    const uint32_t my = p_m2y[il];
    if (!lt(ry ^ SIGN, my)) return false;
    return sub_lt_pos(mx, rx, img_w) && sub_lt_pos(my, ry, img_h);  // rx, ry > 0
}

}  // namespace gather_pred
