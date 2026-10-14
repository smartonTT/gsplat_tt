// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// subtile_route.h — task #738 (B1 full): which sub-tile lists of a split tile
// a record goes to. Header-only, no tt-metal types: the sort bin movers
// (sort_bin_onelaunch.cpp count + emit) and tests/unit/test_subtile_route.cpp
// use the same code.
//
// Why the sort bin and not the K2 emit (PLAN.md §4 put it in pfwc_fuse.h): the
// K2 pair index space is fixed by the tile-box areas (lofs prefix), one pair per
// (gaussian, tile), and the K2 sees only the tile box, no conic or opacity. The
// bin movers already load the record's conic, opacity and mean (pack_invariants),
// so they can count and scatter one record per sub-tile the record may touch.
//
// Lossless rule: a record goes to every sub-tile whose microblock mask meets
// box_mask(). box_mask() is a superset of the microblocks the device cull
// (microblock_band_cull_compute.cpp, model band_keep_f32 in
// opt/golden-check/cull_mask_check.py) can keep: the cull keeps microblock m iff
// the ellipse m2 <= thr meets m's pixel-centre box, thr = 2 ln(op / floor) +
// kThrMargin (minus a log underestimate). That ellipse lies in |dx| <= sqrt(thr
// Sxx), |dy| <= sqrt(thr Syy) (S = conic^-1); box_mask() takes an upper bound
// of thr and compares squares (no sqrt, no log: ln comes from the fp32 exponent
// and mantissa, ln(1 + m) <= ln2 (m + 0.0861)). A degenerate record (conic not
// positive definite, non-finite words) gets the full mask.
//
// Record words (== pack_rec): A = -ca/2, B = -cb, C = -cc/2 (fp32), tile-local
// mean mx, my (fp32, pixels from the PARENT tile's origin), opacity UNORM16.
// Microblock m = (my << 2) | mx: 8 px columns mx 0..3, 4 px rows my 0..7.

#pragma once

#include <cstdint>

namespace subtile_route {

constexpr uint32_t kFullMask = 0xFFFFFFFFu;
// thr slack over the device's 2 ln(op/floor) + 0.05 (kThrMargin): covers fp32
// rounding of the device log and of the products below.
constexpr float kThrMargin = 0.05f;
constexpr float kThrSlack = 0.02f;
constexpr float kRelSlack = 1.0e-3f;

inline uint32_t bits_of(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}
inline float float_of(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

// An upper bound of ln(x) for finite x > 0 (within 0.06 of ln x for normal x).
inline float ln_upper(float x) {
    const uint32_t u = bits_of(x);
    const int32_t e = static_cast<int32_t>((u >> 23) & 0xFFu) - 127;
    const float m = float_of((u & 0x007FFFFFu) | 0x3F800000u) - 1.0f;  // [0, 1)
    return 0.69314720f * (static_cast<float>(e) + m + 0.0862f);
}

// Squared distance from v to the interval [lo, hi] (0 inside).
inline float dist2(float v, float lo, float hi) {
    const float d = v < lo ? lo - v : (v > hi ? v - hi : 0.0f);
    return d * d;
}

// Superset of the microblocks the cull keeps for this record in its parent
// tile. inv_floor = 1 / contrib floor (the cull's floor for this view).
inline uint32_t box_mask(uint32_t a_bits, uint32_t b_bits, uint32_t c_bits, uint32_t mx_bits,
                         uint32_t my_bits, uint32_t q16, float inv_floor) {
    if (q16 == 0u) return 0u;  // op 0: thr = -inf, the cull keeps nothing
    const float ca = -2.0f * float_of(a_bits);
    const float cb = -float_of(b_bits);
    const float cc = -2.0f * float_of(c_bits);
    const float mx = float_of(mx_bits);
    const float my = float_of(my_bits);
    const float det = ca * cc - cb * cb;
    // NaN fails every comparison below, so it takes the full mask too.
    if (!(ca > 0.0f) || !(cc > 0.0f) || !(det > 0.0f) || !(mx - mx == 0.0f) || !(my - my == 0.0f))
        return kFullMask;
    const float ratio = static_cast<float>(q16) * (inv_floor * (1.0f / 65535.0f));
    if (!(ratio > 0.0f) || !(ratio < 3.0e38f)) return kFullMask;
    const float thr = (2.0f * ln_upper(ratio) + (kThrMargin + kThrSlack)) * (1.0f + kRelSlack);
    if (!(thr > 0.0f)) return 0u;
    // thr * Sxx and thr * Syy, Sxx = cc / det, Syy = ca / det.
    const float rx2 = thr * cc / det * (1.0f + kRelSlack);
    const float ry2 = thr * ca / det * (1.0f + kRelSlack);
    if (!(rx2 < 3.0e38f) || !(ry2 < 3.0e38f)) return kFullMask;
    uint32_t cols = 0, rows = 0;
    for (uint32_t c = 0; c < 4u; ++c)
        if (dist2(mx, 8.0f * c + 0.5f, 8.0f * c + 7.5f) <= rx2) cols |= 1u << c;
    for (uint32_t r = 0; r < 8u; ++r)
        if (dist2(my, 4.0f * r + 0.5f, 4.0f * r + 3.5f) <= ry2) rows |= 1u << r;
    uint32_t m = 0;
    for (uint32_t r = 0; r < 8u; ++r)
        if (rows & (1u << r)) m |= cols << (r << 2);
    return m;
}

// Sub-tile k of a split tile is a target iff its mask meets the record's box
// mask. Returns a bit per sub-tile (n_sub <= 16).
template <typename MaskOf>
inline uint32_t targets(uint32_t box, uint32_t n_sub, MaskOf mask_of) {
    uint32_t t = 0;
    for (uint32_t k = 0; k < n_sub; ++k)
        if (mask_of(k) & box) t |= 1u << k;
    return t;
}

}  // namespace subtile_route
