// SPDX-License-Identifier: Apache-2.0
//
// Task #146: live-microblock mask from the packed bf16 T tile (1024 values,
// row-major device raster, two bf16 per word, low half = even index). Bit m is
// set iff microblock vector m has a T with eps <= T (fp32 compare, NaN and
// negatives never count). Vector V = 2*(r/2) + (c&1) for index r*32 + c, so the
// 32 words [32q, 32q+32) hold exactly vector 2q (low halves) and 2q+1 (high).
//
// blend_t_live_ref is the original per-value max loop (kept as the oracle);
// blend_t_live_fast is bit-identical (tests/unit/test_blend_t_live.cpp): with
// eps a positive non-NaN, "some accepted fb >= eps" is "some half h in
// [ceil(eps / 2^16), 0x7F80]", one unsigned range compare per half, and a row
// pair stops scanning once both of its vectors are live.
#pragma once

#include <cstdint>

#include "../dataflow/dm_fp32.h"

inline uint32_t blend_t_live_ref(const volatile uint32_t* w, uint32_t eps_bits) {
    uint32_t mbmax[32];
    for (uint32_t m = 0; m < 32u; ++m) {
        mbmax[m] = 0u;
    }
    for (uint32_t t = 0; t < 1024u; ++t) {
        const uint32_t word = w[t >> 1];
        const uint32_t half = (t & 1u) ? (word >> 16) : (word & 0xffffu);
        const uint32_t fb = half << 16;
        const uint32_t r = t >> 5;
        const uint32_t c = t & 31u;
        const uint32_t V = (r & ~1u) | (c & 1u);
        if (fb > mbmax[V] && fb <= 0x7F800000u) {
            mbmax[V] = fb;
        }
    }
    uint32_t live = 0u;
    for (uint32_t m = 0; m < 32u; ++m) {
        if (dm_fp32::le(eps_bits, mbmax[m])) {
            live |= (1u << m);
        }
    }
    return live;
}

inline uint32_t blend_t_live_fast(const volatile uint32_t* w, uint32_t eps_bits) {
    // +0, negative or NaN eps: the general form (never on the default path).
    if (eps_bits == 0u || eps_bits > 0x7F800000u) {
        return blend_t_live_ref(w, eps_bits);
    }
    const uint32_t e = (eps_bits + 0xffffu) >> 16;  // 1 .. 0x7F80
    const uint32_t span = 0x7F80u - e;
    uint32_t live = 0u;
    for (uint32_t q = 0; q < 16u; ++q) {
        const volatile uint32_t* p = w + 32u * q;
        uint32_t lo = 0u, hi = 0u;
        for (uint32_t k = 0; k < 32u; k += 4u) {
            const uint32_t x0 = p[k], x1 = p[k + 1], x2 = p[k + 2], x3 = p[k + 3];
            lo |= (((x0 & 0xffffu) - e) <= span) | (((x1 & 0xffffu) - e) <= span) |
                  (((x2 & 0xffffu) - e) <= span) | (((x3 & 0xffffu) - e) <= span);
            hi |= (((x0 >> 16) - e) <= span) | (((x1 >> 16) - e) <= span) |
                  (((x2 >> 16) - e) <= span) | (((x3 >> 16) - e) <= span);
            if ((lo & hi) != 0u) {
                break;
            }
        }
        live |= (lo | (hi << 1)) << (2u * q);
    }
    return live;
}
