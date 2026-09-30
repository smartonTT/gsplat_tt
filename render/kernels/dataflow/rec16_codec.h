// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// 16 B per-pair bucket record (R16, task #23): integer-only encode/decode.
//
// sort_bin.cpp (the emit) packs one record per kept (gaussian, tile) pair;
// sort_subchunk_materialize.cpp expands it back to the 32 B slab record the
// cull/blend kernels read. Everything here is plain integer C++ with no
// tt-metal dependency, so tests/unit/test_rec16_codec.cpp checks it on the host.
// The data movers have no FPU: each float op in the old pack was a libgcc
// soft-float call, so the encode side avoids float arithmetic entirely.
//
// Record words (little-endian u32):
//   [0] half A | half B << 16        conic (power = A dx^2 + B dx dy + C dy^2) x 2^10
//   [1] half C | half mx << 16       mx, my: mean relative to the tile origin, px
//   [2] half my | unorm16 op << 16
//   [3] unorm11 r | unorm11 g << 11 | unorm10 b << 22
// The x2^10 (tile^2 units) conic scale keeps |A|,|C| of splats up to ~2900 px
// sigma in the half normal range; unscaled, anything past ~90 px sigma would
// go denormal. The depth key is not in the record (materialize reads it from
// the emit's keys layout).
//
// Every half written here is normal: magnitudes below the normal range clamp to
// the smallest normal and magnitudes above it saturate, so decode is a single
// exponent rebias with no special cases.

#pragma once

#include <cstdint>

namespace rec16 {

constexpr int32_t kConicScaleLog2 = 10;  // 32x32 px tile => x1024

// fp32 bits -> half bits of x * 2^scale_log2, round half away from zero.
inline uint32_t f32_to_h16(uint32_t u, int32_t scale_log2) {
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((u >> 23) & 0xffu) - 112 + scale_log2;
    if (e <= 0) return sign | 0x0400u;
    uint32_t h = (static_cast<uint32_t>(e) << 10) | ((u >> 13) & 0x3ffu);
    h += (u >> 12) & 1u;
    if (h >= 0x7c00u) h = 0x7bffu;
    return sign | h;
}

// fp32 bits -> round(x * 2^8), i.e. 1/256 px fixed point. Valid for |x| < 2^22.
inline int32_t f32_to_fx8(uint32_t u) {
    const int32_t e = static_cast<int32_t>((u >> 23) & 0xffu) - 127;
    if (e < -9) return 0;
    const uint32_t m = (u & 0x7fffffu) | 0x800000u;  // x = m * 2^(e-23)
    const int32_t sh = e - 15;                         // x * 2^8 = m * 2^sh
    uint32_t a;
    if (sh >= 0) {
        a = m << (sh > 7 ? 7 : sh);
    } else {
        a = (m + (1u << (-sh - 1))) >> (-sh);
    }
    return (u >> 31) ? -static_cast<int32_t>(a) : static_cast<int32_t>(a);
}

// 1/256 px fixed point -> half bits of v / 256, round half up.
inline uint32_t fx8_to_h16(int32_t v) {
    const uint32_t sign = (v < 0) ? 0x8000u : 0u;
    const uint32_t a = (v < 0) ? static_cast<uint32_t>(-v) : static_cast<uint32_t>(v);
    if (a == 0u) return sign | 0x0400u;
    const uint32_t e = 31u - static_cast<uint32_t>(__builtin_clz(a));  // a in [2^e, 2^(e+1))
    uint32_t h;
    if (e >= 10u) {
        const uint32_t s = e - 10u;
        h = ((e + 7u) << 10) | ((a >> s) & 0x3ffu);
        if (s != 0u) h += (a >> (s - 1u)) & 1u;
        if (h >= 0x7c00u) h = 0x7bffu;
    } else {
        h = ((e + 7u) << 10) | ((a << (10u - e)) & 0x3ffu);
    }
    return sign | h;
}

// fp32 bits of v -> round(clamp(v, 0, 1) * (2^n - 1)), n <= 16: exact product,
// one rounding.
inline uint32_t f32_to_unorm(uint32_t u, uint32_t n) {
    if (static_cast<int32_t>(u) <= 0) return 0u;  // +0 and negatives
    if (u >= 0x3f800000u) return (1u << n) - 1u;  // >= 1.0
    const uint32_t sh = 150u - (u >> 23);         // v = m * 2^-sh, sh >= 24
    if (sh > 48u) return 0u;
    const uint32_t m = (u & 0x7fffffu) | 0x800000u;
    const uint64_t p = (static_cast<uint64_t>(m) << n) - m;  // m * (2^n - 1)
    return static_cast<uint32_t>((p + (1ull << (sh - 1u))) >> sh);
}

inline uint32_t f32_to_unorm16(uint32_t u) { return f32_to_unorm(u, 16u); }

// Record word [3] from the fp32 colors: UNORM 11/11/10 (expand() widens them
// back to UNORM16 by bit replication, i.e. q * 65535 / (2^n - 1) to within 1).
inline uint32_t pack_color_word(uint32_t r_bits, uint32_t g_bits, uint32_t b_bits) {
    return f32_to_unorm(r_bits, 11u) | (f32_to_unorm(g_bits, 11u) << 11) |
           (f32_to_unorm(b_bits, 10u) << 22);
}

// Half bits from the encoders above -> fp32 bits of h * 2^-scale_log2 (exact).
inline uint32_t h16_to_f32(uint32_t h, uint32_t scale_log2) {
    return ((h & 0x8000u) << 16) | (((h & 0x7fffu) << 13) + ((112u - scale_log2) << 23));
}

// Record -> the 32 B slab words: [0..2] fp32 A,B,C  [3] key  [4,5] fp32 mean
// [6] unorm16 op | r << 16  [7] unorm16 g | b << 16 (colors re-expanded by bit
// replication, so 0 and full scale are exact).
inline void expand(const uint32_t w[4], uint32_t key, uint32_t out[8]) {
    const uint32_t r11 = w[3] & 0x7ffu, g11 = (w[3] >> 11) & 0x7ffu, b10 = w[3] >> 22;
    out[0] = h16_to_f32(w[0] & 0xffffu, kConicScaleLog2);
    out[1] = h16_to_f32(w[0] >> 16, kConicScaleLog2);
    out[2] = h16_to_f32(w[1] & 0xffffu, kConicScaleLog2);
    out[3] = key;
    out[4] = h16_to_f32(w[1] >> 16, 0u);
    out[5] = h16_to_f32(w[2] & 0xffffu, 0u);
    out[6] = (w[2] >> 16) | (((r11 << 5) | (r11 >> 6)) << 16);
    out[7] = ((g11 << 5) | (g11 >> 6)) | (((b10 << 6) | (b10 >> 4)) << 16);
}

}  // namespace rec16
