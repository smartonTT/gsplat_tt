// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Integer-only, bit-exact replacements for the two fp32 expressions the sort
// emit (sort_bin.cpp) evaluates when it packs the 32 B bucket record.
//
// The data movers have no FPU, so each float operation in the pack was a
// libgcc soft-float call: four UNORM16 conversions per gaussian and an
// int->float plus a subtract per mean coordinate per pair. Measured on
// yyzo-bh-07 (task #23) this soft-float work was ~12 ms of the 30.3 ms
// sort_bucket_emit makespan. These helpers compute the same IEEE results
// (round-to-nearest-even, identical bits) with integer arithmetic.
// tests/unit/test_emit_intpack.cpp checks them against hardware fp32.
//
// No tt-metal dependency, so the host test can include this header.

#pragma once

#include <cstdint>

namespace emit_intpack {

// Round a positive integer mantissa P (value P * 2^k) to 24 significant bits,
// round-to-nearest-even. Returns the rounded mantissa q (2^23 <= q < 2^24) and
// adds the dropped bit count to k.
inline uint32_t round24(uint64_t p, int32_t& k) {
    const int32_t len = 64 - __builtin_clzll(p);
    if (len <= 24) {
        k -= 24 - len;
        return static_cast<uint32_t>(p << (24 - len));
    }
    const int32_t s = len - 24;
    uint64_t q = p >> s;
    const uint64_t rem = p & ((1ull << s) - 1u);
    const uint64_t half = 1ull << (s - 1);
    if (rem > half || (rem == half && (q & 1u))) q++;
    k += s;
    if (q == (1ull << 24)) {
        q >>= 1;
        k += 1;
    }
    return static_cast<uint32_t>(q);
}

// Bits of fl(v * 65535.0f + 0.5f) truncated to uint32, with v <= 0 -> 0 and
// v >= 1 -> 65535: exactly what the soft-float pack computed. Only the product
// needs 64 bits (one mul + mulhu); the rest is 32-bit.
inline uint32_t unorm16(uint32_t u) {
    if (static_cast<int32_t>(u) <= 0) return 0u;  // v <= 0 (incl. -0)
    if (u >= 0x3f800000u) return 65535u;         // v >= 1
    const uint32_t ef = u >> 23;
    if (ef == 0u) return 0u;                     // denormal: v * 65535 + 0.5 < 1
    // v = m * 2^(ef-150). x = fl(v * 65535) = q * 2^k, q in [2^23, 2^24).
    const uint32_t m = (u & 0x7fffffu) | 0x800000u;
    const uint64_t p = static_cast<uint64_t>(m) * 65535u;  // in [2^39 - 2^23, 2^40)
    const uint32_t s = (p >> 39) ? 16u : 15u;               // bits dropped to keep 24
    uint32_t q = static_cast<uint32_t>(p >> s);
    const uint32_t rem = static_cast<uint32_t>(p) & ((1u << s) - 1u);
    const uint32_t half = 1u << (s - 1u);
    if (rem > half || (rem == half && (q & 1u))) q++;
    int32_t k = static_cast<int32_t>(ef) - 150 + static_cast<int32_t>(s);
    if (q == (1u << 24)) {
        q >>= 1;
        k++;
    }
    // y = fl(x + 0.5); result = floor(y). x < 65535 means k <= -8, and
    // k < -26 means x < 1/4, where floor(fl(x + 0.5)) = 0.
    if (k < -26) return 0u;
    uint32_t sum = q + (1u << (-1 - k));  // (x + 0.5) / 2^k, < 2^26
    int32_t k2 = k;
    if (sum >= (1u << 24)) {              // round to 24 bits (at most 2 dropped)
        const uint32_t d = (sum >= (1u << 25)) ? 2u : 1u;
        const uint32_t r = sum & ((1u << d) - 1u), h = 1u << (d - 1u);
        sum >>= d;
        if (r > h || (r == h && (sum & 1u))) sum++;
        k2 += static_cast<int32_t>(d);
    }
    return sum >> (-k2);  // k2 < 0 (y < 65536)
}

// Bits of fl(x - t) for fp32 x (bits xb) and an integer 0 <= t < 2^14,
// round-to-nearest-even: the soft-float `x - static_cast<float>(t)` the pack
// evaluated per pair (t = tile origin in px). 32-bit fast path for
// 4 <= |x| < 2^24 and t <= 1000 (all of a 1024 px frame); 64-bit otherwise.
inline uint32_t sub_int(uint32_t xb, uint32_t t) {
    if (t == 0u) return xb;
    const uint32_t ef = (xb >> 23) & 0xffu;
    const int32_t e = static_cast<int32_t>(ef) - 127;
    const bool neg = (xb >> 31) != 0u;
    const uint32_t m = (xb & 0x7fffffu) | 0x800000u;  // x = m * 2^(e-23)
    if (e >= 2 && e <= 23 && t <= 1000u) {
        const uint32_t f = static_cast<uint32_t>(23 - e);  // <= 21: m + (t << f) < 2^31
        const int32_t d = (neg ? -static_cast<int32_t>(m) : static_cast<int32_t>(m)) -
                          static_cast<int32_t>(t << f);
        if (d == 0) return 0u;
        const uint32_t sign = (d < 0) ? 0x80000000u : 0u;
        uint32_t a = (d < 0) ? static_cast<uint32_t>(-d) : static_cast<uint32_t>(d);
        const uint32_t len = 32u - static_cast<uint32_t>(__builtin_clz(a));
        int32_t k = -static_cast<int32_t>(f);
        if (len <= 24u) {
            a <<= (24u - len);
            k -= static_cast<int32_t>(24u - len);
        } else {
            const uint32_t sh = len - 24u;
            const uint32_t r = a & ((1u << sh) - 1u), h = 1u << (sh - 1u);
            a >>= sh;
            if (r > h || (r == h && (a & 1u))) a++;
            k += static_cast<int32_t>(sh);
            if (a == (1u << 24)) {
                a >>= 1;
                k++;
            }
        }
        return sign | (static_cast<uint32_t>(k + 23 + 127) << 23) | (a & 0x7fffffu);
    }
    if (e > 50) return xb;  // ulp(x) >= 2^28 > 2t: x - t rounds back to x
    if (ef == 0u || e < -26) {
        // |x| < 2^-26 is below half the fp32 spacing next to any t >= 1: -t.
        int32_t k = 0;
        const uint32_t q = round24(t, k);
        return 0x80000000u | (static_cast<uint32_t>(k + 23 + 127) << 23) | (q & 0x7fffffu);
    }
    int64_t d;
    int32_t k;
    if (e >= 23) {
        d = static_cast<int64_t>(static_cast<uint64_t>(m) << (e - 23)) * (neg ? -1 : 1) -
            static_cast<int64_t>(t);
        k = 0;
    } else {
        const int32_t f = 23 - e;  // <= 49, so t << f < 2^63
        d = static_cast<int64_t>(m) * (neg ? -1 : 1) - (static_cast<int64_t>(t) << f);
        k = -f;
    }
    if (d == 0) return 0u;  // x - x = +0 under round-to-nearest
    const uint32_t sign = (d < 0) ? 0x80000000u : 0u;
    const uint64_t a = (d < 0) ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);
    const uint32_t q = round24(a, k);  // value = q * 2^k, q in [2^23, 2^24)
    return sign | (static_cast<uint32_t>(k + 23 + 127) << 23) | (q & 0x7fffffu);
}

}  // namespace emit_intpack
