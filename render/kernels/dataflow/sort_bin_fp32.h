// SPDX-License-Identifier: Apache-2.0
//
// Integer-only, bit-exact replacements for the two fp32 expressions in the
// sort_bin emit loop. NCRISC has no FPU, so each fp32 op there is a libgcc
// soft-float call (__mulsf3/__addsf3/__subsf3/__fixunssfsi/__floatunsisf/
// __lesf2/__gesf2); an ablation on yyzo-bh-07 (p100a) put them at ~15 ms of
// the ~27 ms sort_bucket_emit. Both helpers reproduce IEEE-754 binary32
// round-to-nearest-even exactly, i.e. the same bits libgcc returns, and return
// false for inputs outside their fast range so the caller can keep the float
// path for those (NaN, huge or tiny operands). Exhaustively checked against the
// float expressions by tests/unit/test_sort_bin_fp32.cpp.
#pragma once

#include <cstdint>

namespace sort_bin_fp32 {

// Round a positive integer `m` (bit length nb > 24) to 24 significant bits,
// nearest-even. Returns the 24-bit mantissa and adds the shift to *t.
inline uint32_t rne24(uint64_t m, uint32_t nb, int32_t* t) {
    const uint32_t s = nb - 24u;
    uint64_t r = m >> s;
    const uint64_t rem = m & ((uint64_t(1) << s) - 1u);
    const uint64_t half = uint64_t(1) << (s - 1u);
    if (rem > half || (rem == half && (r & 1u))) r++;
    int32_t sh = static_cast<int32_t>(s);
    if (r == (uint64_t(1) << 24)) { r >>= 1; sh++; }
    *t += sh;
    return static_cast<uint32_t>(r);
}

inline uint32_t bitlen64(uint64_t v) {
    const uint32_t hi = static_cast<uint32_t>(v >> 32);
    if (hi) return 64u - static_cast<uint32_t>(__builtin_clz(hi));
    const uint32_t lo = static_cast<uint32_t>(v);
    return lo ? 32u - static_cast<uint32_t>(__builtin_clz(lo)) : 0u;
}

// Bit-exact: v <= 0 ? 0 : v >= 1 ? 65535 : (uint32_t)(v * 65535.0f + 0.5f)
// with two separately rounded fp32 ops (no FMA), v given as its fp32 bits.
// Returns false only for NaN.
inline bool unorm16(uint32_t bits, uint32_t* out) {
    const uint32_t e = (bits >> 23) & 0xFFu;
    if (e == 0xFFu && (bits & 0x7FFFFFu)) return false;  // NaN
    if (bits >> 31) { *out = 0u; return true; }          // v <= 0 (incl. -0, -inf)
    if (bits >= 0x3F800000u) { *out = 65535u; return true; }  // v >= 1 (incl. +inf)
    if (e < 107u) { *out = 0u; return true; }            // v < 2^-20: v*65535+0.5 < 1
    // p = fl(v * 65535): v = M * 2^(e-150), exact product M*65535 is 39-40 bits.
    const uint64_t q = static_cast<uint64_t>((bits & 0x7FFFFFu) | 0x800000u) * 65535u;
    int32_t x = static_cast<int32_t>(e) - 150;
    const uint32_t r = rne24(q, (q >> 39) ? 40u : 39u, &x);  // p = r * 2^x, x in [-28,-8]
    // a = fl(p + 0.5): exact sum in units of 2^x fits 28 bits.
    const uint32_t S = r + (1u << static_cast<uint32_t>(-1 - x));
    uint32_t nb = 24u;
    while (S >> nb) nb++;
    uint32_t a = S;
    if (nb > 24u) a = rne24(S, nb, &x);
    // trunc(a * 2^x); x <= -8 here since a < 65536.
    const uint32_t rs = static_cast<uint32_t>(-x);
    *out = rs >= 32u ? 0u : (a >> rs);
    return true;
}

// Bit-exact fp32 bits of fl(a - (float)k) for an integer k < 2^23 (exact in
// fp32). Fast range: a normal with 2^-17 <= |a| < 2^24. Returns false outside.
inline bool sub_int(uint32_t abits, uint32_t k, uint32_t* out) {
    const uint32_t ea = (abits >> 23) & 0xFFu;
    if (ea < 110u || ea > 150u) return false;
    const uint32_t lsh = 150u - ea;  // a's ulp is 2^-lsh, lsh in [0,40]
    const int64_t A = static_cast<int64_t>((abits & 0x7FFFFFu) | 0x800000u);
    const int64_t D = ((abits >> 31) ? -A : A) - (static_cast<int64_t>(k) << lsh);
    if (D == 0) { *out = 0u; return true; }  // x - x = +0 under RNE
    const uint32_t sign = D < 0 ? 0x80000000u : 0u;
    const uint64_t mag = static_cast<uint64_t>(D < 0 ? -D : D);
    const uint32_t nb = bitlen64(mag);
    int32_t t = 0;
    uint32_t r;
    if (nb > 24u) {
        r = rne24(mag, nb, &t);
    } else {
        r = static_cast<uint32_t>(mag) << (24u - nb);
        t = static_cast<int32_t>(nb) - 24;
    }
    // value = r * 2^(t - lsh), r in [2^23, 2^24): biased exponent >= ea - 23 >= 87.
    const uint32_t E = static_cast<uint32_t>(t - static_cast<int32_t>(lsh) + 150);
    *out = sign | (E << 23) | (r & 0x7FFFFFu);
    return true;
}

}  // namespace sort_bin_fp32
