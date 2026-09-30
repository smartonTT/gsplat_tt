// SPDX-License-Identifier: Apache-2.0
//
// Integer-only, bit-exact fp32 helpers for the data-mover RISCs (BRISC/NCRISC
// have no FPU: every fp32 add/compare/convert there is a libgcc soft-float call,
// ~85-90 cycles for __addsf3/__mulsf3 and ~20-30 for a compare, docs/hw-ceilings.md).
// Operands and results are fp32 BIT PATTERNS. Every helper reproduces IEEE-754
// binary32 round-to-nearest-even exactly (the bits libgcc returns); the ones
// that return bool report false outside their fast range so the caller can keep
// the float expression for those rare inputs. Checked against native fp32 by
// tests/unit/test_dm_fp32.cpp.
#pragma once

#include <cstdint>

namespace dm_fp32 {

constexpr uint32_t SIGN = 0x80000000u;

inline bool is_nan(uint32_t x) { return (x & 0x7FFFFFFFu) > 0x7F800000u; }

// Total-order key: unsigned compare of keys == fp32 compare for non-NaN, except
// that -0 and +0 get different keys (callers handle the both-zero case).
inline uint32_t key(uint32_t x) { return (x & SIGN) ? ~x : (x | SIGN); }

// a < b, a <= b with IEEE semantics (NaN -> false, -0 == +0).
inline bool lt(uint32_t a, uint32_t b) {
    if (is_nan(a) || is_nan(b)) return false;
    if (((a | b) & 0x7FFFFFFFu) == 0u) return false;
    return key(a) < key(b);
}
inline bool le(uint32_t a, uint32_t b) {
    if (is_nan(a) || is_nan(b)) return false;
    if (((a | b) & 0x7FFFFFFFu) == 0u) return true;
    return key(a) <= key(b);
}

// *out = bits of fl(a + b). Fast range: both operands normal (not zero,
// subnormal, inf or NaN) and the result normal or an exact +0. Returns false
// otherwise. 32-bit only (RV32 has no cheap 64-bit ops): the mantissas carry 6
// guard bits and a sticky bit.
inline bool add(uint32_t a, uint32_t b, uint32_t* out) {
    uint32_t ax = a & 0x7FFFFFFFu, bx = b & 0x7FFFFFFFu;
    if (ax < bx) {
        const uint32_t t = a; a = b; b = t;
        const uint32_t tx = ax; ax = bx; bx = tx;
    }
    const uint32_t ea = ax >> 23, eb = bx >> 23;
    if (ea - 1u >= 254u || eb - 1u >= 254u) return false;
    const uint32_t ma = ((ax & 0x7FFFFFu) | 0x800000u) << 6;  // leading bit 29
    uint32_t mb = ((bx & 0x7FFFFFu) | 0x800000u) << 6;
    const uint32_t d = ea - eb;
    if (d >= 31u) {
        mb = 1u;
    } else if (d) {
        mb = (mb >> d) | ((mb & ((1u << d) - 1u)) != 0u);
    }
    int32_t e = static_cast<int32_t>(ea);
    uint32_t m;
    if (((a ^ b) & SIGN) == 0u) {
        m = ma + mb;
        if (m >> 30) {
            m = (m >> 1) | (m & 1u);
            e++;
        }
    } else {
        m = ma - mb;
        if (m == 0u) {  // x - x = +0 under RNE
            *out = 0u;
            return true;
        }
        const uint32_t lz = static_cast<uint32_t>(__builtin_clz(m)) - 2u;
        m <<= lz;
        e -= static_cast<int32_t>(lz);
        if (e <= 0) return false;
    }
    const uint32_t rem = m & 0x3Fu;
    m >>= 6;
    if (rem > 0x20u || (rem == 0x20u && (m & 1u))) {
        m++;
        if (m >> 24) {
            m >>= 1;
            e++;
        }
    }
    if (e >= 255) return false;
    *out = (a & SIGN) | (static_cast<uint32_t>(e) << 23) | (m & 0x7FFFFFu);
    return true;
}

// fl(a - b) < w, same result as the soft-float expression for every input.
inline bool sub_lt(uint32_t a, uint32_t b, uint32_t w) {
    uint32_t r;
    if (add(a, b ^ SIGN, &r)) return lt(r, w);
    float fa, fb, fw;
    __builtin_memcpy(&fa, &a, 4);
    __builtin_memcpy(&fb, &b, 4);
    __builtin_memcpy(&fw, &w, 4);
    return fa - fb < fw;
}

// (int)(x * 2^-s) for fp32 bits x and s < 32, i.e. the int conversion of
// x * inv_tsf where inv_tsf = 1/2^s is exact. Fast range: |x| < 2^31 and not
// NaN/inf. Returns false otherwise.
inline bool trunc_shr(uint32_t x, uint32_t s, int32_t* out) {
    const uint32_t e = (x >> 23) & 0xFFu;
    if (e >= 158u) return false;  // |x| >= 2^31, inf, NaN
    uint32_t v = 0u;
    if (e >= 127u) {
        const uint32_t m = (x & 0x7FFFFFu) | 0x800000u;
        v = (e <= 150u ? (m >> (150u - e)) : (m << (e - 150u))) >> s;  // floor(|x|) >> s
    }
    *out = (x & SIGN) ? -static_cast<int32_t>(v) : static_cast<int32_t>(v);
    return true;
}

// (int)((a + b) * inv) for fp32 bits a, b where inv == 2^-s exactly; pass
// s >= 32 when it is not (then always the float expression). Integer fast path,
// the float expression for inputs outside it.
inline int32_t add_mul_pow2_to_int(uint32_t a, uint32_t b, uint32_t s, float inv) {
    uint32_t r;
    int32_t v;
    if (s < 32u && add(a, b, &r) && trunc_shr(r, s, &v)) return v;
    float fa, fb;
    __builtin_memcpy(&fa, &a, 4);
    __builtin_memcpy(&fb, &b, 4);
    return static_cast<int32_t>((fa + fb) * inv);
}

// s with 2^-s == 1/(float)n exactly (n a power of two below 2^24), else 32.
inline uint32_t pow2_shift(uint32_t n) {
    if (n == 0u || (n & (n - 1u)) || n >= (1u << 24)) return 32u;
    return static_cast<uint32_t>(__builtin_ctz(n));
}

}  // namespace dm_fp32
