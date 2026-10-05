// Bit-exactness check for render/kernels/dataflow/dm_fp32.h against native
// fp32 (IEEE RNE with subnormals == what libgcc soft-float returns on the data
// movers). Standalone (the sweeps are too slow for the Catch suite):
//
//   c++ -O2 -ffp-contract=off -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_dm_fp32.cpp -o /tmp/test_dm_fp32 && /tmp/test_dm_fp32
//
// Must not be built with -ffast-math / FTZ (subnormal and NaN semantics).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "dm_fp32.h"

static float f(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static uint32_t u(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

static uint64_t bad = 0, checks = 0, fast = 0;
static void fail(const char* what, uint32_t a, uint32_t b, uint32_t got, uint32_t ref) {
    if (bad++ < 20) std::printf("%s mismatch a=%08x b=%08x got=%08x ref=%08x\n", what, a, b, got, ref);
}

static void check_pair(uint32_t a, uint32_t b) {
    checks++;
    if (dm_fp32::lt(a, b) != (f(a) < f(b))) fail("lt", a, b, dm_fp32::lt(a, b), f(a) < f(b));
    if (dm_fp32::le(a, b) != (f(a) <= f(b))) fail("le", a, b, dm_fp32::le(a, b), f(a) <= f(b));
    uint32_t r;
    if (dm_fp32::add(a, b, &r)) {
        fast++;
        const uint32_t ref = u(f(a) + f(b));
        if (r != ref) fail("add", a, b, r, ref);
    }
    // sub_lt against a few thresholds, incl. the gather's img_w / img_h.
    static const uint32_t ws[] = {u(1024.0f), u(0.0f), u(-0.0f), a, b, u(1e-40f)};
    for (uint32_t w : ws)
        if (dm_fp32::sub_lt(a, b, w) != (f(a) - f(b) < f(w))) fail("sub_lt", a, b, w, 0);
    // sub_lt_pos: the gather's fast path, valid for b > 0 finite.
    if (dm_fp32::lt(0u, b) && b < 0x7F800000u)
        for (uint32_t w : ws)
            if (dm_fp32::sub_lt_pos(a, b, w) != (f(a) - f(b) < f(w))) fail("sub_lt_pos", a, b, w, 0);
    // tile_assign's (int)((a + b) * (1/32.f)), fast path only (the fallback is
    // the same expression) and within int range on both sides.
    {
        uint32_t r; int32_t v;
        if (dm_fp32::add(a, b, &r) && dm_fp32::trunc_shr(r, 5u, &v) &&
            dm_fp32::add_mul_pow2_to_int(a, b, 5u, 1.0f / 32.0f) != static_cast<int32_t>((f(a) + f(b)) * (1.0f / 32.0f)))
            fail("add_mul_pow2", a, b, static_cast<uint32_t>(v), 0);
    }
    // the gather's `mx + rx > 0` rewritten as `-rx < mx`
    if (dm_fp32::lt(b ^ dm_fp32::SIGN, a) != (f(a) + f(b) > 0.0f)) fail("addgt0", a, b, 0, 0);
}

int main() {
    // trunc_shr: every fp32 pattern, the production s = 5 plus s = 0, 4.
    for (uint32_t s : {5u, 0u, 4u}) {
        const float inv = 1.0f / static_cast<float>(1u << s);
        for (uint64_t i = 0; i <= 0xFFFFFFFFull; i++) {
            const uint32_t x = static_cast<uint32_t>(i);
            int32_t got;
            if (!dm_fp32::trunc_shr(x, s, &got)) continue;
            checks++;
            const int32_t ref = static_cast<int32_t>(f(x) * inv);
            if (got != ref) fail("trunc_shr", x, s, static_cast<uint32_t>(got), static_cast<uint32_t>(ref));
        }
    }
    std::printf("trunc_shr done, bad=%llu\n", (unsigned long long)bad);

    // add/lt/le/sub_lt: every fp32 pattern `a` against a set of fixed `b`s
    // (pixel-scale radii/means, powers of two, specials), both orders.
    const uint32_t bs[] = {u(0.0f), u(-0.0f), u(1.0f), u(-1.0f), u(3.3f), u(-3.3f), u(32.0f),
                           u(1024.0f), u(-1024.0f), u(511.75f), u(-0.0001f), u(1e-38f),
                           0x00000001u, 0x807FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                           0x7F7FFFFFu, 0xFF7FFFFFu, u(1.5e-5f)};
    for (uint32_t b : bs)
        for (uint64_t i = 0; i <= 0xFFFFFFFFull; i += 61) {
            check_pair(static_cast<uint32_t>(i), b);
            check_pair(b, static_cast<uint32_t>(i));
        }
    std::printf("fixed-b sweep done, bad=%llu\n", (unsigned long long)bad);

    // Random pairs: raw bits, same/near exponent, and near-cancellation.
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint64_t n = 0; n < 1000000000ull; n++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        uint32_t a = static_cast<uint32_t>(x), b = static_cast<uint32_t>(x >> 32);
        switch (n & 3) {
            case 0: break;                                              // raw
            case 1: b = (b & 0x807FFFFFu) | ((a & 0x7F800000u) + ((x >> 60) << 23)); break;  // exp gap 0..15
            case 2: b = (a ^ (x >> 58 & 1 ? dm_fp32::SIGN : 0u)) + static_cast<uint32_t>((x >> 50) & 0xFF) - 128u; break;  // near +-a
            case 3: a = u(static_cast<float>((x >> 40) & 0xFFFFF) / 256.0f - 1500.0f);  // pixel means
                    b = u(static_cast<float>((x >> 20) & 0xFFFFF) / 4096.0f);          // radii
                    if (x & 1) b ^= dm_fp32::SIGN;
                    break;
        }
        check_pair(a, b);
    }
    std::printf("checks=%llu add fast-path=%llu mismatches=%llu\n", (unsigned long long)checks,
                (unsigned long long)fast, (unsigned long long)bad);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
