// Bit-exactness check for the record-decode helpers in
// render/kernels/dataflow/dm_fp32.h used by the cull/blend TRISC compute kernels
// (dm_fp32::unorm16_to_f, dm_fp32::add_sub_roundtrip) against native fp32.
// Standalone:
//
//   c++ -O2 -ffp-contract=off -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_trisc_fp32.cpp -o /tmp/test_trisc_fp32 && /tmp/test_trisc_fp32
//
// Must not be built with -ffast-math / FTZ (subnormal and NaN semantics).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "dm_fp32.h"

static float f(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
static uint32_t u(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

static uint64_t bad = 0, checks = 0;
static void fail(const char* what, uint32_t a, uint32_t b, uint32_t got, uint32_t ref) {
    if (bad++ < 20) std::printf("%s mismatch a=%08x b=%08x got=%08x ref=%08x\n", what, a, b, got, ref);
}

// The kernels' expressions, kept as separate rounded ops.
static uint32_t ref_unorm(uint32_t q) {
    volatile float x = static_cast<float>(q);
    return u(x * (1.0f / 65535.0f));
}
static uint32_t ref_roundtrip(uint32_t a, uint32_t kb) {
    volatile float s = f(a) + f(kb);
    volatile float r = s - f(kb);
    return u(r);
}

static void check_rt(uint32_t a, uint32_t kb) {
    checks++;
    const uint32_t got = dm_fp32::add_sub_roundtrip(a, kb);
    const uint32_t ref = ref_roundtrip(a, kb);
    // NaN payloads may differ between paths; both must be NaN then.
    if (dm_fp32::is_nan(ref) && dm_fp32::is_nan(got)) return;
    if (got != ref) fail("add_sub_roundtrip", a, kb, got, ref);
}

int main() {
    for (uint32_t q = 0; q < 65536u; q++) {
        checks++;
        const uint32_t got = dm_fp32::unorm16_to_f(q), ref = ref_unorm(q);
        if (got != ref) fail("unorm16_to_f", q, 0, got, ref);
    }
    // Tile origins: every multiple of 32 up to 8192, plus other integers.
    uint32_t ks[400];
    uint32_t nk = 0;
    for (uint32_t k = 0; k <= 8192u; k += 32u) ks[nk++] = u(static_cast<float>(k));
    for (uint32_t k : {1u, 3u, 7u, 1000u, 4095u, 65535u, 1u << 20, (1u << 24) - 1u, 1u << 24})
        ks[nk++] = u(static_cast<float>(k));
    // Full sweep of every fp32 pattern against a few origins.
    for (uint32_t kb : {ks[0], ks[1], ks[31], u(8192.0f), u(65535.0f)}) {
        uint32_t a = 0;
        do { check_rt(a, kb); } while (++a != 0u);
    }
    // Dense sweep of pixel-scale tile-local means against all origins.
    for (uint32_t i = 0; i < nk; i++) {
        for (uint32_t a = u(1e-30f); a < u(4096.0f); a += 97u) {
            check_rt(a, ks[i]);
            check_rt(a | 0x80000000u, ks[i]);
        }
        for (uint32_t a = 0; a < 0x00100000u; a += 13u) check_rt(a, ks[i]);  // zeros, subnormals
    }
    std::mt19937 rng(12345);
    for (uint32_t n = 0; n < 200000000u; n++) check_rt(rng(), ks[rng() % nk]);
    std::printf("checks=%llu mismatches=%llu\n", (unsigned long long)checks, (unsigned long long)bad);
    return bad ? 1 : 0;
}
