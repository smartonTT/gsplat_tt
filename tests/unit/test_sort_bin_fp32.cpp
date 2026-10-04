// Exhaustive bit-exactness check for render/kernels/dataflow/sort_bin_fp32.h
// against the fp32 expressions it replaces in sort_bin.cpp. Standalone (the
// full sweep is too slow for the Catch suite):
//
//   c++ -O2 -ffp-contract=off -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_sort_bin_fp32.cpp -o /tmp/test_sort_bin_fp32 && /tmp/test_sort_bin_fp32
//
// -ffp-contract=off matters: the device (soft-float) rounds the multiply and the
// add separately, so the reference must not be fused into an FMA.
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "sort_bin_fp32.h"

static uint32_t ref_unorm(float v) {
    if (v <= 0.0f) return 0u;
    if (v >= 1.0f) return 65535u;
    return static_cast<uint32_t>(v * 65535.0f + 0.5f);
}

static uint32_t ref_sub(uint32_t abits, uint32_t k) {
    float a;
    std::memcpy(&a, &abits, 4);
    const float r = a - static_cast<float>(k);
    uint32_t rb;
    std::memcpy(&rb, &r, 4);
    return rb;
}

int main() {
    uint64_t bad = 0, fast = 0;
    // unorm16: every fp32 bit pattern.
    for (uint64_t i = 0; i <= 0xFFFFFFFFull; i++) {
        const uint32_t b = static_cast<uint32_t>(i);
        uint32_t got;
        if (!sort_bin_fp32::unorm16(b, &got)) continue;  // NaN -> caller's float path
        fast++;
        float v;
        std::memcpy(&v, &b, 4);
        if (got != ref_unorm(v)) {
            if (bad++ < 10) std::printf("unorm16 mismatch bits=%08x got=%u ref=%u\n", b, got, ref_unorm(v));
        }
    }
    std::printf("unorm16: %llu fast-path inputs, %llu mismatches\n",
                (unsigned long long)fast, (unsigned long long)bad);
    uint64_t bad_u = bad;

    // sub_int: every fp32 pattern for the tile origins of a 1024 px frame
    // (k = 32*t, t < 32) sampled at a stride, plus all patterns for a few k.
    bad = 0; fast = 0;
    uint64_t n32 = 0;
    // 1016 is the last k on the 32-bit path for |a| in [4, 8); 1024 is the first past it.
    const uint32_t ks_full[] = {0u, 32u, 480u, 992u, 1016u, 1024u};
    for (uint32_t k : ks_full) {
        for (uint64_t i = 0; i <= 0xFFFFFFFFull; i++) {
            const uint32_t b = static_cast<uint32_t>(i);
            uint32_t got, got32;
            // sub_int32 (the emit's inlined path) must give sub_int's bits when it answers.
            const bool has32 = sort_bin_fp32::sub_int32(b, k, &got32);
            if (has32) n32++;
            if (has32 && got32 != ref_sub(b, k) && bad++ < 10)
                std::printf("sub_int32 mismatch a=%08x k=%u got=%08x ref=%08x\n", b, k, got32, ref_sub(b, k));
            if (!sort_bin_fp32::sub_int(b, k, &got)) {
                if (has32 && bad++ < 10) std::printf("sub_int32 answered outside sub_int a=%08x k=%u\n", b, k);
                continue;
            }
            fast++;
            if (got != ref_sub(b, k) && bad++ < 10)
                std::printf("sub_int mismatch a=%08x k=%u got=%08x ref=%08x\n", b, k, got, ref_sub(b, k));
        }
    }
    uint64_t x = 0x9E3779B97F4A7C15ull;
    for (uint64_t n = 0; n < 2000000000ull; n++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        const uint32_t b = static_cast<uint32_t>(x);
        const uint32_t k = static_cast<uint32_t>(x >> 40) & 0x7FFFFFu;  // any k < 2^23
        uint32_t got, got32;
        const uint32_t kk = (n & 1) ? k : (k & 0xFFE0u);
        if (sort_bin_fp32::sub_int32(b, kk, &got32)) {
            n32++;
            if (got32 != ref_sub(b, kk) && bad++ < 10)
                std::printf("sub_int32 mismatch a=%08x k=%u got=%08x ref=%08x\n", b, kk, got32, ref_sub(b, kk));
        }
        if (!sort_bin_fp32::sub_int(b, kk, &got)) continue;
        fast++;
        if (got != ref_sub(b, kk) && bad++ < 10)
            std::printf("sub_int mismatch a=%08x k=%u got=%08x ref=%08x\n", b, kk, got, ref_sub(b, kk));
    }
    std::printf("sub_int: %llu fast-path inputs (%llu on sub_int32), %llu mismatches\n",
                (unsigned long long)fast, (unsigned long long)n32, (unsigned long long)bad);
    if (n32 == 0) bad++;
    const bool ok = bad_u == 0 && bad == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
