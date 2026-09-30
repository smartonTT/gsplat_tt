// Host check that render/kernels/dataflow/emit_intpack.h reproduces the fp32
// soft-float results of the sort emit's record pack bit for bit.
// Standalone (no Catch2/tt-metal); build with SSE fp32 math, no FMA contraction:
//   g++ -std=c++17 -O2 -ffp-contract=off -I render/kernels/dataflow
//       tests/unit/test_emit_intpack.cpp -o /tmp/t && /tmp/t
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "emit_intpack.h"

namespace {
uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float val(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// The reference expressions, exactly as sort_bin.cpp wrote them.
uint32_t ref_unorm16(float v) {
    if (v <= 0.0f) return 0u;
    if (v >= 1.0f) return 65535u;
    return static_cast<uint32_t>(v * 65535.0f + 0.5f);
}
uint32_t ref_sub(float x, uint32_t t) {
    const float r = x - static_cast<float>(t);
    return bits(r);
}
}  // namespace

int main() {
    uint64_t fails = 0, n = 0;
    // UNORM16: every fp32 in [0, 1], plus negatives, >= 1 and specials.
    for (uint64_t u = 0; u <= 0x3f800010ull; ++u, ++n) {
        const uint32_t a = emit_intpack::unorm16(static_cast<uint32_t>(u));
        const uint32_t b = ref_unorm16(val(static_cast<uint32_t>(u)));
        if (a != b && fails++ < 10) std::printf("unorm16 %08llx: %u vs %u\n", (unsigned long long)u, a, b);
    }
    for (float v : {-1.0f, -0.0f, -1e-30f, 1.5f, 2.0f, 1e30f}) {
        ++n;
        if (emit_intpack::unorm16(bits(v)) != ref_unorm16(v) && fails++ < 10) std::printf("unorm16 special %g\n", v);
    }
    // Mean minus tile origin: random x over many binades, every tile origin.
    std::mt19937_64 rng(23);
    for (int i = 0; i < 4000000; ++i) {
        uint32_t u = static_cast<uint32_t>(rng());
        const uint32_t ef = static_cast<uint32_t>(rng() % 160);  // 2^-127 .. 2^32
        u = (u & 0x807fffffu) | (ef << 23);
        const uint32_t t = 32u * static_cast<uint32_t>(rng() % 64);  // 0..2016
        ++n;
        const uint32_t a = emit_intpack::sub_int(u, t), b = ref_sub(val(u), t);
        if (a != b && fails++ < 10) std::printf("sub %08x - %u: %08x vs %08x\n", u, t, a, b);
    }
    // Near-cancellation: x within a few ulps of t, and exact multiples.
    for (uint32_t t = 0; t <= 2016; t += 32) {
        const uint32_t base = bits(static_cast<float>(t));
        for (int d = -300; d <= 300; ++d) {
            const uint32_t u = base + static_cast<uint32_t>(d);
            ++n;
            if (emit_intpack::sub_int(u, t) != ref_sub(val(u), t) && fails++ < 10)
                std::printf("sub near %08x - %u\n", u, t);
            ++n;
            if (emit_intpack::sub_int(u | 0x80000000u, t) != ref_sub(val(u | 0x80000000u), t) && fails++ < 10)
                std::printf("sub near neg %08x - %u\n", u, t);
        }
    }
    std::printf("emit_intpack: %llu checks, %llu mismatches\n", (unsigned long long)n, (unsigned long long)fails);
    return fails ? 1 : 0;
}
