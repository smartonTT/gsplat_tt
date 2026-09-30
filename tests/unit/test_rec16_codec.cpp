// Host check of the 16 B per-pair bucket record codec
// (render/kernels/dataflow/rec16_codec.h), which the device kernels include.
// Standalone (no Catch2/tt-metal):
//   c++ -std=c++17 -O2 -I render/kernels/dataflow tests/unit/test_rec16_codec.cpp -o /tmp/t && /tmp/t
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>

#include "rec16_codec.h"

namespace {

int g_fail = 0;

void check(bool ok, const char* what, double a, double b) {
    if (!ok && g_fail++ < 20) std::printf("FAIL %s: %.9g vs %.9g\n", what, a, b);
}

uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float val(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// IEEE half -> double, for normal halves (the only kind the encoders emit).
double half_value(uint32_t h) {
    const int e = static_cast<int>((h >> 10) & 0x1f);
    const double m = 1.0 + (h & 0x3ff) / 1024.0;
    const double v = std::ldexp(m, e - 15);
    return (h & 0x8000) ? -v : v;
}

}  // namespace

int main() {
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Decode is exact: h16_to_f32 reproduces the IEEE half value (times 2^-s).
    for (uint32_t h = 0x0400; h < 0x7c00; ++h) {
        for (uint32_t s : {0u, 10u}) {
            const double ref = std::ldexp(half_value(h), -static_cast<int>(s));
            check(val(rec16::h16_to_f32(h, s)) == ref, "decode", val(rec16::h16_to_f32(h, s)), ref);
            check(val(rec16::h16_to_f32(h | 0x8000, s)) == -ref, "decode-neg", 0, ref);
        }
    }

    // Conic: half of x*2^10, relative error <= 2^-11 across 1e-7..3 (|A|,|B|,|C| range).
    for (int i = 0; i < 200000; ++i) {
        const float x = (i & 1 ? -1.0f : 1.0f) * std::pow(10.0f, -7.0f + 7.5f * u01(rng));
        const uint32_t h = rec16::f32_to_h16(bits(x), rec16::kConicScaleLog2);
        const double back = val(rec16::h16_to_f32(h, rec16::kConicScaleLog2));
        const double tol = std::fabs(x) >= 6.2e-8 ? std::ldexp(std::fabs(x), -11) * 1.0001 : 6.2e-8;
        check(std::fabs(back - x) <= tol, "conic", back, x);
        check((h & 0x7c00) != 0 && (h & 0x7c00) != 0x7c00, "conic-normal", h, 0);
    }

    // Mean: exact 1/256 px fixed point, then half of v/256.
    for (int i = 0; i < 400000; ++i) {
        const float x = (u01(rng) - 0.3f) * (i % 4 == 0 ? 16000.0f : 96.0f);
        const int32_t fx = rec16::f32_to_fx8(bits(x));
        const double ref_fx = std::copysign(std::floor(std::fabs(static_cast<double>(x)) * 256.0 + 0.5), x);
        check(fx == ref_fx, "fx8", fx, ref_fx);
        for (int32_t tile = 0; tile < 1024 * 256; tile += 32 * 256 * 7) {
            const int32_t v = fx - tile;
            const double ref = v / 256.0;
            const double back = val(rec16::h16_to_f32(rec16::fx8_to_h16(v), 0));
            const double tol = std::max(std::ldexp(std::fabs(ref), -11) * 1.0001, std::ldexp(1.0, -14));
            check(std::fabs(back - ref) <= tol, "mean", back, ref);
        }
    }

    // UNORM16: exact round(v*65535) with clamping.
    for (int i = 0; i < 400000; ++i) {
        const float v = u01(rng) * 1.2f - 0.1f;
        const double c = v < 0 ? 0.0 : (v > 1 ? 1.0 : v);
        const double ref = std::floor(c * 65535.0 + 0.5);
        check(rec16::f32_to_unorm16(bits(v)) == ref, "unorm16", rec16::f32_to_unorm16(bits(v)), ref);
    }
    check(rec16::f32_to_unorm16(bits(0.0f)) == 0, "unorm16-0", 0, 0);
    check(rec16::f32_to_unorm16(bits(1.0f)) == 65535, "unorm16-1", 0, 0);

    // Full record round trip.
    for (int i = 0; i < 100000; ++i) {
        const float A = -std::pow(10.0f, -6.0f + 6.0f * u01(rng)), B = (u01(rng) - 0.5f) * 0.5f * std::fabs(A);
        const float C = -std::pow(10.0f, -6.0f + 6.0f * u01(rng));
        const float mx = (u01(rng) - 0.5f) * 300.0f, my = (u01(rng) - 0.5f) * 300.0f;
        const uint32_t op = static_cast<uint32_t>(u01(rng) * 65535.0f);
        const float rf = u01(rng), gf = (i & 7) ? u01(rng) : 1.0f, bf = (i & 15) ? u01(rng) : 0.0f;
        const uint32_t r = rec16::f32_to_unorm16(bits(rf)), g = rec16::f32_to_unorm16(bits(gf)),
                       b = rec16::f32_to_unorm16(bits(bf));
        const uint32_t w[4] = {
            rec16::f32_to_h16(bits(A), 10) | (rec16::f32_to_h16(bits(B), 10) << 16),
            rec16::f32_to_h16(bits(C), 10) | (rec16::fx8_to_h16(rec16::f32_to_fx8(bits(mx))) << 16),
            rec16::fx8_to_h16(rec16::f32_to_fx8(bits(my))) | (op << 16),
            rec16::pack_color_word(bits(rf), bits(gf), bits(bf))};
        uint32_t out[8];
        rec16::expand(w, 0x12345678u, out);
        check(std::fabs(val(out[0]) - A) <= std::ldexp(std::fabs(A), -11) * 1.0001, "rec-A", val(out[0]), A);
        check(std::fabs(val(out[2]) - C) <= std::ldexp(std::fabs(C), -11) * 1.0001, "rec-C", val(out[2]), C);
        check(out[3] == 0x12345678u, "rec-key", out[3], 0);
        check(std::fabs(val(out[4]) - mx) <= (std::ldexp(std::fabs(static_cast<double>(mx)), -11) + 1.0 / 512) * 1.0001, "rec-mx", val(out[4]), mx);
        check(std::fabs(val(out[5]) - my) <= (std::ldexp(std::fabs(static_cast<double>(my)), -11) + 1.0 / 512) * 1.0001, "rec-my", val(out[5]), my);
        check((out[6] & 0xffff) == op, "rec-op", out[6] & 0xffff, op);
        const int dr = static_cast<int>(out[6] >> 16) - static_cast<int>(r);
        const int dg = static_cast<int>(out[7] & 0xffff) - static_cast<int>(g);
        const int db = static_cast<int>(out[7] >> 16) - static_cast<int>(b);
        check(std::abs(dr) <= 18 && std::abs(dg) <= 18 && std::abs(db) <= 34, "rec-color", dr, db);
    }
    uint32_t out[8];
    const uint32_t full[4] = {0x3c003c00u, 0x3c003c00u, 0x3c00u,
                              rec16::pack_color_word(bits(1.0f), bits(1.0f), bits(1.0f))};
    rec16::expand(full, 0, out);
    check(out[6] >> 16 == 65535 && out[7] == 0xffffffffu, "color-full-scale", out[6] >> 16, out[7]);

    std::printf(g_fail ? "rec16 codec: %d FAILURES\n" : "rec16 codec: all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
