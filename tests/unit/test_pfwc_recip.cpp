// Task #266: pfwc 1/tz accuracy. Models recip_tile's legacy_compat reciprocal
// (tt-llk blackhole ckernel_sfpu_rsqrt_compat.h, _reciprocal_compat_<3>) and the
// GSPLAT_TT_PFWC_RECIP_NEWTON path (pfwc_recip_nr.h: an approx_recip seed with up to
// 2^-7 relative error, then two pfwc_recip_nr_step calls) over bicycle-like depths,
// including z just below powers of two. SFPMAD is one rounding (fma).
//   tests/unit/run_cpp.sh tests/unit/test_pfwc_recip.cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

// Host stand-in for sfpi::vFloat: a * b + c rounds once, like SFPMAD.
struct Prod { float a, b; };
struct V {
    float f;
    V(float x) : f(x) {}
    V(Prod p) : f(std::fma(p.a, p.b, 0.0f)) {}  // a lone product: SFPMUL
};
inline Prod operator*(V a, V b) { return {a.f, b.f}; }
inline V operator-(V c, Prod p) { return V(std::fma(-p.a, p.b, c.f)); }

#include "render/kernels/compute/pfwc_recip_nr.h"

static inline float mul(float a, float b) { return std::fma(a, b, 0.0f); }

// y * (2 - x y) as the header writes it, with the product y * t a plain SFPMUL.
static float nr_step(float x, float y) {
    const V t = V(2.0f) - V(x) * V(y);
    return mul(y, t.f);
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// _reciprocal_compat_<3> for x > 0: mantissa m in [0.5, 1) as val = -m, seed
// 1.442695 * (val * 1.442695 + 2), two steps r * (val * r + 2), exponent fix-up.
static float recip_compat(float x) {
    const float val = -flt((bits(x) & 0x007FFFFFu) | (126u << 23));
    const float c = 1.442695f;
    float r = mul(c, std::fma(val, c, 2.0f));
    for (int i = 0; i < 2; i++) r = mul(r, std::fma(val, r, 2.0f));
    const int ex = int((bits(x) >> 23) & 0xFF) - 127;
    const int er = int((bits(r) >> 23) & 0xFF) - 127;
    const int ne = er - ex + 126;
    if (ne < 0) return 0.0f;
    return flt((bits(r) & 0x807FFFFFu) | (uint32_t(ne) << 23));
}

// approx_recip stand-in: correct reciprocal with the worst 7-bit seed error.
static float seed(float x, float e) { return (1.0f / x) * (1.0f + e); }

int main() {
    std::mt19937 rng(266);
    std::uniform_real_distribution<float> lz(std::log(0.05f), std::log(400.0f));
    double max_compat = 0, max_nr = 0;
    int n = 0;
    auto one = [&](float z, float e) {
        const double ref = 1.0 / double(z);
        const double ec = std::fabs(recip_compat(z) - ref) / ref;
        float y = seed(z, e);
        y = nr_step(z, y);
        y = nr_step(z, y);
        const double en = std::fabs(y - ref) / ref;
        if (ec > max_compat) max_compat = ec;
        if (en > max_nr) max_nr = en;
        n++;
    };
    for (int i = 0; i < 200000; i++) one(std::exp(lz(rng)), (i & 1 ? 1.0f : -1.0f) / 128.0f);
    for (int k = -4; k <= 8; k++)  // z just below 2^k: compat's worst case
        for (int u = 1; u <= 64; u++) {
            const float z = flt(bits(std::ldexp(1.0f, k)) - uint32_t(u));
            one(z, 1.0f / 128.0f); one(z, -1.0f / 128.0f);
        }
    // The header's step (used by the kernel) must equal the model's step.
    int same = 0;
    for (float x : {0.3f, 1.9916f, 3.997f, 7.999f, 15.995f, 123.4f})
        for (float e : {1.0f / 128, -1.0f / 128, 0.0f}) {
            const float y = seed(x, e);
            same += bits(pfwc_recip_nr_step(V(x), V(y)).f) == bits(nr_step(x, y)) ? 0 : 1;
        }
    std::printf("n=%d compat max rel %.3g, newton max rel %.3g, header mismatches %d\n", n,
                max_compat, max_nr, same);
    // Compat really is ~1.5e-3 off (the #260 residual); the Newton path is ~1 ulp.
    const bool ok = max_compat > 1e-3 && max_compat < 2e-3 && max_nr < 2.5e-7 && same == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
