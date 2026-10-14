// Task #788: the train-mode T early-out (GSPLAT_TT_TRAIN_T_PERIOD) is lossless.
// Scalar model of blend_train_gaussian_math (alpha_blend_compute_mb.cpp) on a 32x32
// tile: blend N random gaussians with no early-out, and again with the readback
// every P records (bf16 T tile, blend_t_live_fast with eps = FLT_MIN) skipping
// dead microblock vectors. R, G, B, T must match bit for bit. With the render eps
// (1/256) the same skip is lossy, which the test also checks (so it can fail).
//   tests/unit/run_cpp.sh tests/unit/test_train_t_early_out.cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

#include "render/kernels/compute/blend_t_live.h"

namespace {
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float setsgn1(float f) { uint32_t u = fbits(f) | 0x80000000u; std::memcpy(&f, &u, 4); return f; }
struct G { float a, b, c, mx, my, op, r, g, bl; };
struct Px { float R, Gc, B, T; };

void blend(Px& p, float x, float y, const G& g) {  // gsplat's exact rules, as the train body
    const float dx = x - g.mx, dy = y - g.my;
    float power = g.a * (dx * dx) + g.b * (dx * dy) + g.c * (dy * dy);
    if (power > 0.0f) power = 0.0f;
    if (power < -80.0f) power = -80.0f;
    float alpha = g.op * std::exp(power);
    if (alpha > 0.99f) alpha = 0.99f;
    if (alpha >= 0.003921569f) {
        const float tn = p.T * (1.0f - alpha);
        if (tn >= 0.0001f) {
            const float at = alpha * p.T;
            p.R += at * g.r; p.Gc += at * g.g; p.B += at * g.bl; p.T = tn;
        } else {
            p.T = setsgn1(p.T);
        }
    }
}

uint16_t bf16(float f) { return static_cast<uint16_t>(fbits(f) >> 16); }  // pack truncates or rounds; sign kept

// Returns the number of mismatching pixels between the full blend and the early-out blend.
int run(std::mt19937& rng, uint32_t eps_bits, uint32_t period, int n, bool round_bf16) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    Px ref[1024], eo[1024];
    for (auto* a : {ref, eo})
        for (int i = 0; i < 1024; ++i) a[i] = {0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t live = 0xffffffffu;
    for (int k = 0; k < n; ++k) {
        const float s = 0.002f + 0.05f * u(rng);
        G g{-s, (u(rng) - 0.5f) * s, -s, 32.0f * u(rng), 32.0f * u(rng), 0.3f + 0.7f * u(rng), u(rng), u(rng), u(rng)};
        if (period != 0u && k != 0 && (k % period) == 0) {
            uint32_t w[512];
            for (int t = 0; t < 1024; t += 2) {
                uint32_t lo = bf16(eo[t].T), hi = bf16(eo[t + 1].T);
                if (round_bf16) {  // round to nearest even, as the packer may
                    lo = static_cast<uint16_t>((fbits(eo[t].T) + 0x7fffu + ((fbits(eo[t].T) >> 16) & 1u)) >> 16);
                    hi = static_cast<uint16_t>((fbits(eo[t + 1].T) + 0x7fffu + ((fbits(eo[t + 1].T) >> 16) & 1u)) >> 16);
                }
                w[t >> 1] = lo | (hi << 16);
            }
            live = blend_t_live_fast(w, eps_bits);
        }
        for (int t = 0; t < 1024; ++t) {
            const int r = t >> 5, c = t & 31;
            const float x = static_cast<float>(c), y = static_cast<float>(r);
            blend(ref[t], x, y, g);
            const uint32_t V = static_cast<uint32_t>((r & ~1) | (c & 1));
            if (live & (1u << V)) blend(eo[t], x, y, g);
        }
    }
    int bad = 0;
    for (int t = 0; t < 1024; ++t)
        bad += std::memcmp(&ref[t], &eo[t], sizeof(Px)) != 0;
    return bad;
}
}  // namespace

int main() {
    std::mt19937 rng(788);
    const uint32_t kExact = 0x00800000u;  // FLT_MIN, the train-mode BLEND_T_EPS
    const uint32_t kRender = 0x3B800000u; // 1/256, the render default
    int exact_bad = 0, render_bad = 0;
    for (int it = 0; it < 60; ++it) {
        const uint32_t period = (it % 3 == 0) ? 1u : (it % 3 == 1) ? 7u : 64u;
        exact_bad += run(rng, kExact, period, 600, it & 1);
        render_bad += run(rng, kRender, period, 600, it & 1);
    }
    std::printf("exact eps: %d mismatching pixels (want 0); render eps: %d (want > 0)\n", exact_bad, render_bad);
    return (exact_bad == 0 && render_bad > 0) ? 0 : 1;
}
