// Task #206: the single-pass cov_cam SFPU sequence (render/kernels/compute/
// pfwc_covcam_sfpu.h, GSPLAT_TT_PFWC_COVCAM_SFPU=1) run on a model of DEST and the
// SFPU LREGs gives bit for bit what the default path (compute_cc_entry_to_scratch:
// copy, mul_unary_tile, then 5 x copy, mul_unary_tile, add_binary_tile per entry)
// gives, on scene-like, wide-range and special values. The model runs the header's
// real instruction stream (TTI_ operands must be constants, as with ".ttinsn" "n").
// SFPMUL/SFPADD are SFPMAD with one rounding: a*b + 0.0 and a*1.0 + c.
// Also reports how often a fused-MAD variant would differ (md5 risk).
//   tests/unit/run_cpp.sh tests/unit/test_covcam_sfpu.cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace emu {
constexpr uint32_t ROWS = 8 * 64;  // 8 fp32 DEST tiles (dst_full_sync_en), 64 rows each
float dest[ROWS / 2][32];          // one 32-lane SFPU vector per even row
float lreg[16][32];
uint64_t n_tti = 0, n_tt = 0, n_err = 0;

inline uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float flt(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float mad(float a, float b, float c) { return std::fma(a, b, c); }
inline void check(bool ok, const char* what) {
    if (!ok && n_err++ < 10) std::printf("bad instruction: %s\n", what);
}
inline void reset_consts() {
    for (int i = 0; i < 32; i++) { lreg[9][i] = 0.0f; lreg[10][i] = 1.0f; }
}
inline void load(uint32_t l, uint32_t m, uint32_t am, uint32_t a) {
    check(l < 8 && m == 0 && am == 7 && a % 2 == 0 && a < ROWS, "SFPLOAD");
    std::memcpy(lreg[l], dest[(a % ROWS) / 2], sizeof lreg[l]);
}
inline void store(uint32_t l, uint32_t m, uint32_t am, uint32_t a) {
    check(l < 8 && m == 0 && am == 7 && a % 2 == 0 && a < ROWS, "SFPSTORE");
    std::memcpy(dest[(a % ROWS) / 2], lreg[l % 16], sizeof lreg[0]);
}
inline void madop(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t m) {
    check(a < 16 && b < 16 && c < 16 && d < 8 && m == 0, "SFPMAD");
    for (int i = 0; i < 32; i++) lreg[d % 8][i] = mad(lreg[a][i], lreg[b][i], lreg[c][i]);
}
inline void loadi(uint32_t l, uint32_t m, uint32_t imm) {
    check(l < 8 && (m == 8 || m == 10) && imm <= 0xFFFF, "SFPLOADI");
    for (int i = 0; i < 32; i++) {
        const uint32_t u = bits(lreg[l % 8][i]);
        lreg[l % 8][i] = flt(m == 8 ? ((u & 0xFFFFu) | (imm << 16)) : ((u & 0xFFFF0000u) | imm));
    }
}
template <uint32_t L, uint32_t M, uint32_t A, uint32_t D> void tti_load() { n_tti++; load(L, M, A, D); }
template <uint32_t L, uint32_t M, uint32_t A, uint32_t D> void tti_store() { n_tti++; store(L, M, A, D); }
template <uint32_t A, uint32_t B, uint32_t C, uint32_t D, uint32_t M> void tti_mad() { n_tti++; madop(A, B, C, D, M); }
}  // namespace emu

namespace ckernel {
struct p_sfpu {
    static constexpr uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3, LREG4 = 4, LREG5 = 5,
                              LREG6 = 6, LREG7 = 7, LCONST_0 = 9, LCONST_1 = 10;
};
constexpr uint8_t ADDR_MOD_7 = 7;
}  // namespace ckernel
#define TTI_SFPLOAD(l, m, a, d) emu::tti_load<(l), (m), (a), (d)>()
#define TTI_SFPSTORE(l, m, a, d) emu::tti_store<(l), (m), (a), (d)>()
#define TTI_SFPMUL(a, b, c, d, m) emu::tti_mad<(a), (b), (c), (d), (m)>()
#define TTI_SFPADD(a, b, c, d, m) emu::tti_mad<(a), (b), (c), (d), (m)>()
#define TT_SFPLOAD(l, m, a, d) (emu::n_tt++, emu::load((l), (m), (a), (d)))
#define TT_SFPSTORE(l, m, a, d) (emu::n_tt++, emu::store((l), (m), (a), (d)))
#define TT_SFPLOADI(l, m, i) (emu::n_tt++, emu::loadi((l), (m), (i)))

#include "render/kernels/compute/pfwc_covcam_sfpu.h"

namespace {
constexpr int N = 1024;  // one chunk tile

// Default path, one entry: mul_unary_tile = val * parameter (SFPMUL, + 0.0),
// add_binary_tile(0, 1, 0) = in0 + in1 (SFPADD, in0 * 1.0 + in1).
float ref_entry(const float* c, const uint32_t* s) {
    float acc = emu::mad(c[0], emu::flt(s[0]), 0.0f);
    for (int k = 1; k < 6; k++) acc = emu::mad(acc, 1.0f, emu::mad(c[k], emu::flt(s[k]), 0.0f));
    return acc;
}

// Plain C++ fp32, every op rounded (volatile stops contraction to FMA).
float plain_entry(const float* c, const uint32_t* s) {
    volatile float acc = c[0] * emu::flt(s[0]);
    for (int k = 1; k < 6; k++) {
        volatile float p = c[k] * emu::flt(s[k]);
        acc = acc + p;
    }
    return acc;
}

// Fused variant: acc = fma(c_k, s_k, acc), one rounding per term.
float fused_entry(const float* c, const uint32_t* s) {
    float acc = emu::mad(c[0], emu::flt(s[0]), 0.0f);
    for (int k = 1; k < 6; k++) acc = emu::mad(c[k], emu::flt(s[k]), acc);
    return acc;
}

bool both_zero(uint32_t a, uint32_t b) { return ((a | b) & 0x7FFFFFFFu) == 0; }
bool is_nan(uint32_t a) { return (a & 0x7FFFFFFFu) > 0x7F800000u; }
}  // namespace

int main() {
    std::mt19937 rng(206);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                           0x00000001u, 0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u};
    static float in[6][N];
    uint32_t s[36];
    uint64_t checks = 0, bad = 0, plain_diff = 0, plain_nonzero_diff = 0, fused_diff = 0,
             scene_vals = 0, swap_diff = 0;
    for (int it = 0; it < 3000; ++it) {
        const int mode = it % 3;  // 0 scene-like, 1 any finite bits, 2 specials mixed in
        // Scales: products of a random rotation's entries (as cov_cam's W.Sigma.W^T has),
        // or random bits.
        float R[3][3];
        {
            float q[4], n = 0;
            for (float& x : q) { x = u01(rng) * 2 - 1; n += x * x; }
            n = std::sqrt(n);
            for (float& x : q) x /= n;
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            const float r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                                   {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                                   {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
            std::memcpy(R, r, sizeof R);
        }
        // Entry e = (i, j) in CC_SCRATCH order, input k = (a, b) in COV3D_CB order.
        const int EI[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
        const int KI[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
        for (int e = 0; e < 6; e++)
            for (int k = 0; k < 6; k++) {
                const int i = EI[e][0], j = EI[e][1], a = KI[k][0], b = KI[k][1];
                const float v = (a == b) ? R[i][a] * R[j][a] : R[i][a] * R[j][b] + R[i][b] * R[j][a];
                uint32_t w = emu::bits(v);
                if (mode == 1) w = rng() & 0xFF7FFFFFu;            // finite, any sign/magnitude
                if (mode == 2 && (rng() & 7) == 0) w = sp[rng() % 10];
                s[6 * e + k] = w;
            }
        for (int t = 0; t < N; t++) {
            // Scene-like cov3d: diagonal >= 0 over 1e-8..1e2, |off-diagonal| <= sqrt(c_aa c_bb).
            float d[3];
            for (float& x : d) x = std::pow(10.0f, -8.0f + 10.0f * u01(rng));
            const float v[6] = {d[0], d[1], d[2], (2 * u01(rng) - 1) * std::sqrt(d[0] * d[1]),
                                (2 * u01(rng) - 1) * std::sqrt(d[0] * d[2]),
                                (2 * u01(rng) - 1) * std::sqrt(d[1] * d[2])};
            for (int k = 0; k < 6; k++) {
                uint32_t w = emu::bits(v[k]);
                if (mode == 1) w = rng() & 0xFF7FFFFFu;
                if (mode == 2 && (rng() & 15) == 0) w = sp[rng() % 10];
                in[k][t] = emu::flt(w);
            }
        }
        // New path: copy_tile COV3D_CB[k] -> tile k, run(), pack tile e -> CC_SCRATCH[e].
        std::memset(emu::dest, 0, sizeof emu::dest);  // acquire after release: DEST cleared
        for (int k = 0; k < 6; k++)
            for (int t = 0; t < N; t++) emu::dest[k * 32 + t / 32][t % 32] = in[k][t];
        emu::reset_consts();
        const uint64_t tti0 = emu::n_tti, tt0 = emu::n_tt;
        pfwc_covcam::run([&](uint32_t j) { return s[j]; });
        if (it == 0)
            std::printf("per chunk: %llu TTI + %llu TT_ SFPU instructions\n",
                        (unsigned long long)(emu::n_tti - tti0), (unsigned long long)(emu::n_tt - tt0));
        for (int e = 0; e < 6; e++)
            for (int t = 0; t < N; t++) {
                float c[6];
                for (int k = 0; k < 6; k++) c[k] = in[k][t];
                const uint32_t got = emu::bits(emu::dest[e * 32 + t / 32][t % 32]);
                const uint32_t want = emu::bits(ref_entry(c, s + 6 * e));
                ++checks;
                if (got != want && bad++ < 10)
                    std::printf("mismatch it=%d e=%d t=%d got=%08x want=%08x\n", it, e, t, got, want);
                const uint32_t pl = emu::bits(plain_entry(c, s + 6 * e));
                if (pl != want && !(is_nan(pl) && is_nan(want))) {
                    ++plain_diff;
                    if (!both_zero(pl, want)) ++plain_nonzero_diff;
                }
                // a + b == b + a under one rounding: SFPADD operand order is free.
                float acc = emu::mad(c[0], emu::flt(s[6 * e]), 0.0f), acc2 = acc;
                for (int k = 1; k < 6; k++) {
                    const float p = emu::mad(c[k], emu::flt(s[6 * e + k]), 0.0f);
                    acc = emu::mad(acc, 1.0f, p);
                    acc2 = emu::mad(p, 1.0f, acc2);
                }
                if (emu::bits(acc) != emu::bits(acc2) && !(is_nan(emu::bits(acc)) && is_nan(emu::bits(acc2))))
                    ++swap_diff;
                if (mode == 0) {
                    ++scene_vals;
                    if (emu::bits(fused_entry(c, s + 6 * e)) != want) ++fused_diff;
                }
            }
    }
    std::printf("%llu checks, %llu bad, %llu instruction errors\n", (unsigned long long)checks,
                (unsigned long long)bad, (unsigned long long)emu::n_err);
    std::printf("vs plain fp32: %llu differ, %llu not +-0 only; operand swap: %llu differ\n",
                (unsigned long long)plain_diff, (unsigned long long)plain_nonzero_diff,
                (unsigned long long)swap_diff);
    std::printf("fused-MAD variant on scene-like data: %llu of %llu values differ (%.1f%%)\n",
                (unsigned long long)fused_diff, (unsigned long long)scene_vals,
                scene_vals ? 100.0 * fused_diff / scene_vals : 0.0);
    return (bad != 0 || emu::n_err != 0 || plain_nonzero_diff != 0 || swap_diff != 0) ? 1 : 0;
}
