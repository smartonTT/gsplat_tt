// Task #228 (P2): the two cov2d SFPU passes (render/kernels/compute/
// pfwc_cov2d_sfpu.h, GSPLAT_TT_PFWC_COV2D_SFPU=1) run on a model of DEST, the SFPU
// LREGs and the DEST counter give bit for bit the a, b and c of the default path
// (pfwc steps 7, 8 and 9: mul_unary_tile, mul_binary_tile, add_binary_tile and
// add_unary_tile, each one SFPMUL / SFPADD), on scene-like, wide-range and
// special values. The model runs the header's real instruction stream (TTI_
// operands must be constants, as with ".ttinsn" "n"); SFPMUL / SFPADD are SFPMAD
// with one rounding: a*b + 0.0 and a*1.0 + c. LREGs start as garbage, so a value
// the passes do not load is caught. Also checks the inputs the passes must keep.
//   tests/unit/run_cpp.sh tests/unit/test_cov2d_sfpu.cpp
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

namespace emu {
constexpr uint32_t ROWS = 8 * 64;  // 8 fp32 DEST tiles (dst_full_sync_en), 64 rows each
float dest[ROWS / 2][32];          // one 32-lane SFPU vector per even row
float lreg[16][32];
uint32_t rwc = 0;                  // SFPU DEST counter (rows), sfpi::dst_reg++ adds 2
uint64_t n_tti = 0, n_tt = 0, n_incr = 0, n_err = 0;

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
    const uint32_t r = rwc + a;
    check(l < 8 && m == 0 && am == 7 && r % 2 == 0 && r < ROWS, "SFPLOAD");
    std::memcpy(lreg[l % 8], dest[(r % ROWS) / 2], sizeof lreg[0]);
}
inline void store(uint32_t l, uint32_t m, uint32_t am, uint32_t a) {
    const uint32_t r = rwc + a;
    check(l < 8 && m == 0 && am == 7 && r % 2 == 0 && r < ROWS, "SFPSTORE");
    std::memcpy(dest[(r % ROWS) / 2], lreg[l % 8], sizeof lreg[0]);
}
inline void madop(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t m) {
    check(a < 16 && b < 16 && c < 16 && d < 8 && m == 0, "SFPMAD");
    for (int i = 0; i < 32; i++) lreg[d % 8][i] = mad(lreg[a % 16][i], lreg[b % 16][i], lreg[c % 16][i]);
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
template <uint32_t L, uint32_t M, uint32_t I> void tti_loadi() { n_tti++; loadi(L, M, I); }
}  // namespace emu

namespace ckernel {
struct p_sfpu {
    static constexpr uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3, LREG4 = 4, LREG5 = 5,
                              LREG6 = 6, LREG7 = 7, LCONST_0 = 9, LCONST_1 = 10;
};
constexpr uint8_t ADDR_MOD_7 = 7;
}  // namespace ckernel
namespace sfpi {
struct DstReg {
    void operator++(int) const { emu::rwc += 2; emu::n_incr++; }
};
const DstReg dst_reg{};
}  // namespace sfpi
#define TTI_SFPLOAD(l, m, a, d) emu::tti_load<(l), (m), (a), (d)>()
#define TTI_SFPSTORE(l, m, a, d) emu::tti_store<(l), (m), (a), (d)>()
#define TTI_SFPMUL(a, b, c, d, m) emu::tti_mad<(a), (b), (c), (d), (m)>()
#define TTI_SFPADD(a, b, c, d, m) emu::tti_mad<(a), (b), (c), (d), (m)>()
#define TTI_SFPLOADI(l, m, i) emu::tti_loadi<(l), (m), (i)>()
#define TT_SFPLOADI(l, m, i) (emu::n_tt++, emu::loadi((l), (m), (i)))

#include "render/kernels/compute/pfwc_cov2d_sfpu.h"

namespace {
constexpr int N = 1024;  // one chunk tile
using emu::flt;
using emu::mad;

// Default path ops: mul_unary_tile / mul_binary_tile = a * b (SFPMUL, + 0.0),
// add_binary_tile / add_unary_tile = a + b (SFPADD, a * 1.0 + b).
float mul(float a, float b) { return mad(a, b, 0.0f); }
float add(float a, float b) { return mad(a, 1.0f, b); }

struct In { float cc00, cc01, cc02, cc11, cc12, cc22, inv, tx, ty; };
struct Abc { float a, b, c; };

// Steps 7, 8 and 9 of project_pfwc_compute.cpp, in their op order.
Abc ref(const In& x, float fx, float nfx, float fy, float nfy) {
    const float two = flt(0x40000000u), pt3 = flt(0x3E99999Au);
    const float j00 = mul(x.inv, fx);
    const float j02 = mul(mul(mul(x.tx, nfx), x.inv), x.inv);
    const float j11 = mul(x.inv, fy);
    const float j12 = mul(mul(mul(x.ty, nfy), x.inv), x.inv);
    Abc r;
    r.a = mul(mul(x.cc00, j00), j00);
    r.a = add(r.a, mul(mul(mul(x.cc02, j00), j02), two));
    r.a = add(r.a, mul(mul(x.cc22, j02), j02));
    r.a = add(r.a, pt3);
    r.b = mul(mul(x.cc01, j00), j11);
    r.b = add(r.b, mul(mul(x.cc02, j00), j12));
    r.b = add(r.b, mul(mul(x.cc12, j02), j11));
    r.b = add(r.b, mul(mul(x.cc22, j02), j12));
    r.c = mul(mul(x.cc11, j11), j11);
    r.c = add(r.c, mul(mul(mul(x.cc12, j11), j12), two));
    r.c = add(r.c, mul(mul(x.cc22, j12), j12));
    r.c = add(r.c, pt3);
    return r;
}

float& at(int tile, int t) { return emu::dest[tile * 32 + t / 32][t % 32]; }
void garbage(std::mt19937& rng) {
    for (int l = 0; l < 8; l++)
        for (int i = 0; i < 32; i++) emu::lreg[l][i] = flt(rng());
}
}  // namespace

int main() {
    std::mt19937 rng(228);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                           0x00000001u, 0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u};
    static In in[N];
    uint64_t checks = 0, bad = 0, kept_bad = 0;
    for (int it = 0; it < 3000; ++it) {
        const int mode = it % 3;  // 0 scene-like, 1 any finite bits, 2 specials mixed in
        // Focal lengths as fp32 bits; -f is its exact negation (runtime args 54 / 55).
        uint32_t fxb = emu::bits(300.0f + 3000.0f * u01(rng)), fyb = emu::bits(300.0f + 3000.0f * u01(rng));
        if (mode == 1) { fxb = rng() & 0x7F7FFFFFu; fyb = rng() & 0x7F7FFFFFu; }
        const uint32_t nfxb = fxb ^ 0x80000000u, nfyb = fyb ^ 0x80000000u;
        for (int t = 0; t < N; t++) {
            // Scene-like: cov_cam PSD with diagonal over 1e-8..1e2, tz in [0.01, 100],
            // |tx|, |ty| up to 2 tz (off-screen lanes too).
            float d[3];
            for (float& x : d) x = std::pow(10.0f, -8.0f + 10.0f * u01(rng));
            const float tz = std::pow(10.0f, -2.0f + 4.0f * u01(rng));
            float v[9] = {d[0], (2 * u01(rng) - 1) * std::sqrt(d[0] * d[1]),
                          (2 * u01(rng) - 1) * std::sqrt(d[0] * d[2]), d[1],
                          (2 * u01(rng) - 1) * std::sqrt(d[1] * d[2]), d[2], 1.0f / tz,
                          (4 * u01(rng) - 2) * tz, (4 * u01(rng) - 2) * tz};
            for (float& x : v) {
                uint32_t w = emu::bits(x);
                if (mode == 1) w = rng() & 0xFF7FFFFFu;  // finite, any sign/magnitude
                if (mode == 2 && (rng() & 15) == 0) w = sp[rng() % 10];
                x = flt(w);
            }
            in[t] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]};
        }
        const float fx = flt(fxb), nfx = flt(nfxb), fy = flt(fyb), nfy = flt(nfyb);

        // S_AC: tiles 0 cc00, 1 cc02, 2 cc22, 3 cc11, 4 cc12, 5 inv_tz, 6 tx, 7 ty.
        std::memset(emu::dest, 0, sizeof emu::dest);  // acquire after release: DEST cleared
        for (int t = 0; t < N; t++) {
            at(0, t) = in[t].cc00; at(1, t) = in[t].cc02; at(2, t) = in[t].cc22; at(3, t) = in[t].cc11;
            at(4, t) = in[t].cc12; at(5, t) = in[t].inv; at(6, t) = in[t].tx; at(7, t) = in[t].ty;
        }
        emu::reset_consts();
        garbage(rng);
        emu::rwc = 0;
        uint64_t tti0 = emu::n_tti, tt0 = emu::n_tt, inc0 = emu::n_incr;
        pfwc_cov2d::run_ac(fxb, nfxb, fyb, nfyb);
        if (it == 0)
            std::printf("run_ac per chunk: %llu TTI + %llu TT_ SFPU instructions, %llu INCRWC\n",
                        (unsigned long long)(emu::n_tti - tti0), (unsigned long long)(emu::n_tt - tt0),
                        (unsigned long long)(emu::n_incr - inc0));
        emu::check(emu::rwc == 64, "run_ac DEST counter");
        float a_got[N], c_got[N], a6[N], c7[N];
        for (int t = 0; t < N; t++) {
            a_got[t] = at(0, t); c_got[t] = at(3, t); a6[t] = at(6, t); c7[t] = at(7, t);
            const bool kept = emu::bits(at(1, t)) == emu::bits(in[t].cc02) &&
                              emu::bits(at(2, t)) == emu::bits(in[t].cc22) &&
                              emu::bits(at(4, t)) == emu::bits(in[t].cc12) &&
                              emu::bits(at(5, t)) == emu::bits(in[t].inv);
            if (!kept && kept_bad++ < 10) std::printf("run_ac clobbered an input it=%d t=%d\n", it, t);
        }

        // S_BC: tiles 0 cc02, 1 cc01, 2 cc12, 3 cc22, 4 inv_tz, 5 tx, 6 ty.
        std::memset(emu::dest, 0, sizeof emu::dest);
        for (int t = 0; t < N; t++) {
            at(0, t) = in[t].cc02; at(1, t) = in[t].cc01; at(2, t) = in[t].cc12; at(3, t) = in[t].cc22;
            at(4, t) = in[t].inv; at(5, t) = in[t].tx; at(6, t) = in[t].ty;
        }
        emu::reset_consts();
        garbage(rng);
        emu::rwc = 0;
        tti0 = emu::n_tti; tt0 = emu::n_tt; inc0 = emu::n_incr;
        pfwc_cov2d::run_b(fxb, nfxb, fyb, nfyb);
        if (it == 0)
            std::printf("run_b  per chunk: %llu TTI + %llu TT_ SFPU instructions, %llu INCRWC\n",
                        (unsigned long long)(emu::n_tti - tti0), (unsigned long long)(emu::n_tt - tt0),
                        (unsigned long long)(emu::n_incr - inc0));
        emu::check(emu::rwc == 64, "run_b DEST counter");

        for (int t = 0; t < N; t++) {
            const Abc w = ref(in[t], fx, nfx, fy, nfy);
            const uint32_t got[6] = {emu::bits(a_got[t]), emu::bits(a6[t]), emu::bits(at(1, t)),
                                     emu::bits(at(7, t)), emu::bits(c_got[t]), emu::bits(c7[t])};
            const uint32_t want[6] = {emu::bits(w.a), emu::bits(w.a), emu::bits(w.b),
                                      emu::bits(w.b), emu::bits(w.c), emu::bits(w.c)};
            static const char* name[6] = {"a@0", "a@6", "b@1", "b@7", "c@3", "c@7"};
            for (int k = 0; k < 6; k++) {
                ++checks;
                if (got[k] != want[k] && bad++ < 10)
                    std::printf("mismatch it=%d t=%d %s got=%08x want=%08x\n", it, t, name[k], got[k], want[k]);
            }
        }
    }
    std::printf("%llu checks, %llu bad, %llu clobbered, %llu instruction errors\n",
                (unsigned long long)checks, (unsigned long long)bad, (unsigned long long)kept_bad,
                (unsigned long long)emu::n_err);
    return (bad != 0 || kept_bad != 0 || emu::n_err != 0) ? 1 : 0;
}
