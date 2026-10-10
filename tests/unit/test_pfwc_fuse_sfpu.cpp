// Task #489: the fused pfwc SFPU passes (render/kernels/compute/pfwc_fuse_sfpu.h,
// GSPLAT_TT_PFWC_FUSE) on a model of DEST, the LREGs and the DEST counter (copied
// from test_cov2d_sfpu.cpp) give bit for bit what the tile-op path gives: xform x3 and
// run_mean against steps 1, 4 and 5 in their op order, run_a + run_c against
// pfwc_cov2d::run_ac (itself checked against the tile ops in test_cov2d_sfpu.cpp), on
// scene-like, any-finite and special values, LREGs garbage-filled before each pass.
//   tests/unit/run_cpp.sh tests/unit/test_pfwc_fuse_sfpu.cpp
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
#include "render/kernels/compute/pfwc_fuse_sfpu.h"

namespace {
constexpr int N = 1024;  // one chunk tile
using emu::bits;
using emu::flt;
using emu::mad;

float mul(float a, float b) { return mad(a, b, 0.0f); }
float add(float a, float b) { return mad(a, 1.0f, b); }

float& at(int tile, int t) { return emu::dest[tile * 32 + t / 32][t % 32]; }
void garbage(std::mt19937& rng) {
    for (int l = 0; l < 8; l++)
        for (int i = 0; i < 32; i++) emu::lreg[l][i] = flt(rng());
}
// SFPU start(0) / done between passes: counter back to 0, LREGs keep garbage.
void pass_start(std::mt19937& rng) { emu::rwc = 0; emu::reset_consts(); garbage(rng); }
uint64_t bad = 0, checks = 0;
void expect(uint32_t got, uint32_t want, const char* what, int it, int t) {
    ++checks;
    if (got != want && bad++ < 10) std::printf("mismatch it=%d t=%d %s got=%08x want=%08x\n", it, t, what, got, want);
}
}  // namespace

int main() {
    std::mt19937 rng(489);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const uint32_t sp[] = {0x00000000u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u,
                           0x00000001u, 0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x3F800000u};
    auto val = [&](float x, int mode) {
        uint32_t w = bits(x);
        if (mode == 1) w = rng() & 0xFF7FFFFFu;
        if (mode == 2 && (rng() & 15) == 0) w = sp[rng() % 10];
        return flt(w);
    };
    static float mx[N], my[N], mz[N], inv[N], cc[6][N], tx[N], ty[N];
    for (int it = 0; it < 2000; ++it) {
        const int mode = it % 3;
        // ── projection: xform x3 + run_mean vs steps 1, 4, 5 (tile-op order).
        uint32_t rb[9], tb[3], fxb, fyb, cxb, cyb;
        for (auto& r : rb) r = bits(val(2 * u01(rng) - 1, mode));
        for (auto& t : tb) t = bits(val(10 * (2 * u01(rng) - 1), mode));
        fxb = bits(val(300.0f + 3000.0f * u01(rng), mode));
        fyb = bits(val(300.0f + 3000.0f * u01(rng), mode));
        cxb = bits(val(2000.0f * u01(rng), mode));
        cyb = bits(val(2000.0f * u01(rng), mode));
        for (int t = 0; t < N; t++) {
            mx[t] = val(50 * (2 * u01(rng) - 1), mode);
            my[t] = val(50 * (2 * u01(rng) - 1), mode);
            mz[t] = val(50 * (2 * u01(rng) - 1), mode);
            inv[t] = val(1.0f / (0.01f + 100 * u01(rng)), mode);  // stands in for 1/tz
        }
        std::memset(emu::dest, 0, sizeof emu::dest);
        for (int t = 0; t < N; t++) { at(0, t) = mx[t]; at(1, t) = my[t]; at(2, t) = mz[t]; }
        pass_start(rng); pfwc_fuse::xform<3, 3>(rb[0], rb[1], rb[2], tb[0]); emu::check(emu::rwc == 64, "xform ctr");
        pass_start(rng); pfwc_fuse::xform<4, 4>(rb[3], rb[4], rb[5], tb[1]);
        pass_start(rng); pfwc_fuse::xform<5, 6>(rb[6], rb[7], rb[8], tb[2]);
        for (int t = 0; t < N; t++) {
            const float m[3] = {mx[t], my[t], mz[t]};
            for (int j = 0; j < 3; j++) {
                const float w = add(add(add(mul(m[0], flt(rb[3 * j])), mul(m[1], flt(rb[3 * j + 1]))),
                                        mul(m[2], flt(rb[3 * j + 2]))), flt(tb[j]));
                expect(bits(at(3 + j, t)), bits(w), "xform", it, t);
            }
            expect(bits(at(6, t)), bits(at(5, t)), "tz@6", it, t);
            expect(bits(at(0, t)), bits(mx[t]), "mx kept", it, t);
            at(6, t) = inv[t];  // the 1/tz pass is the kernel's own (pfwc_inv_tz_nr<6>)
        }
        pass_start(rng); pfwc_fuse::run_mean(fxb, cxb, fyb, cyb); emu::check(emu::rwc == 64, "mean ctr");
        for (int t = 0; t < N; t++) {
            const float wx = add(mul(mul(at(3, t), inv[t]), flt(fxb)), flt(cxb));
            const float wy = add(mul(mul(at(4, t), inv[t]), flt(fyb)), flt(cyb));
            expect(bits(at(0, t)), bits(wx), "mean_x", it, t);
            expect(bits(at(1, t)), bits(wy), "mean_y", it, t);
            expect(bits(at(6, t)), bits(inv[t]), "inv kept", it, t);
        }

        // ── cov: run_a + run_c (cc00..cc22 in tiles 0..5) vs pfwc_cov2d::run_ac (S_AC).
        const uint32_t nfxb = fxb ^ 0x80000000u, nfyb = fyb ^ 0x80000000u;
        for (int t = 0; t < N; t++) {
            float d[3];
            for (float& x : d) x = std::pow(10.0f, -8.0f + 10.0f * u01(rng));
            const float tz = std::pow(10.0f, -2.0f + 4.0f * u01(rng));
            const float v[6] = {d[0], (2 * u01(rng) - 1) * std::sqrt(d[0] * d[1]),
                                (2 * u01(rng) - 1) * std::sqrt(d[0] * d[2]), d[1],
                                (2 * u01(rng) - 1) * std::sqrt(d[1] * d[2]), d[2]};
            for (int e = 0; e < 6; e++) cc[e][t] = val(v[e], mode);
            inv[t] = val(1.0f / tz, mode);
            tx[t] = val((4 * u01(rng) - 2) * tz, mode);
            ty[t] = val((4 * u01(rng) - 2) * tz, mode);
        }
        // Reference: S_AC layout 0 cc00, 1 cc02, 2 cc22, 3 cc11, 4 cc12, 5 inv, 6 tx, 7 ty.
        std::memset(emu::dest, 0, sizeof emu::dest);
        for (int t = 0; t < N; t++) {
            at(0, t) = cc[0][t]; at(1, t) = cc[2][t]; at(2, t) = cc[5][t]; at(3, t) = cc[3][t];
            at(4, t) = cc[4][t]; at(5, t) = inv[t]; at(6, t) = tx[t]; at(7, t) = ty[t];
        }
        pass_start(rng); pfwc_cov2d::run_ac(fxb, nfxb, fyb, nfyb);
        static float ra[N], rc[N];
        for (int t = 0; t < N; t++) { ra[t] = at(0, t); rc[t] = at(3, t); }
        // Fused: 0 cc00, 1 cc01, 2 cc02, 3 cc11, 4 cc12, 5 cc22, 6 inv, 7 tx -> ty.
        std::memset(emu::dest, 0, sizeof emu::dest);
        for (int t = 0; t < N; t++) {
            for (int e = 0; e < 6; e++) at(e, t) = cc[e][t];
            at(6, t) = inv[t]; at(7, t) = tx[t];
        }
        pass_start(rng); pfwc_fuse::run_a(fxb, nfxb); emu::check(emu::rwc == 64, "run_a ctr");
        for (int t = 0; t < N; t++) at(7, t) = ty[t];
        pass_start(rng); pfwc_fuse::run_c(fyb, nfyb); emu::check(emu::rwc == 64, "run_c ctr");
        for (int t = 0; t < N; t++) {
            expect(bits(at(0, t)), bits(ra[t]), "a@0", it, t);
            expect(bits(at(6, t)), bits(ra[t]), "a@6", it, t);
            expect(bits(at(3, t)), bits(rc[t]), "c@3", it, t);
            expect(bits(at(7, t)), bits(rc[t]), "c@7", it, t);
            const int keep[4] = {1, 2, 4, 5};
            for (int e : keep) expect(bits(at(e, t)), bits(cc[e][t]), "cc kept", it, t);
        }
        if (it == 0) std::printf("SFPU instructions so far: %llu TTI + %llu TT_\n",
                                 (unsigned long long)emu::n_tti, (unsigned long long)emu::n_tt);
    }
    std::printf("%llu checks, %llu bad, %llu instruction errors\n", (unsigned long long)checks,
                (unsigned long long)bad, (unsigned long long)emu::n_err);
    return (bad != 0 || emu::n_err != 0) ? 1 : 0;
}
