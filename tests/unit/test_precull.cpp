// Host checks for lever C (task #140, GSPLAT_TT_PRECULL): the opacity-aware
// radii of project_pfwc_compute.cpp step 11.6 (pfwc_precull_tile). Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_precull.cpp
//
// precull_model() is the lane sequence of pfwc_precull_tile in host fp32, with
// t and the SFPU sqrt pushed to the low side of their error (smaller radius
// = the risky direction). band_keep() is the microblock band cull's keep test
// (microblock_band_cull_compute.cpp: conic from the cov2d, opacity quantized to
// q / 65535, t = 2 ln(op / floor) + 0.05, ellipse vs the microblock's pixel-
// centre box), evaluated in fp32 and in double; a microblock counts as kept if
// either keeps it. Checks, on random and adversarial gaussians:
//   1. every tile with a kept microblock lies in the rectangle built from the
//      shrunk radii (so only dead records are dropped: same image);
//   2. r' <= r, (r' > 0) == (r > 0), (r' > max_radius) == (r > max_radius);
//   3. the test has teeth: without the margin and the 1 px slack it fails.
// Task #157 (PRECULL_PC, GSPLAT_TT_PRECULL=2, the pixel-centre rect): r' = max(
// sqrt(t cov) - 0.5 + 1/8, 1/8), not an integer. A tile's pixel centres meet
// [m - e, m + e] iff the tile meets the rect of radius e - 0.5, so this is the
// same keep region minus the 1.5-2.5 px the integer rect added on each side.
// Checked the same way (1, 2), plus: it keeps fewer rect tiles than the integer
// rect, and with the slack at -1/4 px it loses kept tiles.
// Must not be built with -ffast-math.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

namespace {

constexpr float kFloor = 0.00392156862745098f;  // contrib_floor (1 / 255)
constexpr float kMargin = 0.25f;                // PRECULL_T_MARGIN (pfwc_device.cpp)
constexpr float kRmax = 4096.0f;                // PRECULL_RMAX
constexpr int kTile = 32, kW = 1024, kH = 1024;
constexpr float kK = 1.3862944f / 16384.0f;     // PC_K (2 ln 2 / 2^14)
constexpr float kTErr = 1e-4f;                  // t rounding, worst case low
constexpr float kSqrtErr = 1.0f - 4.0f * 1.2e-7f;  // ~23-bit sqrt, low side
constexpr float kPcSlack = 0.125f;              // PC_SLACK (project_pfwc_compute.cpp)

struct G { float a, b, c, op, mx, my; };

float rad3(float cov) { return std::ceil(3.0f * std::sqrt(std::max(cov, 0.0f))); }

// pfwc_precull_tile, one lane, one axis.
float precull_axis(float t, float cov, float r, bool ok, float slack, bool pc) {
    const float s = std::sqrt(t * cov) * kSqrtErr;
    // pc: r' = max(s + (slack - 0.5), slack) in fp32 (kernel: one SFPMAD and an
    // SFPSWAP). Else sqrt + slack + 1 rounded to an integer, either way (faithful).
    const float rr = pc ? std::max(s + (slack - 0.5f), kPcSlack) : std::floor(s + slack + 1.0f);
    return (ok && rr < r) ? rr : r;
}

void precull_model(const G& g, float rx, float ry, float rlim, float margin, float slack, bool pc,
                   float* rpx, float* rpy) {
    // Kernel step 1: t from the log2 upper bound on the opacity bits (arg 65
    // folds the constants, pfwc_device.cpp); kTErr covers SFPMAD rounding.
    const float c0 = static_cast<float>(2.0 * std::log(1.0 / kFloor) + margin +
                                        2.0 * std::log(2.0) * (0.0860713 + 1.0 / 16384.0 - 127.0));
    const float ac = g.a * g.c;
    const float cond = (ac - g.b * g.b) * 64.0f - ac;
    const bool ok = std::isfinite(g.a) && std::isfinite(g.b) && std::isfinite(g.c) &&
                    !(cond < 0.0f) && !(rx - rlim > 0.0f) && !(ry - rlim > 0.0f);
    uint32_t bits;
    std::memcpy(&bits, &g.op, sizeof bits);
    const float t = std::max(static_cast<float>(bits >> 9) * kK + c0 - kTErr, 0.0f);
    *rpx = precull_axis(t, g.a, rx, ok, slack, pc);
    *rpy = precull_axis(t, g.c, ry, ok, slack, pc);
}

// Minimum of the conic form over the box [u0, u1] x [v0, v1] (relative to the mean).
template <typename T>
T box_min(T ia, T ib, T ic, T u0, T u1, T v0, T v1) {
    if (u0 <= 0 && u1 >= 0 && v0 <= 0 && v1 >= 0) return 0;
    T best = std::numeric_limits<T>::infinity();
    for (T u : {u0, u1}) {
        const T v = std::clamp(-ib * u / ic, v0, v1);
        best = std::min(best, ia * u * u + 2 * ib * u * v + ic * v * v);
    }
    for (T v : {v0, v1}) {
        const T uu = std::clamp(-ib * v / ia, u0, u1);
        best = std::min(best, ia * uu * uu + 2 * ib * uu * v + ic * v * v);
    }
    return best;
}

template <typename T>
bool band_keep_t(const G& g, int tx, int ty) {
    const T det = T(g.a) * T(g.c) - T(g.b) * T(g.b);
    if (!(det > 0)) return false;
    const T ia = T(g.c) / det, ib = -T(g.b) / det, ic = T(g.a) / det;
    const T opq = std::round(T(g.op) * T(65535)) / T(65535);
    const T t = 2 * std::log(opq / T(kFloor)) + T(0.05);
    if (t < 0) return false;
    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 4; k++) {
            const T u0 = T(tx * kTile + 8 * k) + T(0.5) - T(g.mx), u1 = u0 + 7;
            const T v0 = T(ty * kTile + 4 * j) + T(0.5) - T(g.my), v1 = v0 + 3;
            if (box_min<T>(ia, ib, ic, u0, u1, v0, v1) <= t) return true;
        }
    return false;
}

bool band_keep(const G& g, int tx, int ty) {
    return band_keep_t<float>(g, tx, ty) || band_keep_t<double>(g, tx, ty);
}

int cell(float m, float r, int hi) {  // pfwc_vis_cell(fl(m +- r) / tile)
    const float q = (m + r) * (1.0f / kTile);
    return static_cast<int>(std::floor(std::clamp(q, 0.0f, static_cast<float>(hi))));
}

struct Stats {
    long gaussians = 0, shrunk = 0, tiles = 0, tiles_kept = 0, tiles_rect = 0, lost = 0, bad_r = 0;
};

void check(const G& g, float max_radius, float margin, float slack, bool pc, Stats& s) {
    const float rx = rad3(g.a), ry = rad3(g.c);
    float rpx, rpy;
    precull_model(g, rx, ry, std::min(max_radius, kRmax), margin, slack, pc, &rpx, &rpy);
    s.gaussians++;
    if (rpx < rx || rpy < ry) s.shrunk++;
    if (rpx > rx || rpy > ry || (rpx > 0) != (rx > 0) || (rpy > 0) != (ry > 0) ||
        (rpx > max_radius) != (rx > max_radius) || (rpy > max_radius) != (ry > max_radius))
        s.bad_r++;
    if (!(rx > 0 && ry > 0 && rx <= max_radius && ry <= max_radius)) return;
    if (!(g.mx + rx > 0 && kW - (g.mx - rx) > 0 && g.my + ry > 0 && kH - (g.my - ry) > 0)) return;
    const int x0 = cell(g.mx, -rx, kW / kTile - 1), x1 = cell(g.mx, rx, kW / kTile - 1);
    const int y0 = cell(g.my, -ry, kH / kTile - 1), y1 = cell(g.my, ry, kH / kTile - 1);
    // Shrunk lane: the SFPU visibility tests on r' may also cull it (on-screen tests).
    const bool vis2 = g.mx + rpx > 0 && kW - (g.mx - rpx) > 0 && g.my + rpy > 0 &&
                      kH - (g.my - rpy) > 0;
    const int px0 = cell(g.mx, -rpx, kW / kTile - 1), px1 = cell(g.mx, rpx, kW / kTile - 1);
    const int py0 = cell(g.my, -rpy, kH / kTile - 1), py1 = cell(g.my, rpy, kH / kTile - 1);
    for (int ty = y0; ty <= y1; ty++)
        for (int tx = x0; tx <= x1; tx++) {
            s.tiles++;
            const bool in = vis2 && tx >= px0 && tx <= px1 && ty >= py0 && ty <= py1;
            s.tiles_rect += in;
            if (!band_keep(g, tx, ty)) continue;
            s.tiles_kept++;
            if (!in) s.lost++;
        }
}

Stats run(float margin, float slack, bool pc, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    auto logu = [&](float lo, float hi) { return lo * std::pow(hi / lo, U(rng)); };
    Stats s;
    const float max_radius = 1024.0f;
    for (int i = 0; i < 150000; i++) {
        // cov2d = R diag(s1, s2) R^T + 0.3 I (the dilated cov pfwc stores).
        const float s1 = logu(0.01f, 2e4f), s2 = (i % 5 == 0) ? logu(1e-4f, 0.05f) : logu(0.01f, 2e4f);
        const float th = U(rng) * 6.2831853f;
        const float cs = std::cos(th), sn = std::sin(th);
        G g;
        g.a = cs * cs * s1 + sn * sn * s2 + 0.3f;
        g.c = sn * sn * s1 + cs * cs * s2 + 0.3f;
        g.b = cs * sn * (s1 - s2);
        g.op = (i % 7 == 0) ? kFloor * (0.97f + 0.06f * U(rng)) : logu(kFloor * 0.9f, 1.0f);
        if (i % 11 == 0) g.op = 1.0f;
        g.mx = -150.0f + U(rng) * (kW + 300.0f);
        g.my = -150.0f + U(rng) * (kH + 300.0f);
        if (i % 13 == 0) g.mx = std::round(g.mx / kTile) * kTile;  // means on tile borders
        if (i % 17 == 0) g.my = std::round(g.my / kTile) * kTile + 0.5f;
        check(g, max_radius, margin, slack, pc, s);
    }
    return s;
}

}  // namespace

int main() {
    int fail = 0;
    auto report = [&](const char* name, const Stats& s) {
        std::printf("%s: %ld gaussians, %ld shrunk, 3-sigma tiles %ld, in r' rect %ld, kept %ld, "
                    "lost %ld, bad radii %ld\n",
                    name, s.gaussians, s.shrunk, s.tiles, s.tiles_rect, s.tiles_kept, s.lost, s.bad_r);
        if (s.lost != 0 || s.bad_r != 0) fail = 1;
        if (s.shrunk < s.gaussians / 10) { std::printf("FAIL: too few lanes shrink\n"); fail = 1; }
    };
    const Stats s = run(kMargin, 1.0f, false, 140);
    report("precull (integer rect)", s);
    const Stats p = run(kMargin, kPcSlack, true, 140);
    report("precull (pixel-centre rect)", p);
    if (!(p.tiles_rect < s.tiles_rect)) { std::printf("FAIL: pixel-centre rect not tighter\n"); fail = 1; }
    // Teeth: no margin and no slack must lose kept tiles; so must the
    // pixel-centre rect with the slack at -1/4 px.
    const Stats z = run(0.0f, 0.0f, false, 140);
    std::printf("no margin / slack: lost %ld (must be > 0)\n", z.lost);
    if (z.lost == 0) fail = 1;
    const Stats zp = run(kMargin, -0.25f, true, 140);
    std::printf("pixel-centre rect, slack -1/4: lost %ld (must be > 0)\n", zp.lost);
    if (zp.lost == 0) fail = 1;
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}
