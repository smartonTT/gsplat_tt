// Host checks for task #738 (B1 full): render/kernels/dataflow/subtile_route.h.
// Standalone:
//
//   CXXFLAGS="-Irender/host -Irender/kernels/dataflow" tests/unit/run_cpp.sh tests/unit/test_subtile_route.cpp
//
//  - ln_upper is an upper bound of ln within 0.06 over the opacity / floor range;
//  - box_mask covers every microblock the cull can keep: the exact "ellipse
//    m2 <= 2 ln(op/floor) + 0.05 meets the pixel-centre box" test (boxmin, as in
//    tests/spec/test_band_cull.py), with the device's boundary tolerance added;
//  - degenerate and zero-opacity records: full mask / empty mask;
//  - routing by box_mask to the sub-tiles of a split (subtile_split.h plans)
//    is lossless: every microblock a record's cull keeps lies in a sub-tile the
//    record was routed to, and parent order is kept per sub list;
//  - the box is tight enough to matter (printed: mean box vs kept microblocks).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "subtile_route.h"
#include "subtile_split.h"
#include "subtile_unit.h"

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

namespace {

constexpr double kFloor = 1.0 / 255.0;

struct Rec {
    float A, B, C, mx, my;
    uint32_t q;
};

uint32_t bits(float f) { return subtile_route::bits_of(f); }

// Exact minimum of ca u^2 + 2 cb u v + cc v^2 over [ulo, uhi] x [vlo, vhi]
// (positive definite: convex, so 0 inside, else on an edge).
double boxmin_m2(double ca, double cb, double cc, double ulo, double uhi, double vlo, double vhi) {
    if (ulo <= 0 && 0 <= uhi && vlo <= 0 && 0 <= vhi) return 0.0;
    auto q = [&](double u, double v) { return ca * u * u + 2 * cb * u * v + cc * v * v; };
    auto clampd = [](double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); };
    double best = 1e300;
    for (double u : {ulo, uhi}) best = std::fmin(best, q(u, clampd(-cb * u / cc, vlo, vhi)));
    for (double v : {vlo, vhi}) best = std::fmin(best, q(clampd(-cb * v / ca, ulo, uhi), v));
    return best;
}

// Microblocks the cull may keep (device tolerance on the thr boundary added).
uint32_t exact_keep(const Rec& r) {
    const double ca = -2.0 * r.A, cb = -static_cast<double>(r.B), cc = -2.0 * r.C;
    const double op = r.q / 65535.0;
    if (op <= 0) return 0;
    const double thr = 2 * std::log(op / kFloor) + 0.05;
    const double eps = 1e-3 * std::fmax(1.0, std::fabs(thr));
    uint32_t m = 0;
    for (uint32_t k = 0; k < 32; ++k) {
        const double ulo = (k & 3) * 8 + 0.5 - r.mx, vlo = (k >> 2) * 4 + 0.5 - r.my;
        if (boxmin_m2(ca, cb, cc, ulo, ulo + 7.0, vlo, vlo + 3.0) <= thr + eps) m |= 1u << k;
    }
    return m;
}

uint32_t box_of(const Rec& r) {
    return subtile_route::box_mask(bits(r.A), bits(r.B), bits(r.C), bits(r.mx), bits(r.my), r.q,
                                   static_cast<float>(1.0 / kFloor));
}

std::vector<Rec> random_recs(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<Rec> v;
    for (size_t i = 0; i < n; ++i) {
        const double sa = std::pow(10.0, -1 + 3 * u01(g)), sb = std::pow(10.0, -1 + 3 * u01(g));
        const double th = M_PI * u01(g), c = std::cos(th), s = std::sin(th);
        // cov = R diag(sa^2, sb^2) R^T; conic = cov^-1
        const double xx = c * c * sa * sa + s * s * sb * sb, yy = s * s * sa * sa + c * c * sb * sb;
        const double xy = c * s * (sa * sa - sb * sb), det = xx * yy - xy * xy;
        const double ca = yy / det, cb = -xy / det, cc = xx / det;
        Rec r;
        r.A = static_cast<float>(-0.5 * ca);
        r.B = static_cast<float>(-cb);
        r.C = static_cast<float>(-0.5 * cc);
        r.mx = static_cast<float>(-40 + 112 * u01(g));
        r.my = static_cast<float>(-40 + 112 * u01(g));
        r.q = static_cast<uint32_t>(std::lround(std::pow(10.0, -2.5 * u01(g)) * 65535));
        v.push_back(r);
    }
    return v;
}

void test_ln_upper() {
    for (double x = 1e-6; x < 1e6; x *= 1.0007) {
        const double l = subtile_route::ln_upper(static_cast<float>(x));
        CHECK(l >= std::log(static_cast<float>(x)) - 1e-6);
        CHECK(l <= std::log(x) + 0.065);
    }
}

void test_box_covers_cull() {
    const auto recs = random_recs(20000, 11);
    size_t kept = 0, boxed = 0, nonempty = 0;
    for (const auto& r : recs) {
        const uint32_t k = exact_keep(r), b = box_of(r);
        if (k & ~b) {
            std::fprintf(stderr, "miss keep=%08x box=%08x\n", k, b);
            CHECK(false);
        }
        kept += __builtin_popcount(k);
        boxed += __builtin_popcount(b);
        nonempty += k != 0;
    }
    std::printf("box microblocks %.3f x kept (%zu records keep something)\n",
                kept ? static_cast<double>(boxed) / kept : 0.0, nonempty);
    CHECK(nonempty > recs.size() / 20);
    CHECK(boxed < 3 * kept);  // tight enough that a split actually shortens lists
}

void test_degenerate() {
    const float inv = static_cast<float>(1.0 / kFloor);
    // op 0: nothing kept
    CHECK(subtile_route::box_mask(bits(-0.5f), 0, bits(-0.5f), bits(16.f), bits(16.f), 0, inv) == 0u);
    // not positive definite / NaN / inf: full mask
    CHECK(subtile_route::box_mask(bits(0.5f), 0, bits(-0.5f), bits(16.f), bits(16.f), 65535, inv) ==
          subtile_route::kFullMask);
    CHECK(subtile_route::box_mask(bits(-0.5f), bits(2.0f), bits(-0.5f), bits(16.f), bits(16.f), 65535,
                                  inv) == subtile_route::kFullMask);
    CHECK(subtile_route::box_mask(bits(-0.5f), 0, bits(-0.5f), bits(NAN), bits(16.f), 65535, inv) ==
          subtile_route::kFullMask);
    CHECK(subtile_route::box_mask(bits(-1e-30f), 0, bits(-1e-30f), bits(16.f), bits(16.f), 65535,
                                  inv) == subtile_route::kFullMask);
    // op below the floor: thr < 0, nothing kept
    CHECK(subtile_route::box_mask(bits(-0.5f), 0, bits(-0.5f), bits(16.f), bits(16.f), 100, inv) == 0u);
    // a point-like splat at the tile centre keeps only its microblock
    CHECK(subtile_route::box_mask(bits(-500.f), 0, bits(-500.f), bits(12.f), bits(18.f), 65535, inv) ==
          (1u << ((4 << 2) | 1)));
}

void test_routing_lossless() {
    using namespace gsplat_tt::subtile;
    // One tile split into 4 quarters with two heavy quarters (8x8), plus an unsplit tile.
    std::vector<double> cost = {100.0, 1.0};
    std::vector<std::array<double, 4>> qc = {{{40.0, 10.0, 40.0, 10.0}}, {{0.25, 0.25, 0.25, 0.25}}};
    const Plan p = plan_split(cost, &qc, 4);
    CHECK(p.first_id[0] != kNone && p.first_id[1] == kNone);
    const auto recs = random_recs(5000, 23);
    std::vector<std::vector<uint32_t>> lists(p.num_ids());
    for (uint32_t i = 0; i < recs.size(); ++i) {
        const uint32_t b = box_of(recs[i]);
        const uint32_t t = subtile_route::targets(
            b, p.n_sub[0], [&](uint32_t k) { return p.mask_of(p.first_id[0] + k); });
        for (uint32_t k = 0; k < p.n_sub[0]; ++k)
            if (t & (1u << k)) lists[p.first_id[0] + k].push_back(i);
    }
    // Per microblock: the records the cull keeps there, in parent order, equal
    // the cull-kept records of the sub list that owns the microblock.
    for (uint32_t m = 0; m < 32; ++m) {
        std::vector<uint32_t> want, got;
        for (uint32_t i = 0; i < recs.size(); ++i)
            if (exact_keep(recs[i]) & (1u << m)) want.push_back(i);
        uint32_t owner = kNone;
        for (uint32_t k = 0; k < p.n_sub[0]; ++k)
            if (p.mask_of(p.first_id[0] + k) & (1u << m)) owner = p.first_id[0] + k;
        CHECK(owner != kNone);
        for (uint32_t i : lists[owner])
            if (exact_keep(recs[i]) & (1u << m)) got.push_back(i);
        CHECK(want == got);
    }
    size_t longest = 0;
    for (uint32_t id = p.num_tiles; id < p.num_ids(); ++id) longest = std::max(longest, lists[id].size());
    std::printf("longest sub list %zu of %zu parent records (%u subs)\n", longest, recs.size(), p.n_sub[0]);
}

// The sort's count step: global mean + parent origin, B1' unit codes as sub-tiles.
void test_global_and_unit_routing() {
    const auto recs = random_recs(5000, 31);
    const float inv = static_cast<float>(1.0 / kFloor);
    for (uint32_t n : {4u, 16u}) {
        std::vector<std::vector<uint32_t>> lists(n);
        for (uint32_t i = 0; i < recs.size(); ++i) {
            const Rec& r = recs[i];
            const uint32_t ox = 32u * (7u + i % 31u), oy = 32u * (3u + i % 17u);
            const float gx = r.mx + static_cast<float>(ox), gy = r.my + static_cast<float>(oy);
            // pack_rec's tile-local mean, as the emit writes it
            Rec loc = r;
            loc.mx = gx - static_cast<float>(ox);
            loc.my = gy - static_cast<float>(oy);
            const uint32_t b = subtile_route::box_mask_global(bits(r.A), bits(r.B), bits(r.C), bits(gx), bits(gy),
                                                              ox, oy, r.q, inv);
            CHECK((exact_keep(loc) & ~b) == 0u);
            const uint32_t t = subtile_route::targets(
                b, n, [&](uint32_t k) { return subtile_unit::mask(subtile_unit::split_code(n, k)); });
            for (uint32_t k = 0; k < n; ++k)
                if (t & (1u << k)) lists[k].push_back(i);
        }
        uint32_t all = 0;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t um = subtile_unit::mask(subtile_unit::split_code(n, k));
            CHECK((all & um) == 0u);  // units partition the tile
            all |= um;
            for (uint32_t m = 0; m < 32; ++m) {
                if (!(um & (1u << m))) continue;
                std::vector<uint32_t> want, got;
                for (uint32_t i = 0; i < recs.size(); ++i)
                    if (exact_keep(recs[i]) & (1u << m)) want.push_back(i);
                for (uint32_t i : lists[k])
                    if (exact_keep(recs[i]) & (1u << m)) got.push_back(i);
                CHECK(want == got);
            }
        }
        CHECK(all == 0xFFFFFFFFu);
    }
    CHECK(subtile_route::plan_first(subtile_route::plan_word(70000u, 16u)) == 70000u);
    CHECK(subtile_route::plan_n(subtile_route::plan_word(70000u, 16u)) == 16u);
}

}  // namespace

int main() {
    test_global_and_unit_routing();
    test_ln_upper();
    test_box_covers_cull();
    test_degenerate();
    test_routing_lossless();
    if (fails) std::fprintf(stderr, "%d failures\n", fails);
    return fails ? 1 : 0;
}
