// Host checks for task #719 (B1 adaptive heavy-tile split): render/host/subtile_split.h.
// Standalone:
//
//   CXXFLAGS=-Irender/host tests/unit/run_cpp.sh tests/unit/test_subtile_split.cpp
//
//  - quarter and 8x8 masks partition the 32 microblocks;
//  - a split tile's sub-tile masks partition the tile; unsplit tiles keep it whole;
//  - only tiles over total / num_cores split, and only heavy quarters go to 8x8;
//  - lossless: for random parent lists (records with microblock boxes and cull
//    masks inside them), every microblock sees the same records in the same
//    order from its sub-tile's list as from the parent list;
//  - the claim order (blend_claim_order.h) takes the mixed id space unchanged.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "blend_claim_order.h"
#include "subtile_split.h"

using namespace gsplat_tt;

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

static void check_masks() {
    uint32_t u = 0;
    for (uint32_t q = 0; q < 4; ++q) {
        CHECK(__builtin_popcount(subtile::quarter_mask(q)) == 8);
        CHECK((u & subtile::quarter_mask(q)) == 0);
        u |= subtile::quarter_mask(q);
    }
    CHECK(u == subtile::kFullMask);
    u = 0;
    for (uint32_t ey = 0; ey < 4; ++ey)
        for (uint32_t ex = 0; ex < 4; ++ex) {
            const uint32_t m = subtile::eighth_mask(ex, ey);
            CHECK(__builtin_popcount(m) == 2);
            CHECK((u & m) == 0);
            CHECK((m & subtile::quarter_mask((ex >> 1) | ((ey >> 1) << 1))) == m);
            u |= m;
        }
    CHECK(u == subtile::kFullMask);
    // quarter 0 = pixels x 0..15, y 0..15 = mx 0..1, my 0..3
    CHECK(subtile::quarter_mask(0) == subtile::box_mask(0, 1, 0, 3));
    CHECK(subtile::eighth_mask(3, 3) == subtile::box_mask(3, 3, 6, 7));
}

static void check_plan(uint32_t num_tiles, uint32_t num_cores, uint32_t seed, bool quarters) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> cost(num_tiles);
    std::vector<std::array<double, 4>> qc(num_tiles);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cost[t] = u(rng) < 0.2 ? 0.0 : std::pow(u(rng), 6.0) * 1e5;
        double s = 0;
        for (auto& q : qc[t]) s += (q = u(rng) * u(rng));
        for (auto& q : qc[t]) q = s > 0 ? q / s * cost[t] : 0.0;
    }
    const subtile::Plan p = subtile::plan_split(cost, quarters ? &qc : nullptr, num_cores);
    double total = 0;
    for (double c : cost) total += c;
    const double thr = total / num_cores;
    CHECK(p.num_tiles == num_tiles);
    uint32_t next = num_tiles;
    for (uint32_t t = 0; t < num_tiles; ++t) {
        const bool split = cost[t] > thr;
        CHECK((p.first_id[t] != subtile::kNone) == split);
        if (!split) {
            CHECK(p.n_sub[t] == 0);
            CHECK(p.mask_of(t) == subtile::kFullMask);
            continue;
        }
        CHECK(p.first_id[t] == next);  // contiguous, in tile order
        next += p.n_sub[t];
        uint32_t u32 = 0;
        uint32_t n8 = 0;
        for (uint32_t k = 0; k < p.n_sub[t]; ++k) {
            const uint32_t id = p.first_id[t] + k;
            CHECK(p.parent_of(id) == t);
            CHECK((u32 & p.mask_of(id)) == 0);
            u32 |= p.mask_of(id);
            n8 += __builtin_popcount(p.mask_of(id)) == 2;
        }
        CHECK(u32 == subtile::kFullMask);
        uint32_t heavy = 0;
        for (uint32_t q = 0; q < 4; ++q) heavy += (quarters ? qc[t][q] : cost[t] / 4) > thr;
        CHECK(n8 == heavy * 4);
        CHECK(p.n_sub[t] == 4 + heavy * 3);
    }
    CHECK(p.num_ids() == next);

    // Claim order over the mixed ids: counts per id, split parents empty.
    std::vector<int64_t> counts(p.num_ids(), 0);
    for (uint32_t id = 0; id < p.num_ids(); ++id) {
        const uint32_t t = p.parent_of(id);
        if (id < num_tiles && p.first_id[t] != subtile::kNone) continue;
        counts[id] = static_cast<int64_t>(cost[t] / (id < num_tiles ? 1 : p.n_sub[t])) + (cost[t] > 0);
    }
    std::vector<uint32_t> flat, off, cnt;
    blend_order::desc_round_robin(counts, p.num_ids(), num_cores, flat, off, cnt);
    std::vector<int> seen(p.num_ids(), 0);
    for (uint32_t id : flat) seen[id]++;
    for (uint32_t id = 0; id < p.num_ids(); ++id) CHECK(seen[id] == (counts[id] > 0 ? 1 : 0));
}

// Lossless: random parent list of records (microblock box + cull mask inside the
// box); emit through for_each_target; each microblock's record sequence (records
// whose cull mask has its bit) is the same from the sub-tile list as from the parent.
static void check_lossless(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> mxd(0, 3), myd(0, 7);
    std::uniform_int_distribution<uint32_t> bits;
    // one heavy tile among cheap ones, so tile 0 splits; quarters vary
    std::vector<double> cost = {1e6, 1, 1, 1, 1, 1, 1, 1};
    std::vector<std::array<double, 4>> qc(cost.size(), {0, 0, 0, 0});
    qc[0] = {seed % 2 ? 6e5 : 1e5, 1e5, seed % 3 ? 2e5 : 1e5, 1e5};
    const subtile::Plan p = subtile::plan_split(cost, &qc, 4);
    CHECK(p.first_id[0] != subtile::kNone);
    const uint32_t n = 3000;
    std::vector<uint32_t> rec_mask(n);
    std::vector<std::vector<uint32_t>> lists(p.num_ids());
    for (uint32_t r = 0; r < n; ++r) {
        int x0 = mxd(rng), x1 = mxd(rng), y0 = myd(rng), y1 = myd(rng);
        if (x0 > x1) std::swap(x0, x1);
        if (y0 > y1) std::swap(y0, y1);
        const uint32_t bm = subtile::box_mask(x0, x1, y0, y1);
        rec_mask[r] = bm & bits(rng);  // the cull keeps a subset of the box
        subtile::for_each_target(p, 0, bm, [&](uint32_t id) { lists[id].push_back(r); });
    }
    CHECK(lists[0].empty());
    for (uint32_t m = 0; m < 32; ++m) {
        std::vector<uint32_t> want, got;
        for (uint32_t r = 0; r < n; ++r)
            if (rec_mask[r] >> m & 1u) want.push_back(r);
        uint32_t owners = 0;
        for (uint32_t k = 0; k < p.n_sub[0]; ++k) {
            const uint32_t id = p.first_id[0] + k;
            if (!(p.mask_of(id) >> m & 1u)) continue;
            ++owners;
            for (uint32_t r : lists[id])
                if ((rec_mask[r] >> m) & 1u) got.push_back(r);
        }
        CHECK(owners == 1);
        CHECK(got == want);
    }
    // no sub-tile list holds a record whose box misses it
    for (uint32_t k = 0; k < p.n_sub[0]; ++k) CHECK(lists[p.first_id[0] + k].size() < n);
}

// #744: B1 route plan in B1' unit codes (4 quarters or 16 bands), the device
// plan words, the id budget and the over-capacity rule.
static void check_plan_units(uint32_t num_tiles, uint32_t num_cores, uint32_t seed) {
    std::vector<double> cost(num_tiles);
    uint32_t r = seed * 2654435761u;
    for (auto& c : cost) {
        r = r * 1664525u + 1013904223u;
        c = static_cast<double>(r >> 20);
    }
    cost[seed % num_tiles] = 200000.0;  // over the bucket cap
    const double cap = 65472.0;
    for (const uint32_t max_ids : {num_tiles + 4u, num_tiles + 40u, num_tiles + 4096u}) {
        const subtile::Plan p = subtile::plan_units(cost, num_cores, max_ids, cap);
        CHECK(p.num_ids() <= max_ids);
        CHECK(p.code.size() == p.parent.size());
        const std::vector<uint32_t> words = subtile::plan_words(p);
        CHECK(words.size() == num_tiles);
        double total = 0.0;
        for (double c : cost) total += c;
        const double thr = total / num_cores;
        uint32_t nsplit = 0;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const bool split = p.first_id[t] != subtile::kNone;
            if (!split) {
                CHECK(words[t] == 0u);
                continue;
            }
            ++nsplit;
            CHECK(cost[t] > thr || cost[t] > cap);
            CHECK(p.n_sub[t] == 4u || p.n_sub[t] == 16u);
            if (cost[t] > cap && max_ids == num_tiles + 4096u) CHECK(p.n_sub[t] == 16u);
            CHECK(subtile_route::plan_first(words[t]) == p.first_id[t]);
            CHECK(subtile_route::plan_n(words[t]) == p.n_sub[t]);
            uint32_t u = 0;
            for (uint32_t k = 0; k < p.n_sub[t]; ++k) {
                const uint32_t id = p.first_id[t] + k;
                CHECK(p.parent_of(id) == t);
                CHECK(p.code[id - num_tiles] == subtile_unit::split_code(p.n_sub[t], k));
                CHECK(p.mask_of(id) == subtile_unit::mask(p.code[id - num_tiles]));
                CHECK((u & p.mask_of(id)) == 0u);
                u |= p.mask_of(id);
            }
            CHECK(u == subtile::kFullMask);
        }
        // the heaviest tile always fits the smallest budget (4 extra ids: quarters)
        CHECK(p.first_id[seed % num_tiles] != subtile::kNone);
        if (max_ids == num_tiles + 4096u) {
            uint32_t want = 0;
            for (uint32_t t = 0; t < num_tiles; ++t) want += (cost[t] > thr || cost[t] > cap) ? 1u : 0u;
            CHECK(nsplit == want);
        }
    }
}

int main() {
    check_masks();
    for (uint32_t s = 1; s <= 6; ++s) {
        check_plan(70, 120, s, s % 2);    // 1/4 res
        check_plan(260, 120, s, s % 2);   // 1/2 res
        check_plan(1014, 120, s, s % 2);  // full res
        check_lossless(s);
        check_plan_units(70, 120, s);
        check_plan_units(260, 120, s);
        check_plan_units(1014, 120, s);
    }
    {  // all empty: nothing splits
        const subtile::Plan p = subtile::plan_split(std::vector<double>(16, 0.0), nullptr, 120);
        CHECK(p.num_ids() == 16);
    }
    if (fails) {
        std::fprintf(stderr, "%d failures\n", fails);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
