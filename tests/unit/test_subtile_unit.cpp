// Host checks for task #722 (B1' heavy-tile units): render/kernels/dataflow/subtile_unit.h.
// Standalone:
//
//   CXXFLAGS="-Irender/host -Irender/kernels/dataflow" tests/unit/run_cpp.sh tests/unit/test_subtile_unit.cpp
//
//  - quarter masks equal subtile_split.h's; 4 quarters and 16 bands each partition the tile;
//  - each unit's output region (rows x byte span) is exactly the pixels of its mask,
//    16 B aligned;
//  - the heavy list is the count-descending top-cap of the tiles over thr;
//  - lossless: for random records (masks), a unit blends exactly the parent's
//    records restricted to its microblocks, in order, so per-pixel results match.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "subtile_split.h"
#include "subtile_unit.h"

namespace su = subtile_unit;

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

// Microblock of pixel (x, y) in a 32x32 tile.
static uint32_t mb_of(uint32_t x, uint32_t y) { return ((y / 4u) << 2) | (x / 8u); }

static void check_masks() {
    CHECK(su::mask(0) == 0xFFFFFFFFu);
    for (uint32_t n : {4u, 16u}) {
        uint32_t u = 0;
        for (uint32_t k = 0; k < n; ++k) {
            const uint32_t code = su::split_code(n, k);
            const uint32_t m = su::mask(code);
            CHECK(__builtin_popcount(m) == 32 / static_cast<int>(n));
            CHECK((u & m) == 0u);
            u |= m;
            if (n == 4u) CHECK(m == gsplat_tt::subtile::quarter_mask(k));
            CHECK(su::code_of(su::word(1013u, code)) == code);
            CHECK(su::tile_of(su::word(1013u, code)) == 1013u);
        }
        CHECK(u == 0xFFFFFFFFu);
    }
    for (uint32_t code = 0; code < su::kNumCodes; ++code) {
        const uint32_t m = su::mask(code);
        CHECK(su::col_byte0(code) % 16u == 0u && su::col_bytes(code) % 16u == 0u);
        for (uint32_t y = 0; y < 32; ++y)
            for (uint32_t x = 0; x < 32; ++x) {
                const bool in_mask = (m >> mb_of(x, y)) & 1u;
                const bool in_rect = y >= su::row0(code) && y < su::row0(code) + su::rows(code) &&
                                     3u * x >= su::col_byte0(code) &&
                                     3u * x < su::col_byte0(code) + su::col_bytes(code);
                CHECK(in_mask == in_rect);
            }
    }
}

static void check_heavy(uint32_t seed) {
    std::mt19937 rng(seed);
    const uint32_t num_tiles = 20u + rng() % 1000u, num_cores = 120u;
    std::vector<uint32_t> cnt(num_tiles);
    uint64_t total = 0;
    for (auto& c : cnt) {
        c = (rng() % 4u == 0u) ? rng() % 70000u : rng() % 3000u;
        c = std::min(c, 65472u);
        total += c;
    }
    const uint32_t thr = static_cast<uint32_t>(total / num_cores);
    const uint32_t cap = (seed % 3u == 0u) ? 8u : 128u;
    std::vector<uint32_t> h(cap + 1u, 0u);
    uint32_t n = 0;
    for (uint32_t t = 0; t < num_tiles; ++t)
        if (su::is_heavy(cnt[t], thr, 256u)) su::heavy_insert(h.data(), n, cap, cnt[t], t);
    std::vector<std::pair<uint32_t, uint32_t>> ref;  // (-count, tile)
    for (uint32_t t = 0; t < num_tiles; ++t)
        if (cnt[t] > thr && cnt[t] >= 256u) ref.push_back({cnt[t], t});
    std::stable_sort(ref.begin(), ref.end(), [](auto a, auto b) { return a.first > b.first; });
    CHECK(n == std::min<size_t>(ref.size(), cap));
    for (uint32_t i = 0; i < n; ++i) CHECK(h[i] == su::heavy_entry(ref[i].first, ref[i].second));
    CHECK(ref.size() < num_cores);  // count > total / num_cores for fewer than num_cores tiles
}

// A unit walks the parent's records and blends rec_mask & unit mask: per
// microblock the sequence of records equals the parent's.
static void check_lossless(uint32_t seed) {
    std::mt19937 rng(seed);
    const uint32_t L = 1u + rng() % 500u;
    std::vector<uint32_t> rec(L);
    for (auto& m : rec) {
        const int x0 = rng() % 4, y0 = rng() % 8;
        const int x1 = x0 + static_cast<int>(rng() % (4 - x0)), y1 = y0 + static_cast<int>(rng() % (8 - y0));
        m = gsplat_tt::subtile::box_mask(x0, x1, y0, y1) & rng();
    }
    for (uint32_t n : {4u, 16u})
        for (uint32_t mb = 0; mb < 32; ++mb) {
            std::vector<uint32_t> parent_seq, unit_seq;
            for (uint32_t g = 0; g < L; ++g)
                if ((rec[g] >> mb) & 1u) parent_seq.push_back(g);
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t um = su::mask(su::split_code(n, k));
                if (!((um >> mb) & 1u)) continue;
                for (uint32_t g = 0; g < L; ++g)
                    if (((rec[g] & um) >> mb) & 1u) unit_seq.push_back(g);
            }
            CHECK(parent_seq == unit_seq);
        }
}

int main() {
    check_masks();
    for (uint32_t s = 1; s <= 200; ++s) {
        check_heavy(s);
        check_lossless(s);
    }
    CHECK(su::n_units(1000, 100, 1) == 4u && su::n_units(1000, 100, 2) == 16u && su::n_units(300, 100, 2) == 4u);
    if (fails) std::fprintf(stderr, "%d failures\n", fails);
    return fails ? 1 : 0;
}
