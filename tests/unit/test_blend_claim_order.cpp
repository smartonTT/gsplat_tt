// Host checks for task #188 (GSPLAT_TT_BLEND_CLAIM_DESC): the blend per-core
// lists of render/host/blend_claim_order.h. Standalone:
//
//   CXXFLAGS=-Irender/host tests/unit/run_cpp.sh tests/unit/test_blend_claim_order.cpp
//
//  - every non-empty tile appears exactly once, empty tiles never;
//  - core c's k-th tile is global rank c + k*n of the (count, tile id)
//    descending order;
//  - replaying the blend reader's rank-interleaved claim decode
//    (reader_alpha_blend_mb_devcull.cpp: claim -> rank, core_sel -> flat)
//    over claims 0..m-1 visits the tiles in exact descending count order,
//    for m below, at and above num_cores and for m not a multiple of it.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "blend_claim_order.h"

static int fails = 0;
#define CHECK(c)                                                      \
    do {                                                              \
        if (!(c)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                  \
        }                                                             \
    } while (0)

// The reader's claim decode, verbatim in logic.
static std::vector<uint32_t> replay_claims(
    const std::vector<uint32_t>& flat, const std::vector<uint32_t>& off,
    const std::vector<uint32_t>& cnt, uint32_t num_cores) {
    uint32_t total = 0;
    for (uint32_t c : cnt) total += c;
    std::vector<uint32_t> order;
    uint32_t rank = 0, rank_base = 0, rank_width = num_cores;
    for (uint32_t claim = 0; claim < total; ++claim) {
        while (claim - rank_base >= rank_width) {
            rank_base += rank_width;
            ++rank;
            rank_width = 0;
            for (uint32_t c = 0; c < num_cores; ++c) rank_width += (cnt[c] > rank) ? 1u : 0u;
        }
        uint32_t pos = claim - rank_base, core_sel = 0;
        for (uint32_t c = 0; c < num_cores; ++c) {
            if (cnt[c] > rank) {
                if (pos == 0u) { core_sel = c; break; }
                --pos;
            }
        }
        order.push_back(flat[off[core_sel] + rank]);
    }
    return order;
}

static void run_case(uint32_t num_tiles, uint32_t num_cores, uint32_t seed, double empty_frac) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int64_t> counts(num_tiles, 0);
    uint32_t nonempty = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        if (u(rng) < empty_frac) continue;
        // heavy tail plus duplicates (tie-break on tile id)
        counts[t] = 1 + static_cast<int64_t>(std::pow(u(rng), 4.0) * 30000.0) / 7 * 7;
        ++nonempty;
    }
    std::vector<uint32_t> flat, off, cnt;
    gsplat_tt::blend_order::desc_round_robin(counts, num_tiles, num_cores, flat, off, cnt);
    CHECK(flat.size() == nonempty);
    CHECK(off.size() == num_cores && cnt.size() == num_cores);
    std::vector<int> seen(num_tiles, 0);
    for (uint32_t id : flat) {
        CHECK(id < num_tiles && counts[id] > 0);
        if (id < num_tiles) seen[id]++;
    }
    for (uint32_t t = 0; t < num_tiles; t++) CHECK(seen[t] == (counts[t] > 0 ? 1 : 0));
    for (uint32_t c = 0; c + 1 < num_cores; c++) {
        CHECK(off[c] + cnt[c] == off[c + 1]);
        CHECK(cnt[c] >= cnt[c + 1] && cnt[c] - cnt[c + 1] <= 1);
    }
    const std::vector<uint32_t> order = replay_claims(flat, off, cnt, num_cores);
    CHECK(order.size() == nonempty);
    for (size_t i = 0; i + 1 < order.size(); i++) {
        const int64_t a = counts[order[i]], b = counts[order[i + 1]];
        CHECK(a > b || (a == b && order[i] > order[i + 1]));
    }
    // core c, position k == global rank c + k*n
    for (uint32_t c = 0; c < num_cores; c++)
        for (uint32_t k = 0; k < cnt[c]; k++) CHECK(flat[off[c] + k] == order[c + k * num_cores]);
}

int main() {
    run_case(1024, 110, 1, 0.1);
    run_case(1024, 110, 2, 0.0);
    run_case(1024, 130, 3, 0.5);
    run_case(64, 110, 4, 0.2);    // fewer tiles than cores
    run_case(220, 110, 5, 0.0);   // exact multiple
    run_case(1, 8, 6, 0.0);
    run_case(16, 8, 7, 1.0);      // all empty
    if (fails) { std::fprintf(stderr, "%d failures\n", fails); return 1; }
    std::printf("ok\n");
    return 0;
}
