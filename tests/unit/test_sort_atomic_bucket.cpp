// Checks the host-side model of the atomic fixed-capacity bucket append
// (task #24: render/kernels/dataflow/sort_bin_atomic.cpp,
// sort_subchunk_materialize.cpp atomic layout, render/host/sort_mover_split.h).
// Standalone:
//
//   c++ -O2 -std=c++17 -Irender/host tests/unit/test_sort_atomic_bucket.cpp
//     -o /tmp/t_atomic && /tmp/t_atomic
//
// Random scenes (heavy-tailed tiles, many equal depth keys) are laid out two
// ways: the prefix-sum layout (cores in order inside each tile) and the atomic
// layout (each core's chunk at the counter value it got, cores arriving in a
// random order). The materialize rebuilds the canonical order from the chunk
// table entries and stable-sorts by key; that must equal the prefix-sum
// layout's stable sort for every tile (byte-identical output). Sorting the
// arrival order directly must NOT (it differs on equal keys), which is why the
// chunk table exists. The worklist must schedule every non-empty tile once,
// and stay under MAX_WORK.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "sort_atomic_layout.h"

using namespace gsplat_tt::sort_atomic;

namespace {

struct Rec {
    uint32_t key;
    uint32_t gid;
};

int failures = 0;
#define CHECK(c, ...)                         \
    do {                                      \
        if (!(c)) {                           \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                \
            failures++;                       \
        }                                     \
    } while (0)

std::vector<uint32_t> stable_by_key(const std::vector<Rec>& recs, std::vector<uint32_t> idx) {
    std::stable_sort(idx.begin(), idx.end(),
                     [&](uint32_t a, uint32_t b) { return recs[a].key < recs[b].key; });
    std::vector<uint32_t> gids;
    gids.reserve(idx.size());
    for (const uint32_t i : idx) gids.push_back(recs[i].gid);
    return gids;
}

}  // namespace

int main() {
    std::mt19937 rng(24);
    int arrival_differs = 0;
    for (int scene = 0; scene < 40; ++scene) {
        const uint32_t num_cores = 110;
        const uint32_t num_tiles = 256;
        // Per core: a gaussian-major run of pairs (g ascending), each with a
        // tile and the gaussian's depth key (few distinct keys => many ties).
        const uint32_t key_range = 1u + (rng() % 64u);
        std::vector<std::vector<std::vector<Rec>>> per(num_cores,
                                                       std::vector<std::vector<Rec>>(num_tiles));
        uint32_t g = 0;
        for (uint32_t c = 0; c < num_cores; ++c) {
            const uint32_t ng = 20 + rng() % 200;
            for (uint32_t i = 0; i < ng; ++i, ++g) {
                const uint32_t key = rng() % key_range;
                const uint32_t fan = 1 + rng() % 6;
                const uint32_t t0 = (rng() % 8 == 0) ? 7u : rng() % num_tiles;  // tile 7 is heavy
                for (uint32_t f = 0; f < fan; ++f) per[c][(t0 + f) % num_tiles].push_back({key, g});
            }
        }
        std::vector<int64_t> counts(num_tiles, 0);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            // Prefix-sum layout: cores in order.
            std::vector<Rec> prefix;
            for (uint32_t c = 0; c < num_cores; ++c)
                prefix.insert(prefix.end(), per[c][t].begin(), per[c][t].end());
            const uint32_t n = static_cast<uint32_t>(prefix.size());
            counts[t] = n;
            if (n == 0) continue;
            std::vector<uint32_t> ident(n);
            std::iota(ident.begin(), ident.end(), 0u);
            const std::vector<uint32_t> want = stable_by_key(prefix, ident);

            // Atomic layout: cores reach the counter in a random order.
            std::vector<uint32_t> order(num_cores);
            std::iota(order.begin(), order.end(), 0u);
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<Rec> bucket(n);
            std::vector<uint32_t> entry(num_cores, 0u);
            uint32_t counter = 0;
            for (const uint32_t c : order) {
                const uint32_t h = static_cast<uint32_t>(per[c][t].size());
                if (h == 0) continue;
                const uint32_t base = counter;  // fetch-and-add's old value
                counter += h;
                for (uint32_t k = 0; k < h; ++k) bucket[base + k] = per[c][t][k];
                entry[c] = (h << 16) | base;
            }
            CHECK(counter == n, "scene %d tile %u counter %u != %u", scene, t, counter, n);
            const std::vector<uint32_t> canon = canonical_order(entry, n);
            CHECK(canon.size() == n, "scene %d tile %u canonical size %zu", scene, t, canon.size());
            CHECK(stable_by_key(bucket, canon) == want,
                  "scene %d tile %u: canonical-order sort differs from the prefix layout", scene, t);
            if (stable_by_key(bucket, ident) != want) arrival_differs++;
        }
        // Worklist: each non-empty tile scheduled exactly once.
        {
            const auto per_core = lpt_whole_tiles(counts, num_tiles, num_cores, 16384u);
            std::vector<int> seen(num_tiles, 0);
            for (const auto& tiles : per_core)
                for (const uint32_t t : tiles) seen[t]++;
            for (uint32_t t = 0; t < num_tiles; ++t)
                CHECK(seen[t] == (counts[t] > 0 ? 1 : 0), "tile %u scheduled %d times", t, seen[t]);
        }
    }
    CHECK(arrival_differs > 0, "arrival-order sort never differed; the test has no ties");
    std::printf("atomic bucket: %s (arrival order would differ on %d tiles)\n",
                failures ? "FAIL" : "PASS", arrival_differs);
    return failures ? 1 : 0;
}
