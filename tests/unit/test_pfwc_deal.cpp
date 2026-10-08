// Task #439: weighted pfwc tile deal (render/host/pfwc_deal.h).
// 1. Every tile is dealt exactly once and core c gets SeqMap.count(c) tiles (seg_base holds).
// 2. Within a core, chunk k is a round-k tile (non-increasing weight down the list).
// 3. On skewed weights LPT's max core load beats the strided deal's, and is within one
//    max tile weight of the mean.
// 4. pack() puts core c's list at c * words.
#include "render/host/pfwc_deal.h"

#include <cstdio>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static void one(uint32_t T, uint32_t C, uint32_t seed) {
    std::mt19937 rng(seed);
    std::gamma_distribution<double> g(2.0, 5000.0);
    std::vector<double> w(T);
    for (auto& x : w) x = 46.2 + 0.05 * g(rng) * (1.0 + 0.5 * ((&x - w.data()) % 7 == 2));
    const auto list = pfwc_deal::lpt(w, C);
    CHECK(list.size() == C);
    const uint32_t base = T / C, rem = T % C;
    std::vector<int> seen(T, 0);
    double lpt_max = 0.0, tot = 0.0, wmax = 0.0;
    for (uint32_t c = 0; c < C; ++c) {
        CHECK(list[c].size() == base + (c < rem ? 1u : 0u));
        double l = 0.0;
        for (std::size_t k = 0; k < list[c].size(); ++k) {
            const uint32_t t = list[c][k];
            CHECK(t < T);
            if (t < T) {
                ++seen[t];
                l += w[t];
            }
            if (k > 0) CHECK(w[list[c][k - 1]] >= w[t]);
        }
        lpt_max = std::max(lpt_max, l);
    }
    for (uint32_t t = 0; t < T; ++t) {
        CHECK(seen[t] == 1);
        tot += w[t];
        wmax = std::max(wmax, w[t]);
    }
    double str_max = 0.0;
    for (uint32_t c = 0; c < C; ++c) {
        double l = 0.0;
        for (uint32_t t = c; t < T; t += C) l += w[t];
        str_max = std::max(str_max, l);
    }
    if (base >= 10) CHECK(lpt_max < str_max);  // few rounds: one heavy tile can set both maxima
    CHECK(lpt_max <= tot / C + wmax);
    std::vector<uint32_t> host;
    const uint32_t words = 64;
    pfwc_deal::pack(list, words, host);
    CHECK(host.size() == std::size_t{words} * C);
    for (uint32_t c = 0; c < C; ++c)
        for (std::size_t k = 0; k < list[c].size(); ++k) CHECK(host[c * words + k] == list[c][k]);
}

int main() {
    one(5989, 120, 1);  // bicycle at ETH 12x10
    one(5989, 110, 2);  // 11x10
    one(5989, 130, 3);
    one(240, 120, 4);   // exact multiple, rem 0
    one(130, 120, 5);   // base 1
    if (fails) std::printf("%d failures\n", fails);
    return fails ? 1 : 0;
}
