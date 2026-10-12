// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #519 (GSPLAT_TT_MAT_EARLY_CULL): host checks of the id math of
// sort_subchunk_materialize.cpp's early-cull paths.
//  - whole tile: record_keys + sort_ids_gathered == sort_record_ids (the keys
//    are taken before the TRISC cull overwrites word 3).
//  - big item: select_bins, select_compact_ids (in place), the candidates'
//    keys in gather order, sort_ids_gathered, ranks [lo, hi) at candidate
//    positions [lo - base, hi - base) == the whole tile's stable sort ids, for
//    every subchunk of the tile.
// Build and run from the repo root:
//   c++ -O2 -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_mat_early_cull.cpp -o /tmp/t519 && /tmp/t519

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "sort_onelaunch_algo.h"

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                       \
    do {                                                    \
        if (!(c)) {                                         \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
            ++g_fail;                                       \
        }                                                   \
    } while (0)

std::vector<uint32_t> make_keys(int kind, uint32_t n, std::mt19937& rng) {
    std::vector<uint32_t> k(n);
    for (uint32_t i = 0; i < n; ++i) {
        switch (kind) {
            case 0: k[i] = rng(); break;  // uniform u32
            case 1:                       // cluster + outliers
                k[i] = (rng() % 64u == 0u) ? rng() : 0x3F800000u + (rng() % 5000u);
                break;
            case 2: k[i] = rng() % 7u; break;  // few distinct
            case 3: k[i] = 12345u; break;      // all equal
            default: {                         // fp32 depth bits
                const float d = 0.5f + std::exponential_distribution<float>(0.3f)(rng);
                std::memcpy(&k[i], &d, 4);
            }
        }
    }
    return k;
}

std::vector<uint32_t> stable_order(const std::vector<uint32_t>& k) {
    std::vector<uint32_t> ord(k.size());
    std::iota(ord.begin(), ord.end(), 0u);
    std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return k[a] < k[b]; });
    return ord;
}

void test_whole_tile() {
    std::mt19937 rng(519);
    sort_radix_tile::hist_t hist[sort_radix_tile::HIST_ENTRIES];
    const uint32_t sizes[] = {1u, 5u, 16u, 17u, 100u, 4097u, 8192u, 12000u, 16384u};
    for (int kind = 0; kind < 5; ++kind) {
        for (const uint32_t n : sizes) {
            const std::vector<uint32_t> key = make_keys(kind, n, rng);
            std::vector<uint32_t> recs(8u * n);
            for (uint32_t i = 0; i < n; ++i) {
                for (uint32_t w = 0; w < 8u; ++w) recs[8u * i + w] = rng();
                recs[8u * i + 3u] = key[i];
            }
            std::vector<uint32_t> s1(4u * n), s2(4u * n);
            const uint32_t* ref = sort_radix_tile::sort_record_ids(
                recs.data(), n, s1.data(), s1.data() + n, s1.data() + 2u * n, s1.data() + 3u * n,
                hist);
            uint32_t kmin = 0, kmax = 0;
            sort_radix_tile::record_keys(recs.data(), n, s2.data(), &kmin, &kmax);
            // The cull overwrites word 3 after the keys were taken.
            for (uint32_t i = 0; i < n; ++i) recs[8u * i + 3u] = 0xDEADBEEFu;
            const uint32_t* got = sort_radix_tile::sort_ids_gathered(
                s2.data(), n, kmin, kmax, s2.data() + n, s2.data() + 2u * n, s2.data() + 3u * n,
                hist);
            const std::vector<uint32_t> ord = stable_order(key);
            uint32_t bad = 0;
            for (uint32_t i = 0; i < n; ++i) bad += (got[i] != ref[i] || got[i] != ord[i]) ? 1u : 0u;
            CHECK(bad == 0u, "whole kind %d n %u: %u ids differ", kind, n, bad);
        }
    }
}

void test_big_item() {
    std::mt19937 rng(5190);
    sort_radix_tile::hist_t hist[sort_radix_tile::HIST_ENTRIES];
    const uint32_t sizes[] = {8193u, 16385u, 20000u, 25700u, 32768u};
    const uint32_t SC = 8192u;    // kBucketFit: one item per subchunk
    const uint32_t CAP = 16384u;  // candidates: CB_BUCKET records / CB_SLAB 4 u32 each
    uint32_t items = 0, fallback = 0, fallback_spread = 0, max_m_x100 = 0;
    for (int kind = 0; kind < 5; ++kind) {
        for (const uint32_t n : sizes) {
            const std::vector<uint32_t> k = make_keys(kind, n, rng);
            const std::vector<uint32_t> ord = stable_order(k);
            for (uint32_t lo = 0; lo < n; lo += SC) {
                const uint32_t hi = lo + std::min(SC, n - lo);
                ++items;
                const sort_ol::Select s = sort_ol::select_bins(k.data(), n, lo, hi, hist);
                CHECK(s.base <= lo && s.base + s.m >= hi, "kind %d n %u [%u,%u): candidates miss ranks",
                      kind, n, lo, hi);
                if (s.m > CAP) {  // the kernel keeps the full sort for these
                    ++fallback;
                    if (kind == 0 || kind == 1 || kind == 4) ++fallback_spread;
                    continue;
                }
                if (kind == 0 || kind == 1 || kind == 4) {
                    max_m_x100 = std::max(max_m_x100, 100u * s.m / (hi - lo));
                }
                std::vector<uint32_t> cv = k;  // in place over k, as in CB_BSORT
                const uint32_t m = sort_ol::select_compact_ids(cv.data(), n, s);
                CHECK(m == s.m, "kind %d n %u [%u,%u): compact %u != m %u", kind, n, lo, hi, m, s.m);
                bool mono = true;
                for (uint32_t j = 1; j < m; ++j) mono = mono && cv[j - 1u] < cv[j];
                CHECK(mono, "candidate ids not increasing");
                // Gathered records' keys (buck word 3) into the slab arrays.
                std::vector<uint32_t> sl(4u * m);
                uint32_t kmin = 0xFFFFFFFFu, kmax = 0u;
                for (uint32_t j = 0; j < m; ++j) {
                    sl[j] = k[cv[j]];
                    kmin = std::min(kmin, sl[j]);
                    kmax = std::max(kmax, sl[j]);
                }
                const uint32_t* res = sort_radix_tile::sort_ids_gathered(
                    sl.data(), m, kmin, kmax, sl.data() + m, sl.data() + 2u * m, sl.data() + 3u * m,
                    hist);
                // ids (candidate = buck positions) overwrite cv in CB_BSORT.
                std::vector<uint32_t> ids(hi - lo);
                for (uint32_t i = lo; i < hi; ++i) ids[i - lo] = res[i - s.base];
                uint32_t bad = 0;
                for (uint32_t i = lo; i < hi; ++i) bad += (cv[ids[i - lo]] != ord[i]) ? 1u : 0u;
                CHECK(bad == 0u, "big kind %d n %u [%u,%u): %u ids differ", kind, n, lo, hi, bad);
            }
        }
    }
    CHECK(fallback_spread == 0u, "%u spread-key items over the candidate cap", fallback_spread);
    std::printf("big: %u items, %u full-sort fallbacks (tied keys), max candidates %u%% of L\n",
                items, fallback, max_m_x100);
}

}  // namespace

int main() {
    test_whole_tile();
    test_big_item();
    if (g_fail != 0) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
