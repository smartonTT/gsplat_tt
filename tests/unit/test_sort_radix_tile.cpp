// Checks render/kernels/dataflow/sort_radix_tile_algo.h (adaptive stable LSD
// radix, R11) against std::stable_sort on random tiles. Standalone:
//
//   c++ -O2 -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_sort_radix_tile.cpp -o /tmp/test_sort_radix_tile && /tmp/test_sort_radix_tile
//
// Covers every pass count (1..4), ties, all-equal keys, n <= 16, full 32-bit
// key ranges, and float-depth-like keys (the real sort_tile_depth input).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "sort_radix_tile_algo.h"

namespace srt = sort_radix_tile;

static uint32_t fbits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

// Returns false on mismatch.
static bool check(const std::vector<uint32_t>& keys, const char* what, uint32_t* passes_seen) {
    const uint32_t n = static_cast<uint32_t>(keys.size());
    std::vector<uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 1000u);

    std::vector<uint32_t> ref(n);
    std::iota(ref.begin(), ref.end(), 0u);
    std::stable_sort(ref.begin(), ref.end(),
                     [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });

    std::vector<uint32_t> k = keys, v = ids, k2(n + 1, 0xDEADBEEFu), v2(n + 1, 0xDEADBEEFu);
    std::vector<srt::hist_t> hist(srt::HIST_ENTRIES + 1, 0xBEEFu);
    const uint32_t guard = 0xA5A5A5A5u;
    hist[srt::HIST_ENTRIES] = 0xA5A5u;
    k2[n] = guard;
    v2[n] = guard;
    const bool in2 = srt::sort_pairs(k.data(), v.data(), k2.data(), v2.data(), n, hist.data());
    const std::vector<uint32_t>& out = in2 ? v2 : v;
    if (hist[srt::HIST_ENTRIES] != 0xA5A5u || k2[n] != guard || v2[n] != guard) {
        std::printf("FAIL %s n=%u: scratch overrun\n", what, n);
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (out[i] != ids[ref[i]]) {
            std::printf("FAIL %s n=%u at %u: got %u want %u\n", what, n, i, out[i], ids[ref[i]]);
            return false;
        }
    }
    if (n > 16) {
        uint32_t kmin = *std::min_element(keys.begin(), keys.end());
        uint32_t kmax = *std::max_element(keys.begin(), keys.end());
        if (kmin != kmax) {
            const srt::Plan pl = srt::choose_plan(n, srt::bit_length(kmax - kmin));
            passes_seen[pl.passes] = 1;
        }
    }
    return true;
}


// The pre-task-#47 sort_subchunk_materialize depth sort, verbatim in effect:
// n <= 16 insertion sort, else 4 x 8-bit LSD radix through an index array,
// reading the key (word 3 of 32 B record i) through the index every pass.
static std::vector<uint32_t> old_materialize_sort(const std::vector<uint32_t>& recs, uint32_t n) {
    auto key = [&](uint32_t i) { return recs[i * 8u + 3u]; };
    std::vector<uint32_t> a(n), b(n), cnt(256);
    std::iota(a.begin(), a.end(), 0u);
    if (n <= 16u) {
        for (uint32_t i = 1; i < n; ++i) {
            const uint32_t tmp = a[i];
            const uint32_t ki = key(tmp);
            uint32_t j = i;
            while (j > 0 && key(a[j - 1]) > ki) {
                a[j] = a[j - 1];
                --j;
            }
            a[j] = tmp;
        }
        return a;
    }
    for (uint32_t byte = 0; byte < 4u; ++byte) {
        const uint32_t shift = byte * 8u;
        std::fill(cnt.begin(), cnt.end(), 0u);
        for (uint32_t i = 0; i < n; ++i) cnt[(key(a[i]) >> shift) & 0xFFu]++;
        uint32_t sum = 0;
        for (uint32_t c = 0; c < 256u; ++c) {
            const uint32_t t = cnt[c];
            cnt[c] = sum;
            sum += t;
        }
        for (uint32_t i = 0; i < n; ++i) b[cnt[(key(a[i]) >> shift) & 0xFFu]++] = a[i];
        std::swap(a, b);
    }
    return a;
}

// sort_record_ids (materialize) must give exactly the old permutation.
static bool check_records(const std::vector<uint32_t>& keys, const char* what) {
    const uint32_t n = static_cast<uint32_t>(keys.size());
    std::vector<uint32_t> recs(n * 8u);
    for (uint32_t i = 0; i < n * 8u; i++) recs[i] = 0x1000u + i;
    for (uint32_t i = 0; i < n; i++) recs[i * 8u + 3u] = keys[i];
    const std::vector<uint32_t> ref = old_materialize_sort(recs, n);
    std::vector<uint32_t> k(n + 1, 0xDEADBEEFu), v(n + 1, 0xDEADBEEFu);
    std::vector<uint32_t> k2(n + 1, 0xDEADBEEFu), v2(n + 1, 0xDEADBEEFu);
    std::vector<srt::hist_t> hist(srt::HIST_ENTRIES);
    const uint32_t* out = srt::sort_record_ids(recs.data(), n, k.data(), v.data(), k2.data(),
                                               v2.data(), hist.data());
    if (out != v.data() || v[n] != 0xDEADBEEFu || v2[n] != 0xDEADBEEFu) {
        std::printf("FAIL records %s n=%u: result not in v or scratch overrun\n", what, n);
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (out[i] != ref[i]) {
            std::printf("FAIL records %s n=%u at %u: got %u want %u\n", what, n, i, out[i], ref[i]);
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937 rng(12345);
    uint32_t passes_seen[5] = {0, 0, 0, 0, 0};
    uint32_t cases = 0, fails = 0;
    const uint32_t sizes[] = {0, 1, 2, 7, 16, 17, 18, 31, 64, 100, 257, 1000, 2600, 9000, 25000, srt::MAX_N};
    for (uint32_t n : sizes) {
        for (uint32_t trial = 0; trial < 6; trial++) {
            std::vector<uint32_t> keys(n);
            // Full-range random keys.
            for (auto& x : keys) x = rng();
            fails += !check(keys, "full32", passes_seen); cases++;
            // Few distinct keys (many ties).
            for (auto& x : keys) x = 0x3F800000u + (rng() % 5u) * 977u;
            fails += !check(keys, "ties", passes_seen); cases++;
            // All equal.
            std::fill(keys.begin(), keys.end(), 0x41200000u);
            fails += !check(keys, "equal", passes_seen); cases++;
            // Float depths in [lo, hi) like the per-tile depth_bits key.
            const float lo = 0.2f + static_cast<float>(rng() % 1000) * 0.01f;
            const float spans[] = {1.0001f, 1.05f, 1.9f, 8.0f, 200.0f};
            for (float span : spans) {
                std::uniform_real_distribution<float> dist(lo, lo * span);
                for (auto& x : keys) x = fbits(dist(rng));
                fails += !check(keys, "depth", passes_seen); cases++;
            }
            // Narrow integer ranges -> exercise every B (1..32).
            for (uint32_t B = 1; B <= 32; B += 3) {
                const uint32_t base = rng();
                const uint32_t span = B >= 32 ? 0xFFFFFFFFu : ((1u << B) - 1u);
                for (auto& x : keys) {
                    x = base + (span ? rng() % span : 0u);
                }
                fails += !check(keys, "narrow", passes_seen); cases++;
            }
        }
    }
    // Wrap-around base + span keys are not monotonic in the relative key if
    // they overflow; check that too (kmin-relative arithmetic is mod 2^32).
    {
        std::vector<uint32_t> keys(5000);
        for (auto& x : keys) x = 0xFFFFF000u + (rng() % 0x1000u);
        fails += !check(keys, "top", passes_seen); cases++;
    }
    // Materialize record sort (n <= kOverflowL1Cap = 16384) vs the old kernel.
    {
        const uint32_t rsizes[] = {0, 1, 5, 16, 17, 18, 33, 300, 2048, 2049, 8192, 16384, 32768};
        for (uint32_t n : rsizes) {
            for (uint32_t trial = 0; trial < 3; trial++) {
                std::vector<uint32_t> keys(n);
                for (auto& x : keys) x = rng();
                fails += !check_records(keys, "full32"); cases++;
                for (auto& x : keys) x = 0x3F800000u + (rng() % 7u) * 131u;
                fails += !check_records(keys, "ties"); cases++;
                std::fill(keys.begin(), keys.end(), 0x40000000u);
                fails += !check_records(keys, "equal"); cases++;
                std::uniform_real_distribution<float> dist(0.5f, 60.0f);
                for (auto& x : keys) x = fbits(dist(rng));
                fails += !check_records(keys, "depth"); cases++;
                // Task #418: every key width B, so the packed path (B - d + ib <=
                // 32) and its pair-sort fallback both run for every n.
                for (uint32_t B = 1; B <= 32; B += 1) {
                    const uint32_t base = rng();
                    const uint32_t span = B >= 32 ? 0xFFFFFFFFu : ((1u << B) - 1u);
                    for (auto& x : keys) x = base + (span ? rng() % span : 0u);
                    fails += !check_records(keys, "narrow"); cases++;
                }
            }
        }
    }
    std::printf("cases=%u fails=%u passes_seen=[1:%u 2:%u 3:%u 4:%u] HIST_ENTRIES=%u\n",
                cases, fails, passes_seen[1], passes_seen[2], passes_seen[3], passes_seen[4],
                srt::HIST_ENTRIES);
    const bool all_passes = passes_seen[1] && passes_seen[2] && passes_seen[3] && passes_seen[4];
    if (!all_passes) std::printf("FAIL: not every pass count was exercised\n");
    return (fails == 0 && all_passes) ? 0 : 1;
}
