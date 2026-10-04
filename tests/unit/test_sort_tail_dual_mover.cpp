// Checks the host work splits behind the dual-data-mover sort tail
// (render/host/sort_mover_split.h; kernels sort_subchunk_materialize.cpp and
// sort_radix_tile.cpp, launched from render/host/sort_device.cpp). Standalone:
//
//   c++ -O2 -std=c++17 -Irender/host tests/unit/test_sort_tail_dual_mover.cpp
//     -o /tmp/t_tail && /tmp/t_tail
//
// Every materialize item and every radix tile writes only its own output, so
// the split is byte-identical iff each unit of work runs exactly once and fits
// the mover's buffers. On random heavy-tailed per-tile counts:
//  - materialize: the dual-mover item multiset equals the single-mover one,
//    and the gather parts cover each over-cap subchunk's records exactly once;
//    BRISC slots never get a whole-tile item above kMatMover0Cap (its L1
//    buffers would overflow); and the busiest slot is lighter than the
//    single-mover busiest core.
//  - radix: the split point stays inside the core's slice and minimizes the
//    busier mover's cost over all split points.
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "sort_mover_split.h"

using namespace gsplat_tt::sort_split;

namespace {

constexpr uint32_t kBucketFit = render_config::kBucketFit;
constexpr uint32_t kCores = 110;
constexpr uint32_t kTiles = 1024;
constexpr uint64_t kGatherWeight = 1;  // build_mat_worklist default

std::vector<int64_t> random_counts(std::mt19937& rng) {
    std::lognormal_distribution<double> ln(7.0, 1.3);
    std::bernoulli_distribution empty(0.05);
    std::vector<int64_t> c(kTiles, 0);
    for (auto& v : c) {
        if (empty(rng)) continue;
        v = std::min<int64_t>(static_cast<int64_t>(ln(rng)), 32768);
    }
    return c;
}

using ItemSet = std::map<std::pair<uint32_t, uint32_t>, int>;

ItemSet items_of(const MatWorkAssignment& a) {
    ItemSet s;
    for (std::size_t i = 0; i + 1 < a.flat.size(); i += 2) s[{a.flat[i], a.flat[i + 1]}]++;
    return s;
}

uint64_t item_cost(const std::vector<int64_t>& counts, uint32_t t, uint32_t w) {
    const uint32_t cnt = static_cast<uint32_t>(counts[t]);
    if (cnt <= render_config::kOverflowL1Cap) return cnt;
    const uint32_t sc = w & 0xFFu, part = w >> 8;
    const uint32_t l_sub = std::min(kBucketFit, cnt - sc * kBucketFit);
    const uint32_t p0 = part * kGatherPartRecs;
    return kGatherWeight * std::min(kGatherPartRecs, l_sub - p0);
}

// Every record of every over-cap subchunk is covered by exactly one part.
int check_parts(const std::vector<int64_t>& counts, const MatWorkAssignment& a) {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> covered;  // (tile, sc) -> recs
    for (std::size_t i = 0; i + 1 < a.flat.size(); i += 2) {
        const uint32_t t = a.flat[i];
        if (static_cast<uint32_t>(counts[t]) <= render_config::kOverflowL1Cap) continue;
        covered[{t, a.flat[i + 1] & 0xFFu}] +=
            item_cost(counts, t, a.flat[i + 1]) / kGatherWeight;
    }
    int bad = 0;
    for (uint32_t t = 0; t < kTiles; ++t) {
        const uint32_t cnt = static_cast<uint32_t>(counts[t]);
        if (cnt <= render_config::kOverflowL1Cap) continue;
        for (uint32_t sc = 0; sc * kBucketFit < cnt; ++sc) {
            const uint32_t l_sub = std::min(kBucketFit, cnt - sc * kBucketFit);
            if (covered[{t, sc}] != l_sub) {
                std::printf("mat: tile %u sc %u parts cover %u of %u\n", t, sc,
                            covered[{t, sc}], l_sub);
                ++bad;
            }
        }
    }
    return bad;
}

int check_mat(const std::vector<int64_t>& counts) {
    const auto one = build_mat_worklist(counts, kTiles, kCores, kBucketFit, 1, 0);
    const auto two = build_mat_worklist(counts, kTiles, kCores, kBucketFit, 2, kMatMover0Cap);
    int bad = 0;
    if (items_of(one) != items_of(two)) {
        std::printf("mat: item sets differ\n");
        ++bad;
    }
    for (const auto& [k, n] : items_of(two)) {
        if (n != 1) {
            std::printf("mat: item (%u,%u) x%d\n", k.first, k.second, n);
            ++bad;
        }
    }
    auto max_load = [&](const MatWorkAssignment& a, uint32_t slots) {
        uint64_t mx = 0;
        for (uint32_t s = 0; s < slots; ++s) {
            uint64_t l = 0;
            for (uint32_t i = 0; i < a.per_core_count[s]; ++i) {
                const uint32_t j = 2 * (a.per_core_offset[s] + i);
                l += item_cost(counts, a.flat[j], a.flat[j + 1]);
            }
            mx = std::max(mx, l);
        }
        return mx;
    };
    for (uint32_t s = 0; s < 2 * kCores; ++s) {
        if ((s & 1u) == 0u) continue;  // NCRISC slot
        for (uint32_t i = 0; i < two.per_core_count[s]; ++i) {
            const uint32_t j = 2 * (two.per_core_offset[s] + i);
            const uint32_t cnt = static_cast<uint32_t>(counts[two.flat[j]]);
            const bool whole_tile = cnt <= render_config::kOverflowL1Cap;
            if (whole_tile && cnt > kMatMover0Cap) {
                std::printf("mat: BRISC slot %u got a %u-record whole-tile item\n", s, cnt);
                ++bad;
            }
        }
    }
    bad += check_parts(counts, two);
    const uint64_t m1 = max_load(one, kCores), m2 = max_load(two, 2 * kCores);
    if (!(m2 < m1)) {
        std::printf("mat: dual busiest slot %llu not below single %llu\n",
                    (unsigned long long)m2, (unsigned long long)m1);
        ++bad;
    }
    return bad;
}

int check_radix(std::mt19937& rng, const std::vector<int64_t>& counts) {
    std::uniform_int_distribution<uint32_t> tile(0, kTiles - 1), len(0, 24);
    int bad = 0;
    for (int r = 0; r < 200; ++r) {
        std::vector<uint32_t> ids(8);
        for (auto& v : ids) v = tile(rng);
        const uint32_t start = 8u, count = len(rng);
        std::vector<uint32_t> flat(start, 0);
        for (uint32_t i = 0; i < count; ++i) flat.push_back(tile(rng));
        flat.push_back(0);
        const uint32_t k = radix_split_point(flat, start, count, counts);
        if (k > count) {
            std::printf("radix: k=%u > count=%u\n", k, count);
            ++bad;
            continue;
        }
        auto c = [&](uint32_t i) -> uint64_t {
            const uint64_t n = counts[flat[start + i]];
            return n <= 16 ? 32 : n + 256;
        };
        auto busier = [&](uint32_t kk) {
            uint64_t a = 0, b = 0;
            for (uint32_t i = 0; i < count; ++i) (i < kk ? a : b) += c(i);
            return std::max(a, b);
        };
        for (uint32_t kk = 0; kk <= count; ++kk) {
            if (busier(kk) < busier(k)) {
                std::printf("radix: k=%u not optimal (k=%u better)\n", k, kk);
                ++bad;
                break;
            }
        }
    }
    return bad;
}

// Task #144: the calibrated one-launch LPT (select off) keeps the legacy item
// set and the BRISC cap, and under its own cost model is no busier than the
// record-count assignment (never busier per scene, lighter summed over scenes:
// in many scenes one big item sets both).
uint64_t g_calib_sum = 0, g_legacy_sum = 0;

uint64_t model_cost(const MatCostModel& cm, const std::vector<int64_t>& counts, uint32_t t,
                    uint32_t w, int r) {
    const uint32_t cnt = static_cast<uint32_t>(counts[t]);
    if (cnt <= render_config::kOverflowL1Cap) return cm.whole_ns(r, cnt, kMatMover0Cap);
    const uint32_t l_sub = std::min(kBucketFit, cnt - (w & 0xFFu) * kBucketFit);
    return cm.big_ns(cnt, l_sub);
}

int check_mat_calib(const std::vector<int64_t>& counts) {
    const MatCostModel cm = parse_mat_cost(kMatCostDefault);
    const MatCostModel off = parse_mat_cost("0");
    const auto leg = build_mat_worklist(counts, kTiles, kCores, kBucketFit, 2, kMatMover0Cap,
                                        true, false, off);
    const auto cal = build_mat_worklist(counts, kTiles, kCores, kBucketFit, 2, kMatMover0Cap,
                                        true, false, cm);
    int bad = 0;
    if (items_of(leg) != items_of(cal)) {
        std::printf("calib: item sets differ\n");
        ++bad;
    }
    auto busiest = [&](const MatWorkAssignment& a) {
        uint64_t mx = 0;
        for (uint32_t s = 0; s < 2 * kCores; ++s) {
            uint64_t l = 0;
            for (uint32_t i = 0; i < a.per_core_count[s]; ++i) {
                const uint32_t j = 2 * (a.per_core_offset[s] + i);
                const uint32_t cnt = static_cast<uint32_t>(counts[a.flat[j]]);
                if ((s & 1u) && (cnt > kMatMover0Cap)) {
                    std::printf("calib: BRISC slot %u got a %u-record item\n", s, cnt);
                    ++bad;
                }
                l += model_cost(cm, counts, a.flat[j], a.flat[j + 1], static_cast<int>(s & 1u));
            }
            mx = std::max(mx, l);
        }
        return mx;
    };
    const uint64_t ml = busiest(leg), mc = busiest(cal);
    g_calib_sum += mc;
    g_legacy_sum += ml;
    if (mc > ml) {
        std::printf("calib: busiest %llu above legacy %llu\n", (unsigned long long)mc,
                    (unsigned long long)ml);
        ++bad;
    }
    return bad;
}

int check_cost_parse() {
    int bad = 0;
    if (parse_mat_cost("0").on || parse_mat_cost("").on || parse_mat_cost(nullptr).on) ++bad;
    if (!parse_mat_cost(kMatCostDefault).on) ++bad;
    const MatCostModel m = parse_mat_cost("1,2,3,4,5,6,7,8,9");
    if (!m.on || m.whole_ns(0, 10, 6144) != 21000u || m.whole_ns(1, 10, 6144) != 43000u ||
        m.whole_ns(0, 7000, 6144) != 42005000u || m.big_ns(2, 1) != 32000u)
        ++bad;
    if (bad) std::printf("cost parse: %d failures\n", bad);
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(35);
    int bad = 0, cases = 0;
    for (int trial = 0; trial < 60; ++trial) {
        const auto counts = random_counts(rng);
        bad += check_mat(counts);
        bad += check_radix(rng, counts);
        bad += check_mat_calib(counts);
        ++cases;
    }
    bad += check_cost_parse();
    if (!(g_calib_sum < g_legacy_sum)) {
        std::printf("calib: summed busiest %llu not below legacy %llu\n",
                    (unsigned long long)g_calib_sum, (unsigned long long)g_legacy_sum);
        ++bad;
    }
    std::printf("%s: %d random scenes, %d failures\n", bad ? "FAIL" : "PASS", cases, bad);
    return bad ? 1 : 0;
}
