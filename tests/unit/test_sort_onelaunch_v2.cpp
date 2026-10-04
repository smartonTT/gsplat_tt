// Checks the task #124 one-launch sort v2 pieces (lever A):
// render/kernels/dataflow/sort_onelaunch_algo.h and the OL_MAT_SELECT work
// items of render/host/sort_mover_split.h. Standalone:
//
//   c++ -O2 -std=c++17 -Irender/kernels/dataflow -Irender/host
//     tests/unit/test_sort_onelaunch_v2.cpp -o /tmp/t_ol2 && /tmp/t_ol2
//
//  - select_ranks gives exactly the ids of stable ranks [lo, hi) of a whole
//    tile stable sort, for every 4096-record part of every 8192-record
//    subchunk, on uniform, clustered-with-outliers, few-distinct, all-equal
//    and fp32-depth keys, N up to 32768; its candidate count stays small when
//    outliers stretch the key range (the second histogram level);
//  - the emit's per-tile record runs (ring_run_start / ring_drain, modelled
//    like sort_bin_onelaunch.cpp's emit with two movers sharing every tile)
//    write the same bucket image as one write per record, never touch a slot
//    outside the mover's own cursors, drop the slots past tile_cap, and keep
//    every write inside one 2 KB record page;
//  - build_mat_worklist(onelaunch, ol_select) splits every over-cap subchunk
//    into kOlMatPartRecs parts covering it exactly, keeps them NCRISC-only,
//    and its largest item is <= 1.5x the mean slot on a heavy-tailed frame.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "sort_mover_split.h"
#include "sort_onelaunch_algo.h"

using gsplat_tt::sort_split::build_mat_worklist;
using gsplat_tt::sort_split::kMatMover0Cap;
using gsplat_tt::sort_split::kOlKindGather;
using gsplat_tt::sort_split::kOlKindSort;
using gsplat_tt::sort_split::kOlMatPartRecs;
using gsplat_tt::sort_split::kOlSharedSlots;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                   \
            std::printf("\n");                          \
            ++g_fail;                                   \
        }                                               \
    } while (0)

// ── select ────────────────────────────────────────────────────────────────
std::vector<uint32_t> make_keys(int kind, uint32_t n, std::mt19937& rng) {
    std::vector<uint32_t> k(n);
    for (uint32_t i = 0; i < n; ++i) {
        switch (kind) {
            case 0: k[i] = rng(); break;                          // uniform u32
            case 1:                                               // cluster + outliers
                k[i] = (rng() % 64u == 0u) ? rng() : 0x3F800000u + (rng() % 5000u);
                break;
            case 2: k[i] = rng() % 7u; break;                     // few distinct
            case 3: k[i] = 12345u; break;                         // all equal
            default: {                                            // fp32 depth bits
                const float d = 0.5f + std::exponential_distribution<float>(0.3f)(rng);
                std::memcpy(&k[i], &d, 4);
            }
        }
    }
    return k;
}

void test_select() {
    std::mt19937 rng(124);
    sort_radix_tile::hist_t hist[sort_radix_tile::HIST_ENTRIES];
    const uint32_t sizes[] = {1u, 17u, 4097u, 16385u, 25700u, 32768u};
    const uint32_t SC = 8192u;
    uint32_t max_m_ratio_x100 = 0;
    for (int kind = 0; kind < 5; ++kind) {
        for (const uint32_t n : sizes) {
            const std::vector<uint32_t> k = make_keys(kind, n, rng);
            std::vector<uint32_t> ord(n);
            std::iota(ord.begin(), ord.end(), 0u);
            std::stable_sort(ord.begin(), ord.end(),
                             [&](uint32_t a, uint32_t b) { return k[a] < k[b]; });
            std::vector<uint32_t> scratch(4u * n), out(n);
            for (uint32_t sc0 = 0; sc0 < n; sc0 += SC) {
                const uint32_t ls = std::min(SC, n - sc0);
                for (uint32_t p0 = 0; p0 < ls; p0 += kOlMatPartRecs) {
                    const uint32_t lo = sc0 + p0, hi = lo + std::min(kOlMatPartRecs, ls - p0);
                    // out aliases a copy of k, as in the kernel.
                    std::vector<uint32_t> kk = k;
                    sort_ol::select_ranks(kk.data(), n, lo, hi, scratch.data(), scratch.data() + n,
                                          scratch.data() + 2u * n, scratch.data() + 3u * n,
                                          kk.data(), hist);
                    uint32_t bad = 0;
                    for (uint32_t i = lo; i < hi; ++i) bad += (kk[i - lo] != ord[i]) ? 1u : 0u;
                    CHECK(bad == 0u, "select kind %d n %u ranks [%u,%u): %u wrong", kind, n, lo, hi,
                          bad);
                    const sort_ol::Select s = sort_ol::select_bins(k.data(), n, lo, hi, hist);
                    std::vector<uint32_t> ck(n), cv(n);
                    CHECK(sort_ol::select_collect(k.data(), n, s, ck.data(), cv.data()) == s.m,
                          "collect count != m");
                    CHECK(s.base <= lo && s.base + s.m >= hi, "candidates miss ranks");
                    if ((kind == 0 || kind == 1 || kind == 4) && std::getenv("T124_DBG") &&
                        s.m > 2u * (hi - lo)) {
                        std::printf("kind %d n %u [%u,%u) m %u base %u klo %u khi %u\n", kind, n,
                                    lo, hi, s.m, s.base, s.klo, s.khi);
                    }
                    if (kind == 0 || kind == 1 || kind == 4) {
                        max_m_ratio_x100 =
                            std::max(max_m_ratio_x100, 100u * s.m / (hi - lo));
                    }
                }
            }
        }
    }
    // Spread keys (with outliers too): candidates stay within 2x the part
    // (the stop rule), so the per-item sort is ~1/8 of a 32k-record tile.
    CHECK(max_m_ratio_x100 <= 200u, "select candidates %u%% of the part", max_m_ratio_x100);
    std::printf("select: max candidates %u%% of the part on spread keys\n", max_m_ratio_x100);
}

// ── emit record runs ──────────────────────────────────────────────────────
constexpr uint32_t REC_PAGE_RECS = 64;

// One mover's emit of records (tile, id) from cursors `cur`: the bucket image
// with one write per record (ring 0) or with runs of R (the kernel's
// process_batch + final drain). Writes are logged as (first slot, count).
struct Write { uint32_t slot, n; };
void emit(const std::vector<std::pair<uint32_t, uint32_t>>& recs, std::vector<uint32_t> cur,
          uint32_t cap, uint32_t R, std::vector<uint32_t>& bucket, std::vector<Write>& log) {
    const uint32_t T = static_cast<uint32_t>(cur.size());
    if (R == 0u) {
        for (const auto& r : recs) {
            const uint32_t c = cur[r.first]++;
            if (c >= cap) continue;
            bucket[r.first * cap + c] = r.second;
            log.push_back({r.first * cap + c, 1u});
        }
        return;
    }
    std::vector<uint32_t> ring(T * R, 0xDEADu), start = cur;
    auto flush_run = [&](uint32_t t, uint32_t last) {
        const uint32_t s0 = sort_ol::ring_run_start(start[t], last, R);
        for (uint32_t s = s0; s <= last; ++s) bucket[t * cap + s] = ring[t * R + (s & (R - 1u))];
        log.push_back({t * cap + s0, last + 1u - s0});
    };
    for (const auto& r : recs) {
        const uint32_t t = r.first, c = cur[t]++;
        if (c >= cap) continue;
        ring[t * R + (c & (R - 1u))] = r.second;
        if ((c & (R - 1u)) == R - 1u) flush_run(t, c);
    }
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t last;
        if (sort_ol::ring_drain(start[t], cur[t], cap, R, &last)) flush_run(t, last);
    }
}

void test_ring() {
    std::mt19937 rng(7);
    for (const uint32_t R : {2u, 4u, 8u, 16u}) {
        for (int trial = 0; trial < 40; ++trial) {
            const uint32_t T = 1u + rng() % 40u;
            const uint32_t cap = REC_PAGE_RECS * (1u + rng() % 3u);
            // Two movers per core, three cores: core c mover m owns cursors
            // [base, base + its count) of every tile, in canonical order.
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> rec(6);
            std::vector<std::vector<uint32_t>> first(6, std::vector<uint32_t>(T, 0));
            std::vector<uint32_t> fill(T, 0);
            uint32_t id = 1;
            for (uint32_t w = 0; w < 6; ++w) {
                const uint32_t n = rng() % 400u;
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t t = (rng() % 4u == 0u) ? rng() % T : (rng() % std::min(T, 3u));
                    rec[w].push_back({t, id++});
                }
                for (uint32_t t = 0; t < T; ++t) first[w][t] = fill[t];
                for (const auto& r : rec[w]) fill[r.first]++;
            }
            std::vector<uint32_t> ref(T * cap, 0u), got(T * cap, 0u);
            std::vector<Write> lref, lgot;
            for (uint32_t w = 0; w < 6; ++w) {
                emit(rec[w], first[w], cap, 0u, ref, lref);
                std::vector<Write> lw;
                emit(rec[w], first[w], cap, R, got, lw);
                const uint32_t T0 = T;
                for (const Write& x : lw) {
                    const uint32_t t = x.slot / cap, s = x.slot % cap;
                    const uint32_t end = (w + 1u < 6u) ? first[w + 1][t] : fill[t];
                    CHECK(t < T0 && s >= first[w][t] && s + x.n <= std::min(end, cap),
                          "R %u write [%u,+%u) of tile %u outside mover cursors [%u,%u)", R, s, x.n,
                          t, first[w][t], std::min(end, cap));
                    CHECK(s / REC_PAGE_RECS == (s + x.n - 1u) / REC_PAGE_RECS,
                          "R %u write crosses a record page", R);
                    CHECK(x.n <= R, "run longer than R");
                }
                lgot.insert(lgot.end(), lw.begin(), lw.end());
            }
            CHECK(ref == got, "R %u trial %d: bucket image differs", R, trial);
            uint32_t nref = 0, ngot = 0;
            for (const Write& x : lref) nref += x.n;
            for (const Write& x : lgot) ngot += x.n;
            CHECK(nref == ngot, "R %u: %u records written, want %u", R, ngot, nref);
        }
    }
}

// ── work items ────────────────────────────────────────────────────────────
void test_worklist() {
    // Bicycle-like frame: 39 x 26 tiles, a few huge, heavy tail.
    std::mt19937 rng(3);
    const uint32_t T = 1014, cores = 110, fit = 8192;
    std::vector<int64_t> counts(T);
    for (uint32_t t = 0; t < T; ++t) {
        const double x = std::exponential_distribution<double>(1.0 / 2600.0)(rng);
        counts[t] = static_cast<int64_t>(std::min(x, 16000.0));
    }
    counts[100] = 25700;
    counts[101] = 22000;
    counts[300] = 32768;
    counts[500] = 16385;
    const auto a = build_mat_worklist(counts, T, cores, fit, 2, kMatMover0Cap, true, true);
    // Items per tile: (sc, part) -> records; cost per the host model.
    std::vector<std::vector<uint32_t>> cover(T);
    uint64_t total = 0, max_item = 0;
    for (uint32_t slot = 0; slot < cores * 2u; ++slot) {
        for (uint32_t i = 0; i < a.per_core_count[slot]; ++i) {
            const uint32_t t = a.flat[2u * (a.per_core_offset[slot] + i)];
            const uint32_t w1 = a.flat[2u * (a.per_core_offset[slot] + i) + 1u];
            const uint32_t cnt = static_cast<uint32_t>(counts[t]);
            uint64_t cost = cnt;
            if (cnt > render_config::kOverflowL1Cap) {
                const uint32_t sc = w1 & 0xFFu, part = w1 >> 8;
                const uint32_t ls = std::min(fit, cnt - sc * fit);
                const uint32_t p0 = part * kOlMatPartRecs;
                CHECK(p0 < ls, "empty part");
                const uint32_t recs = std::min(kOlMatPartRecs, ls - p0);
                CHECK((slot & 1u) == 0u, "big item on BRISC");
                if (cover[t].empty()) cover[t].assign(cnt, 0u);
                for (uint32_t r = 0; r < recs; ++r) cover[t][sc * fit + p0 + r]++;
                cost = cnt / 4u + 2u * recs;
            }
            total += cost;
            max_item = std::max(max_item, cost);
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        if (static_cast<uint32_t>(counts[t]) <= render_config::kOverflowL1Cap) continue;
        bool exact = cover[t].size() == static_cast<size_t>(counts[t]);
        for (const uint32_t c : cover[t]) exact = exact && c == 1u;
        CHECK(exact, "tile %u: parts do not cover its records exactly once", t);
    }
    const uint64_t mean = total / (cores * 2u);
    std::printf("worklist: max_item %llu mean_slot %llu (%.2fx)\n", (unsigned long long)max_item,
                (unsigned long long)mean, double(max_item) / double(mean));
    CHECK(max_item * 2u <= mean * 3u, "max item %llu > 1.5x mean slot %llu",
          (unsigned long long)max_item, (unsigned long long)mean);
    // v1 items (no select) on the same frame: the 25.7k tile alone sets it.
    const auto v1 = build_mat_worklist(counts, T, cores, fit, 2, kMatMover0Cap, true, false);
    CHECK(v1.flat.size() < a.flat.size(), "select adds parts");
}


// Task #168 (OL_MAT_SHARED): one sort item per big tile (up to kOlSharedSlots
// tiles, the rest keep per-subchunk re-sort items), one gather item per
// subchunk, NCRISC slots only, and each mover runs sorts < plain < gathers.
void test_worklist_shared() {
    const uint32_t T = 300, cores = 110, fit = 8192;
    std::vector<int64_t> counts(T, 3000);
    for (uint32_t t = 0; t < 70; ++t) counts[t] = 16385 + t * 211;  // 70 big tiles
    counts[0] = 32768;
    const auto a = build_mat_worklist(counts, T, cores, fit, 2, kMatMover0Cap, true, false, true);
    std::vector<int> sort_slot(T, -1);
    std::vector<std::vector<uint32_t>> sc_seen(T);
    std::vector<uint32_t> slot_used(kOlSharedSlots, 0u);
    uint32_t legacy_big = 0;
    for (uint32_t m = 0; m < cores * 2u; ++m) {
        int last_rank = 0;
        for (uint32_t i = 0; i < a.per_core_count[m]; ++i) {
            const uint32_t t = a.flat[2u * (a.per_core_offset[m] + i)];
            const uint32_t w1 = a.flat[2u * (a.per_core_offset[m] + i) + 1u];
            const uint32_t kind = (w1 >> 16) & 3u, slot = w1 >> 18, sc = w1 & 0xFFu;
            const bool big = counts[t] > static_cast<int64_t>(render_config::kOverflowL1Cap);
            const int rank = kind == kOlKindSort ? 0 : (kind == kOlKindGather ? 2 : 1);
            CHECK(rank >= last_rank, "mover %u: item kind %u after rank %d", m, kind, last_rank);
            last_rank = rank;
            if (big) CHECK((m & 1u) == 0u, "big item on BRISC");
            if (kind == kOlKindSort) {
                CHECK(sort_slot[t] < 0, "tile %u sorted twice", t);
                CHECK(slot < kOlSharedSlots && slot_used[slot]++ == 0u, "slot %u reused", slot);
                sort_slot[t] = static_cast<int>(slot);
            } else if (big) {
                if (kind == kOlKindGather) CHECK(sort_slot[t] < 0 || sort_slot[t] == int(slot), "slot");
                else ++legacy_big;
                sc_seen[t].push_back(sc | (kind << 16) | (slot << 18));
            }
        }
    }
    uint32_t shared = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t cnt = static_cast<uint32_t>(counts[t]);
        if (cnt <= render_config::kOverflowL1Cap) continue;
        const uint32_t num_sc = (cnt + fit - 1u) / fit;
        std::vector<uint32_t> n(num_sc, 0u);
        for (const uint32_t w : sc_seen[t]) {
            CHECK((w & 0xFFu) < num_sc, "tile %u: sc out of range", t);
            if ((w & 0xFFu) < num_sc) n[w & 0xFFu]++;
            if (sort_slot[t] >= 0) CHECK(((w >> 16) & 3u) == kOlKindGather && (w >> 18) == uint32_t(sort_slot[t]),
                                         "tile %u: non-gather item next to its sort", t);
        }
        for (uint32_t s = 0; s < num_sc; ++s) CHECK(n[s] == 1u, "tile %u sc %u: %u items", t, s, n[s]);
        shared += sort_slot[t] >= 0;
    }
    CHECK(shared == kOlSharedSlots, "%u shared tiles, want %u", shared, kOlSharedSlots);
    uint32_t want_legacy = 0;
    for (uint32_t t = kOlSharedSlots; t < 70u; ++t) want_legacy += (uint32_t(counts[t]) + fit - 1u) / fit;
    CHECK(legacy_big == want_legacy, "%u legacy big items, want %u", legacy_big, want_legacy);
}

}  // namespace

int main() {
    test_select();
    test_ring();
    test_worklist();
    test_worklist_shared();
    if (g_fail != 0) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
