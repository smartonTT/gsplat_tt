// Host checks for task #372: tiles of 32769..32832 records at the p150's
// 513-page bucket stride (sort_onelaunch::bucket_tile_cap(kTileCap, 8) = 32832).
// sort_device.cpp's capacity check accepts them without the #284 grow, so the
// materialize takes its big path (N > sort_radix_tile::MAX_N, select_ranks_big)
// at cap 32832, not at kTileCapBig. Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_tile_window_t372.cpp
//
//  - bucket stride: 513 pages; a full tile (32832 records) read from the
//    materialize's page0 ends on its own last page, before the next tile's;
//    ring_drain clamps a mover's cursor at tile_cap;
//  - emit (layout model, 4 cores x 2 movers) of adjacent tiles of 32832, 32769,
//    32800 and 32833 records: every kept slot stays inside its own tile's
//    bucket, slots are unique, the capacity check (a copy of sort_device.cpp's
//    need = max(total, padded) > cap) accepts the window tiles and rejects
//    32833, and the dropped records of the over-cap tile never land in the
//    next bucket;
//  - materialize: build_mat_worklist items (v1 and OL_MAT_SELECT) of the
//    window tiles are NCRISC-only and cover each tile exactly once; each item
//    runs select_ranks_big in the kernel's L1 layout (keys in CB_BUCKET's first
//    N u32, candidates from u32 65536, ck2 / cv2 in CB_SLAB, kCand 32768) and
//    the gathered pair ids equal the CPU stable depth sort of the tile, on
//    uniform, clustered, few-distinct, all-equal and fp32-depth keys.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "render/host/sort_mover_split.h"
#include "render/host/sort_onelaunch_layout.h"
#include "render/kernels/dataflow/sort_onelaunch_algo.h"

namespace so = gsplat_tt::sort_onelaunch;
namespace ss = gsplat_tt::sort_split;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                       \
    do {                                                    \
        if (!(c)) {                                         \
            if (g_fail < 30) {                              \
                std::printf("FAIL %s:%d ", __FILE__, __LINE__); \
                std::printf(__VA_ARGS__);                   \
                std::printf("\n");                          \
            }                                               \
            ++g_fail;                                       \
        }                                                   \
    } while (0)

constexpr uint32_t kRecPage = 64;  // records per 2 KB bucket page (REC_PAGE_RECS)
constexpr uint32_t kFit = render_config::kBucketFit;
const uint32_t kCap = so::bucket_tile_cap(so::kTileCap, 8u, -1);

uint32_t key_of(int kind, std::mt19937& rng) {
    switch (kind) {
        case 0: return rng();
        case 1: return (rng() % 64u == 0u) ? rng() : 0x3F800000u + (rng() % 5000u);
        case 2: return rng() % 7u;
        case 3: return 12345u;
        default: {
            const float d = 0.5f + std::exponential_distribution<float>(0.3f)(rng);
            uint32_t b;
            std::memcpy(&b, &d, 4);
            return b;
        }
    }
}

void test_stride() {
    CHECK(kCap == 32832u, "p150 cap %u, want 32832 (513 pages)", kCap);
    CHECK(kCap > sort_radix_tile::MAX_N && kCap <= sort_ol::BIG_MAX_N, "cap %u outside big path", kCap);
    const uint32_t pages = kCap / kRecPage;
    for (uint32_t t = 0; t < 4; ++t) {
        const uint32_t page0 = t * pages;  // sort_subchunk_materialize.cpp page0
        const uint32_t last = page0 + (kCap - 1u) / kRecPage;
        CHECK(last < (t + 1u) * pages, "tile %u last page %u reaches tile %u", t, last, t + 1u);
        CHECK((page0 + 1u) * kRecPage <= (t + 1u) * kCap, "page0 of tile %u", t);
    }
    uint32_t last = 0;
    sort_ol::ring_drain(kCap - 10u, kCap + 50u, kCap, 8u, &last);
    CHECK(last == kCap - 1u, "ring_drain last %u past cap", last);
    CHECK(!sort_ol::ring_drain(kCap, kCap + 50u, kCap, 8u, &last), "ring_drain over cap writes");
}

struct Frame {
    std::vector<int32_t> tids, keep;
    std::vector<uint32_t> keys;  // depth key per pair
    uint32_t P = 0;
};

// Pairs of `n[t]` records per tile. pure_pages: every 16-pair page holds one
// tile (each core's count of a tile is a multiple of 16 but its last page),
// else the tiles interleave pair by pair and ~1 in 9 extra pairs is culled.
Frame make_frame(const std::vector<uint32_t>& n, bool pure_pages, int kind, std::mt19937& rng) {
    Frame f;
    std::vector<int32_t> order;
    if (pure_pages) {
        std::vector<std::vector<int32_t>> pages;
        for (uint32_t t = 0; t < n.size(); ++t) {
            for (uint32_t i = 0; i < n[t]; i += 16u) pages.emplace_back(std::min(16u, n[t] - i), int32_t(t));
        }
        std::shuffle(pages.begin(), pages.end(), rng);
        for (auto& p : pages) {
            p.resize(16u, -1);  // pad the page with culled pairs
            order.insert(order.end(), p.begin(), p.end());
        }
    } else {
        for (uint32_t t = 0; t < n.size(); ++t) order.insert(order.end(), n[t], int32_t(t));
        std::shuffle(order.begin(), order.end(), rng);
    }
    for (const int32_t t : order) {
        if (!pure_pages && rng() % 9u == 0u) {  // a culled pair of the tile
            f.tids.push_back(t);
            f.keep.push_back(0);
            f.keys.push_back(0u);
        }
        f.tids.push_back(t < 0 ? 0 : t);
        f.keep.push_back(t < 0 ? 0 : 1);
        f.keys.push_back(key_of(kind, rng));
    }
    f.P = static_cast<uint32_t>(f.tids.size());
    while (f.tids.size() % 16u != 0u) {
        f.tids.push_back(0);
        f.keep.push_back(0);
        f.keys.push_back(0u);
    }
    return f;
}

// The kernel's big-tile item: CB_BUCKET (512 KB) holds the keys from u32 0 and
// the candidates ck / cv from u32 2 * kCand; CB_SLAB (256 KB) ck2 / cv2; the
// sorted slots land in CB_BSORT. Returns false if select_ranks_big failed.
bool big_item(const std::vector<uint32_t>& tile_keys, uint32_t lo, uint32_t L, std::vector<uint32_t>& slots) {
    constexpr uint32_t kCand = sort_radix_tile::MAX_N;
    const uint32_t N = static_cast<uint32_t>(tile_keys.size());
    static std::vector<uint32_t> buck(std::max(kFit * 64u, render_config::kOverflowL1Cap * 32u) / 4u);
    static std::vector<uint32_t> slab(kFit * 32u / 4u);
    static std::vector<uint32_t> bsort(2u * render_config::kOverflowL1Cap + 256u);
    static sort_radix_tile::hist_t hist[sort_radix_tile::HIST_ENTRIES];
    CHECK(N <= 2u * kCand, "keys %u overlap the candidates", N);
    CHECK(3u * kCand + kCand <= buck.size() && 2u * kCand <= slab.size() && L <= bsort.size(),
          "kernel buffers too small");
    std::copy(tile_keys.begin(), tile_keys.end(), buck.begin());
    uint32_t* keys = buck.data();
    uint32_t* ck = keys + 2u * kCand;
    uint32_t* ck2 = slab.data();
    const bool ok = sort_ol::select_ranks_big(keys, N, lo, lo + L, ck, ck + kCand, ck2, ck2 + kCand,
                                              kCand, bsort.data(), hist);
    CHECK(std::equal(tile_keys.begin(), tile_keys.end(), buck.begin()), "keys overwritten (N %u)", N);
    slots.assign(bsort.begin(), bsort.begin() + L);
    return ok;
}

void run_frame(const std::vector<uint32_t>& n, bool pure_pages, int kind, uint32_t seed) {
    std::mt19937 rng(seed);
    const Frame f = make_frame(n, pure_pages, kind, rng);
    const uint32_t T = static_cast<uint32_t>(n.size());
    const uint32_t stride = (T + 15u) / 16u * 16u;
    const uint32_t cores = 4, pages = (f.P + 15u) / 16u;
    std::vector<so::CoreSplit> split(cores);
    for (uint32_t c = 0; c < cores; ++c) {
        split[c].lo = pages * c / cores;
        split[c].hi = pages * (c + 1u) / cores;
        split[c].mid = (split[c].lo + split[c].hi) / 2u;
    }
    const so::Counts cnt = so::count_pass(f.tids, f.keep, f.P, split, stride);
    const so::Prefix pfx = so::device_prefix(cnt.h, cores, stride);
    const std::vector<uint32_t> slot = so::emit_slots(f.tids, f.keep, f.P, split, cnt, pfx, stride, kCap);

    // Bucket image and slot bounds.
    std::vector<uint32_t> bucket(static_cast<size_t>(T) * kCap, so::kDropped);
    std::vector<uint32_t> dropped(T, 0u);
    for (uint32_t i = 0; i < f.P; ++i) {
        if (f.keep[i] == 0) continue;
        const uint32_t t = static_cast<uint32_t>(f.tids[i]);
        if (slot[i] == so::kDropped) {
            dropped[t]++;
            continue;
        }
        CHECK(slot[i] / kCap == t, "pair %u of tile %u in bucket %u", i, t, slot[i] / kCap);
        if (slot[i] / kCap != t) continue;
        CHECK(bucket[slot[i]] == so::kDropped, "slot %u written twice", slot[i]);
        bucket[slot[i]] = i;
    }
    std::vector<int64_t> counts(T, 0);
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t tot = pfx.totals[t], pad = pfx.totals[stride + t];
        CHECK(tot == n[t], "tile %u total %u, want %u", t, tot, n[t]);
        // sort_device.cpp: need = max(tot[t], tot[stride + t]); fail if need > check_cap.
        const bool accepted = std::max(tot, pad) <= kCap;
        if (tot > kCap) CHECK(!accepted, "tile %u of %u records accepted at cap %u", t, tot, kCap);
        CHECK(dropped[t] == (tot > kCap ? tot - kCap : 0u), "tile %u dropped %u", t, dropped[t]);
        if (pure_pages && tot <= kCap) {
            CHECK(accepted, "tile %u: %u records (padded %u) rejected at cap %u", t, tot, pad, kCap);
        }
        if (accepted) counts[t] = tot;
    }

    // Materialize the accepted tiles, per the worklist, against the CPU sort.
    for (const bool sel : {false, true}) {
        const ss::MatWorkAssignment a =
            ss::build_mat_worklist(counts, T, 3u, kFit, 2u, ss::kMatMover0Cap, true, sel);
        std::vector<std::vector<uint32_t>> out(T), cover(T);
        std::vector<std::vector<uint32_t>> tile_keys(T);
        for (uint32_t t = 0; t < T; ++t) {
            if (counts[t] == 0) continue;
            out[t].assign(counts[t], so::kDropped);
            cover[t].assign(counts[t], 0u);
            const uint32_t page0 = t * (kCap / kRecPage);
            for (uint32_t r = 0; r < counts[t]; ++r) {
                const uint32_t id = bucket[page0 * kRecPage + r];
                CHECK(id != so::kDropped, "tile %u bucket slot %u empty", t, r);
                tile_keys[t].push_back(id != so::kDropped ? f.keys[id] : 0u);
            }
        }
        for (uint32_t s = 0; s < a.per_core_offset.size(); ++s) {
            for (uint32_t j = 0; j < a.per_core_count[s]; ++j) {
                const uint32_t t = a.flat[2u * (a.per_core_offset[s] + j)];
                const uint32_t w1 = a.flat[2u * (a.per_core_offset[s] + j) + 1u];
                const uint32_t N = static_cast<uint32_t>(counts[t]);
                CHECK((s & 1u) == 0u, "tile %u item on BRISC slot %u", t, s);
                const uint32_t sc = w1 & 0xFFu, part = w1 >> 8;
                const uint32_t sc_off = sc * kFit, L_sub = std::min(kFit, N - sc_off);
                uint32_t po = 0, L = L_sub;
                if (sel) {
                    po = part * ss::kOlMatPartRecs;
                    CHECK(po < L_sub, "empty part");
                    L = std::min(ss::kOlMatPartRecs, L_sub - po);
                } else {
                    CHECK(part == 0u, "v1 part %u", part);
                }
                std::vector<uint32_t> k;
                const bool ok = big_item(tile_keys[t], sc_off + po, L, k);
                CHECK(ok, "kind %d tile %u N %u [%u,+%u): select_ranks_big failed", kind, t, N, sc_off + po, L);
                const uint32_t page0 = t * (kCap / kRecPage);
                for (uint32_t i = 0; i < L; ++i) {
                    CHECK(k[i] < N, "slot %u past N %u", k[i], N);
                    if (k[i] >= N) continue;
                    out[t][sc_off + po + i] = bucket[page0 * kRecPage + k[i]];  // the gather
                    cover[t][sc_off + po + i]++;
                }
            }
        }
        for (uint32_t t = 0; t < T; ++t) {
            if (counts[t] == 0) continue;
            // CPU sort: the tile's kept pairs in pair order, stable by depth key.
            std::vector<uint32_t> ref;
            for (uint32_t i = 0; i < f.P; ++i) {
                if (f.keep[i] != 0 && static_cast<uint32_t>(f.tids[i]) == t) ref.push_back(i);
            }
            std::stable_sort(ref.begin(), ref.end(), [&](uint32_t x, uint32_t y) { return f.keys[x] < f.keys[y]; });
            uint32_t bad = 0, once = 0;
            for (uint32_t r = 0; r < counts[t]; ++r) {
                bad += out[t][r] != ref[r] ? 1u : 0u;
                once += cover[t][r] == 1u ? 1u : 0u;
            }
            CHECK(once == counts[t], "sel %d tile %u: %u of %lld ranks covered once", sel, t, once,
                  (long long)counts[t]);
            CHECK(bad == 0u, "sel %d kind %d tile %u N %lld: %u ranks differ from the CPU sort", sel, kind,
                  t, (long long)counts[t], bad);
        }
    }
    std::printf("frame pure=%d kind=%d tiles", pure_pages, kind);
    for (uint32_t t = 0; t < T; ++t) {
        std::printf(" %u(pad %u,%s)", n[t], pfx.totals[stride + t], counts[t] ? "kept" : "rejected");
    }
    std::printf("\n");
}

}  // namespace

int main() {
    test_stride();
    for (int kind = 0; kind < 5; ++kind) {
        // Adjacent full tiles: the window's edges, the middle, and one past cap
        // (dropped and rejected) before a full tile, so an overrun would show.
        run_frame({32832u, 32769u, 32800u, 32833u, 32832u}, true, kind, 372u + kind);
        run_frame({32769u, 32831u, 32832u}, false, kind, 900u + kind);
    }
    if (g_fail != 0) {
        std::printf("%d check(s) failed\n", g_fail);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
