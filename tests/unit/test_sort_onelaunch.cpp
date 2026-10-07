// Checks the task #106 one-launch device sort layout (GSPLAT_TT_SORT_ONELAUNCH;
// render/host/sort_onelaunch_layout.h models render/kernels/dataflow/
// sort_bin_onelaunch.cpp; the worklist is render/host/sort_mover_split.h).
// Standalone:
//
//   c++ -O2 -std=c++17 -Irender/host tests/unit/test_sort_onelaunch.cpp
//     -o /tmp/t_ol && /tmp/t_ol
//
// On random heavy-tailed pairs over 110 dual-mover cores:
//  - the device prefix (page-owner column walk) equals a plain per-tile
//    exclusive prefix over cores, and its padded totals equal the legacy
//    tile_pad (so the blend LPT is unchanged);
//  - every kept pair gets its own slot, and each tile's bucket holds its pairs
//    in the legacy prefix layout's order, so the materialize's stable depth
//    radix gives the legacy output (checked on duplicate-heavy keys);
//  - a tile past tile_cap drops exactly the excess and its total shows it;
//  - build_mat_worklist(onelaunch) gives each over-cap tile one item per
//    subchunk (no gather parts), all big items NCRISC-only, and BRISC never
//    gets a tile above kMatMover0Cap;
//  - check_prefix flags a corrupted base or total;
//  - unpack_rows_view (task #355) inverts the 4 KB-page view of the 64 B-page
//    interleaved K2 rows for 7 and 8 DRAM banks.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <utility>
#include <vector>

#include "sort_mover_split.h"
#include "sort_onelaunch_layout.h"

using namespace gsplat_tt::sort_onelaunch;
using gsplat_tt::sort_split::build_mat_worklist;
using gsplat_tt::sort_split::kMatMover0Cap;
using gsplat_tt::sort_split::MatWorkAssignment;

namespace {

constexpr uint32_t kCores = 110;

struct Frame {
    uint32_t P = 0, num_tiles = 0, stride = 0;
    std::vector<int32_t> tids, keep;
    std::vector<uint32_t> key;  // depth key (few distinct values: stability matters)
    std::vector<CoreSplit> split;
};

Frame make_frame(std::mt19937& rng, uint32_t P, uint32_t num_tiles, uint32_t hot_tile,
                 double hot_frac) {
    Frame f;
    f.P = P;
    f.num_tiles = num_tiles;
    f.stride = (num_tiles + 15u) / 16u * 16u;
    const uint32_t pages = (P + 15u) / 16u;
    f.tids.assign(pages * 16u, 0);
    f.keep.assign(pages * 16u, 0);
    f.key.assign(pages * 16u, 0);
    std::lognormal_distribution<double> ln(5.5, 1.0);
    std::bernoulli_distribution hot(hot_frac), kept(0.8);
    std::uniform_int_distribution<uint32_t> k(0, 63);
    for (uint32_t i = 0; i < P; ++i) {
        const uint32_t t = hot(rng) ? hot_tile
                                    : static_cast<uint32_t>(ln(rng)) % num_tiles;
        f.tids[i] = static_cast<int32_t>(t);
        f.keep[i] = kept(rng) ? 1 : 0;
        f.key[i] = k(rng);
    }
    // Garbage past P (the kernel must ignore it).
    for (uint32_t i = P; i < pages * 16u; ++i) {
        f.tids[i] = 0;
        f.keep[i] = 1;
    }
    // Contiguous page slices in core order (the gather worklist), split at a
    // random permille like sort_emit_split_permille().
    std::uniform_int_distribution<uint32_t> pm(0, 1000);
    f.split.resize(kCores);
    for (uint32_t c = 0; c < kCores; ++c) {
        const uint32_t lo = static_cast<uint32_t>(static_cast<uint64_t>(pages) * c / kCores);
        const uint32_t hi = static_cast<uint32_t>(static_cast<uint64_t>(pages) * (c + 1) / kCores);
        f.split[c] = {lo, lo + (hi - lo) * pm(rng) / 1000u, hi};
    }
    return f;
}

// Legacy prefix layout (sort_device.cpp host_bin_layout_into + sort_bin.cpp
// emit): core c's block of tile t starts at 16 * (tile page start + sum of
// ceil16 over earlier cores); a core's k-th kept pair of t in page order sits
// at offset k (mover 1 starts at h0). Returns each tile's pairs in position
// order, and tile_pad.
void legacy_layout(const Frame& f, std::vector<std::vector<uint32_t>>& per_tile,
                   std::vector<uint32_t>& tile_pad) {
    const Counts cnt = count_pass(f.tids, f.keep, f.P, f.split, f.stride);
    std::vector<uint32_t> tpages(f.num_tiles, 0);
    for (uint32_t c = 0; c < kCores; ++c)
        for (uint32_t t = 0; t < f.num_tiles; ++t)
            tpages[t] += (cnt.h[c * f.stride + t] + 15u) / 16u;
    std::vector<uint32_t> blk(static_cast<std::size_t>(kCores) * f.stride, 0);
    tile_pad.assign(f.num_tiles, 0);
    uint32_t apage = 0;
    for (uint32_t t = 0; t < f.num_tiles; ++t) {
        uint32_t pos = apage * 16u;
        for (uint32_t c = 0; c < kCores; ++c) {
            blk[c * f.stride + t] = pos;
            pos += (cnt.h[c * f.stride + t] + 15u) / 16u * 16u;
        }
        apage += tpages[t];
        tile_pad[t] = tpages[t] * 16u;
    }
    std::map<uint32_t, uint32_t> pos_to_pair;
    for (uint32_t c = 0; c < kCores; ++c) {
        std::vector<uint32_t> k(f.stride, 0);
        for (uint32_t pg = f.split[c].lo; pg < f.split[c].hi; ++pg) {
            for (uint32_t j = 0; j < 16u; ++j) {
                const uint32_t i = pg * 16u + j;
                if (i >= f.P) break;
                if (f.keep[i] == 0) continue;
                const uint32_t t = static_cast<uint32_t>(f.tids[i]);
                pos_to_pair[blk[c * f.stride + t] + k[t]++] = i;
            }
        }
    }
    per_tile.assign(f.num_tiles, {});
    for (const auto& [pos, i] : pos_to_pair) per_tile[f.tids[i]].push_back(i);
}

std::vector<uint32_t> stable_by_key(const Frame& f, std::vector<uint32_t> ids) {
    std::stable_sort(ids.begin(), ids.end(),
                     [&](uint32_t a, uint32_t b) { return f.key[a] < f.key[b]; });
    return ids;
}

int check_frame(const Frame& f, uint32_t tile_cap, const char* tag) {
    int bad = 0;
    const Counts cnt = count_pass(f.tids, f.keep, f.P, f.split, f.stride);
    const Prefix pfx = device_prefix(cnt.h, kCores, f.stride);

    // Prefix == per-tile exclusive prefix over cores; padded totals == tile_pad.
    std::vector<std::vector<uint32_t>> legacy;
    std::vector<uint32_t> tile_pad;
    legacy_layout(f, legacy, tile_pad);
    for (uint32_t t = 0; t < f.num_tiles; ++t) {
        uint32_t acc = 0;
        for (uint32_t c = 0; c < kCores; ++c) {
            if (pfx.base[c * f.stride + t] != acc) {
                if (bad++ < 5) std::printf("%s: base[%u][%u] %u != %u\n", tag, c, t,
                                           pfx.base[c * f.stride + t], acc);
            }
            acc += cnt.h[c * f.stride + t];
        }
        if (pfx.totals[t] != acc || pfx.totals[f.stride + t] != tile_pad[t]) {
            if (bad++ < 5) std::printf("%s: tile %u totals %u/%u != %u/%u\n", tag, t,
                                       pfx.totals[t], pfx.totals[f.stride + t], acc, tile_pad[t]);
        }
    }
    if (check_prefix(cnt.h, pfx.base, pfx.totals, kCores, f.stride, f.num_tiles) != 0) {
        std::printf("%s: check_prefix flags a correct prefix\n", tag);
        bad++;
    }

    // Slots: unique, inside the tile's bucket, in legacy order, excess dropped.
    const std::vector<uint32_t> slot =
        emit_slots(f.tids, f.keep, f.P, f.split, cnt, pfx, f.stride, tile_cap);
    std::map<uint32_t, uint32_t> by_slot;
    uint32_t dropped = 0, expect_dropped = 0;
    for (uint32_t i = 0; i < f.P; ++i) {
        if (f.keep[i] == 0) {
            if (slot[i] != kDropped) bad++;
            continue;
        }
        if (slot[i] == kDropped) {
            dropped++;
            continue;
        }
        if (slot[i] / tile_cap != static_cast<uint32_t>(f.tids[i]) || !by_slot.emplace(slot[i], i).second) {
            if (bad++ < 5) std::printf("%s: pair %u slot %u bad or shared\n", tag, i, slot[i]);
        }
    }
    for (uint32_t i = f.P; i < slot.size(); ++i) {
        if (slot[i] != kDropped) {
            if (bad++ < 5) std::printf("%s: pair %u past P got slot %u\n", tag, i, slot[i]);
        }
    }
    uint32_t overcap = 0;
    for (uint32_t t = 0; t < f.num_tiles; ++t) {
        const uint32_t n = pfx.totals[t];
        if (n > tile_cap) {
            expect_dropped += n - tile_cap;
            overcap++;
            continue;  // the host fails such a frame; nothing to compare
        }
        std::vector<uint32_t> bucket;  // slots t*cap .. t*cap+n-1, dense
        for (uint32_t k = 0; k < n; ++k) {
            const auto it = by_slot.find(t * tile_cap + k);
            if (it == by_slot.end()) {
                if (bad++ < 5) std::printf("%s: tile %u slot %u empty\n", tag, t, k);
                break;
            }
            bucket.push_back(it->second);
        }
        if (bucket != legacy[t] || stable_by_key(f, bucket) != stable_by_key(f, legacy[t])) {
            if (bad++ < 5) std::printf("%s: tile %u bucket order != legacy\n", tag, t);
        }
    }
    if (dropped != expect_dropped) {
        std::printf("%s: dropped %u records, expected %u\n", tag, dropped, expect_dropped);
        bad++;
    }
    std::printf("%s: P=%u tiles=%u kept=%zu dropped=%u overcap_tiles=%u bad=%d\n", tag, f.P,
                f.num_tiles, by_slot.size(), dropped, overcap, bad);
    return bad;
}

int check_check_prefix(std::mt19937& rng) {
    const Frame f = make_frame(rng, 20000, 300, 7, 0.1);
    const Counts cnt = count_pass(f.tids, f.keep, f.P, f.split, f.stride);
    Prefix pfx = device_prefix(cnt.h, kCores, f.stride);
    int bad = 0;
    pfx.base[5 * f.stride + 7] += 1;
    if (check_prefix(cnt.h, pfx.base, pfx.totals, kCores, f.stride, f.num_tiles) != 1) bad++;
    pfx.base[5 * f.stride + 7] -= 1;
    pfx.totals[f.stride + 9] += 16;
    if (check_prefix(cnt.h, pfx.base, pfx.totals, kCores, f.stride, f.num_tiles) != 1) bad++;
    std::printf("check_prefix: corruptions flagged bad=%d\n", bad);
    return bad;
}

std::vector<int64_t> random_counts(std::mt19937& rng, uint32_t tiles) {
    std::lognormal_distribution<double> ln(7.0, 1.3);
    std::bernoulli_distribution empty(0.05);
    std::vector<int64_t> c(tiles, 0);
    for (auto& v : c) {
        if (empty(rng)) continue;
        v = std::min<int64_t>(static_cast<int64_t>(ln(rng)), kTileCap);
    }
    c[3] = kTileCap;  // the largest tile the bucket holds
    c[4] = render_config::kOverflowL1Cap + 1;
    return c;
}

int check_worklist(std::mt19937& rng) {
    constexpr uint32_t kTiles = 1024;
    const uint32_t fit = render_config::kBucketFit;
    const uint32_t ov_cap = render_config::kOverflowL1Cap;
    int bad = 0;
    for (int round = 0; round < 20; ++round) {
        const std::vector<int64_t> counts = random_counts(rng, kTiles);
        for (uint32_t movers = 1; movers <= 2; ++movers) {
            const MatWorkAssignment a = build_mat_worklist(counts, kTiles, kCores, fit, movers,
                                                           kMatMover0Cap, /*onelaunch=*/true);
            std::map<std::pair<uint32_t, uint32_t>, int> seen;
            for (uint32_t s = 0; s < kCores * movers; ++s) {
                for (uint32_t k = 0; k < a.per_core_count[s]; ++k) {
                    const uint32_t i = a.per_core_offset[s] + k;
                    const uint32_t t = a.flat[2 * i], w = a.flat[2 * i + 1];
                    const uint32_t n = static_cast<uint32_t>(counts[t]);
                    seen[{t, w}]++;
                    const bool brisc = movers == 2 && (s & 1u);
                    // BRISC's whole path holds kMatMover0Cap records; the
                    // big path (n > ov_cap) runs on NCRISC only.
                    if (brisc && n > kMatMover0Cap) {
                        if (bad++ < 5) std::printf("worklist: tile %u (%u recs) on BRISC\n", t, n);
                    }
                    if ((w >> 8) != 0u) {
                        if (bad++ < 5) std::printf("worklist: tile %u got gather part %u\n", t, w >> 8);
                    }
                }
            }
            for (uint32_t t = 0; t < kTiles; ++t) {
                const uint32_t n = static_cast<uint32_t>(counts[t]);
                const uint32_t items = n == 0 ? 0 : (n <= ov_cap ? 1 : (n + fit - 1) / fit);
                for (uint32_t sc = 0; sc < items; ++sc) {
                    if (seen[{t, sc}] != 1) {
                        if (bad++ < 5) std::printf("worklist: tile %u sc %u seen %d times\n", t, sc,
                                                   seen[{t, sc}]);
                    }
                }
                if (n > 0 && seen.count({t, items}) != 0) {
                    if (bad++ < 5) std::printf("worklist: tile %u has an extra item\n", t);
                }
            }
            if (a.max_items_per_core > 1024u) bad++;
        }
    }
    // Big-path buffer arithmetic (sort_subchunk_materialize.cpp, NCRISC):
    // keys in CB_BSORT (132096 B), v/k2/v2 in CB_BUCKET (512 KB).
    static_assert(kTileCap * 4u <= 132096u, "big-path keys overflow CB_BSORT");
    static_assert(3u * kTileCap * 4u <= 512u * 1024u, "big-path arrays overflow CB_BUCKET");
    std::printf("worklist: 20 rounds x {1,2} movers bad=%d\n", bad);
    return bad;
}

// Task #355: model the interleaved DRAM placement of both page sizes and
// check unpack_rows_view restores the 64 B page order.
int check_rows_view(std::mt19937& rng) {
    int bad = 0;
    for (uint32_t banks : {7u, 8u, 12u}) {
        for (uint32_t pages0 : {14080u, 1000u, 1u}) {
            const uint32_t m = kRowsViewPerPage;
            const uint32_t pages = rows_view_pages(pages0, banks, m);
            if (pages % (banks * m) != 0u || pages < pages0) bad++;
            std::vector<uint32_t> lin(static_cast<std::size_t>(pages) * kElemsPerPage);
            for (auto& v : lin) v = rng();
            // Write the 64 B pages into per-bank memory, read it as m * 64 B pages.
            const uint32_t per_bank = pages / banks * kElemsPerPage;
            std::vector<uint32_t> mem(static_cast<std::size_t>(banks) * per_bank);
            for (uint32_t i = 0; i < pages; ++i)
                std::copy(lin.begin() + i * kElemsPerPage, lin.begin() + (i + 1u) * kElemsPerPage,
                          mem.begin() + (i % banks) * per_bank + (i / banks) * kElemsPerPage);
            std::vector<uint32_t> view(lin.size());
            const uint32_t vw = m * kElemsPerPage;
            for (uint32_t j = 0; j < pages / m; ++j)
                std::copy(mem.begin() + (j % banks) * per_bank + (j / banks) * vw,
                          mem.begin() + (j % banks) * per_bank + (j / banks + 1u) * vw,
                          view.begin() + j * vw);
            std::vector<uint32_t> out(lin.size(), 0u);
            unpack_rows_view(view.data(), pages, banks, m, out.data());
            if (out != lin) {
                if (bad++ < 5) std::printf("rows view: banks %u pages %u mismatch\n", banks, pages0);
            }
        }
    }
    std::printf("rows view: bad=%d\n", bad);
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(106);
    int bad = 0;
    // Bicycle-like: ~1.2 M pairs over 1000 tiles (stride padding exercised),
    // one hot tile near the bucket cap.
    bad += check_frame(make_frame(rng, 1200000, 1000, 17, 0.02), kTileCap, "bicycle-like");
    bad += check_frame(make_frame(rng, 50003, 1024, 3, 0.05), kTileCap, "small");
    // More cores than row pages (most cores own no prefix page).
    bad += check_frame(make_frame(rng, 9000, 40, 1, 0.3), kTileCap, "few-tiles");
    // Tiny cap: drops on several tiles.
    bad += check_frame(make_frame(rng, 60000, 200, 2, 0.2), 256, "overcap");
    bad += check_check_prefix(rng);
    bad += check_worklist(rng);
    bad += check_rows_view(rng);
    std::printf(bad == 0 ? "PASS\n" : "FAIL (%d)\n", bad);
    return bad == 0 ? 0 : 1;
}
