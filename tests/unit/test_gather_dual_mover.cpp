// Model check: the gather's dual-data-mover split (gather_visible_device.cpp
// launch_pass + gather_scan_bases.cpp + gather_visible_scatter.cpp) produces
// the same compact output as the single-mover pass, and every output element /
// padding slot is written by exactly one (core, mover) slot. Standalone:
//
//   c++ -O2 -std=c++17 tests/unit/test_gather_dual_mover.cpp -o /tmp/t_gdm && /tmp/t_gdm
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr uint32_t TILE = 1024, PAGE = 16;

struct Out {
    std::vector<uint32_t> val;     // source index + 1 per compact slot, 0 = pad
    std::vector<uint32_t> writes;  // writers per compact slot
};

// One pass of the host split + device scan + kernel writes with `movers`
// slots per core and mover 0 taking `permille`/1000 of each core's tiles.
Out run(const std::vector<uint8_t>& vis, uint32_t N, uint32_t cores,
        uint32_t movers, uint32_t permille) {
    const uint32_t tiles = (N + TILE - 1) / TILE;
    const uint32_t slots = cores * movers;
    std::vector<uint32_t> first(slots), cnt(slots), count(slots, 0);
    for (uint32_t c = 0; c < cores; c++) {
        const uint32_t n_all = c < tiles ? (tiles - 1 - c) / cores + 1 : 0;  // split_strided
        const uint32_t h0 = movers == 1 ? n_all : (n_all * permille + 500) / 1000;
        for (uint32_t m = 0; m < movers; m++) {
            const uint32_t s = c * movers + m;
            first[s] = c + (m == 0 ? 0 : h0 * cores);
            cnt[s] = m == 0 ? h0 : n_all - h0;
        }
    }
    auto for_each_visible = [&](uint32_t s, auto&& fn) {
        for (uint32_t k = 0, t = first[s]; k < cnt[s]; k++, t += cores)
            for (uint32_t il = 0; il < TILE; il++) {
                const uint32_t i = t * TILE + il;
                if (i >= N) break;
                if (vis[i]) fn(i);
            }
    };
    // count pass + scan (gather_scan_bases.cpp)
    uint32_t M = 0, last_nz = 0;
    bool any = false;
    for (uint32_t s = 0; s < slots; s++) {
        for_each_visible(s, [&](uint32_t) { count[s]++; });
        if (count[s]) { last_nz = s; any = true; }
        M += count[s];
    }
    const uint32_t cap = ((M + PAGE - 1) / PAGE + 1) * PAGE;
    Out o{std::vector<uint32_t>(cap, 0xFFFFFFFFu), std::vector<uint32_t>(cap, 0)};
    uint32_t acc = 0;
    for (uint32_t s = 0; s < slots; s++) {
        const bool is_last = any && s == last_nz;
        uint32_t g = acc;
        for_each_visible(s, [&](uint32_t i) { o.val[g] = i + 1; o.writes[g]++; g++; });
        if (is_last && g % PAGE)  // tail zero-pad of the last page
            for (uint32_t p = g; p % PAGE; p++) { o.val[p] = 0; o.writes[p]++; }
        acc += count[s];
    }
    return o;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    uint64_t cases = 0, bad = 0;
    for (int trial = 0; trial < 300; trial++) {
        const uint32_t cores = 1 + rng() % 12;
        const uint32_t N = 1 + rng() % (TILE * (1 + rng() % 40));
        const double p = (rng() % 5 == 0) ? 0.0 : (rng() % 1000) / 1000.0;
        std::vector<uint8_t> vis(N);
        for (auto& v : vis) v = (rng() % 1000) < p * 1000;
        const Out ref = run(vis, N, cores, 1, 1000);
        for (uint32_t pm : {0u, 1u, 250u, 450u, 500u, 550u, 999u, 1000u}) {
            cases++;
            const Out got = run(vis, N, cores, 2, pm);
            bool ok = got.val == ref.val;
            for (std::size_t g = 0; g < got.writes.size(); g++)
                if (got.writes[g] > 1 || (got.val[g] != 0xFFFFFFFFu && got.writes[g] != 1))
                    ok = false;
            if (!ok) bad++;
        }
    }
    std::printf("gather dual-mover: %llu cases, %llu mismatches\n",
                (unsigned long long)cases, (unsigned long long)bad);
    return bad == 0 ? 0 : 1;
}
