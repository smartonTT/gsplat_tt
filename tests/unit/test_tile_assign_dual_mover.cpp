// Model check for the dual-data-mover split of tile_assign K1 / K2
// (render/kernels/dataflow/tile_assign_bbox.cpp, tile_assign_scatter.cpp; host
// side in render/host/tile_assign_device.cpp). Standalone:
//
//   c++ -O2 -std=c++17 tests/unit/test_tile_assign_dual_mover.cpp -o /tmp/t_ta && /tmp/t_ta
//
// Replays each kernel's page loop for one core's page range on random inputs:
// once on a single mover over [lo, hi), and once split at every page boundary
// mid (BRISC = [lo, mid), NCRISC = [mid, hi)). Both movers write into the same
// output array; the split must leave exactly the single-mover bytes. A negative
// control runs the scatter's mover 1 without its own binary search (resuming a
// stale cursor from gaussian 0) and must mismatch. Returns non-zero on failure.
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int EPP = 16;  // elements per 64 B page

struct Scene {
    int M = 0, tiles_x = 0, tiles_y = 0, P = 0;
    float inv_tsf = 1.0f / 32.0f;
    std::vector<float> px, py, rx, ry;
    std::vector<int> offs;  // M + 1 (+ page padding, = P)
};

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Box { int minx, miny, w, h; };

Box aabb(const Scene& s, int g) {
    const int min_x = clampi(static_cast<int>((s.px[g] - s.rx[g]) * s.inv_tsf), 0, s.tiles_x - 1);
    const int max_x = clampi(static_cast<int>((s.px[g] + s.rx[g]) * s.inv_tsf), 0, s.tiles_x - 1);
    const int min_y = clampi(static_cast<int>((s.py[g] - s.ry[g]) * s.inv_tsf), 0, s.tiles_y - 1);
    const int max_y = clampi(static_cast<int>((s.py[g] + s.ry[g]) * s.inv_tsf), 0, s.tiles_y - 1);
    return {min_x, min_y, max_x - min_x + 1, max_y - min_y + 1};
}

// tile_assign_bbox.cpp page loop: tpg for pages [start, start + count).
void run_bbox(const Scene& s, uint32_t start, uint32_t count, std::vector<int>& tpg) {
    for (uint32_t pg = start; pg < start + count; pg++)
        for (int i = 0; i < EPP; i++) {
            const int g = static_cast<int>(pg) * EPP + i;
            if (g >= s.M) { tpg[g] = 0; continue; }
            const Box b = aabb(s, g);
            tpg[g] = b.w * b.h;
        }
}

// tile_assign_scatter.cpp page loop: (gid, tid) for pair pages [start, start + count).
// stale_cursor = negative control (skip the per-mover binary search).
void run_scatter(const Scene& s, uint32_t start, uint32_t count, std::vector<int>& gid,
                 std::vector<int>& tid, bool stale_cursor = false) {
    if (count == 0) return;
    const int p_start = static_cast<int>(start) * EPP;
    const int p_end = p_start + static_cast<int>(count) * EPP;
    if (p_start >= s.P) return;
    int lo = 0;
    if (!stale_cursor) {
        int hi = s.M - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (s.offs[mid] <= p_start) lo = mid; else hi = mid - 1;
        }
    }
    int g = lo;
    Box b = aabb(s, g);
    for (int p = p_start; p < p_end; p++) {
        int og = 0, ot = 0;
        if (p < s.P) {
            while (p >= s.offs[g + 1]) b = aabb(s, ++g);
            const int local = p - s.offs[g];
            const int dy = local / b.w;
            const int dx = local - dy * b.w;
            og = g;
            ot = (b.miny + dy) * s.tiles_x + (b.minx + dx);
        }
        gid[p] = og;
        tid[p] = ot;
    }
}

Scene make_scene(std::mt19937& rng, int M) {
    Scene s;
    s.M = M;
    s.tiles_x = 32;
    s.tiles_y = 32;
    std::uniform_real_distribution<float> pos(-50.f, 1100.f), rad(0.f, 120.f);
    for (int g = 0; g < M; g++) {
        s.px.push_back(pos(rng));
        s.py.push_back(pos(rng));
        s.rx.push_back(rad(rng));
        s.ry.push_back(rad(rng));
    }
    s.offs.assign(M + 1, 0);
    for (int g = 0; g < M; g++) {
        const Box b = aabb(s, g);
        s.offs[g + 1] = s.offs[g] + b.w * b.h;
    }
    s.P = s.offs[M];
    return s;
}

}  // namespace

int main() {
    std::mt19937 rng(34);
    long cases = 0, fails = 0, control_hits = 0, control_cases = 0;
    for (int trial = 0; trial < 6; trial++) {
        const int M = 37 + trial * 29;  // not page-aligned: exercises the g >= M tail
        const Scene s = make_scene(rng, M);

        // K1: pages over the padded M range (+2 pure-padding pages).
        const uint32_t m_pages = static_cast<uint32_t>((M + EPP - 1) / EPP) + 2;
        std::vector<int> ref1(m_pages * EPP, -1);
        run_bbox(s, 0, m_pages, ref1);
        for (uint32_t mid = 0; mid <= m_pages; mid++) {
            std::vector<int> out(m_pages * EPP, -1);
            run_bbox(s, 0, mid, out);
            run_bbox(s, mid, m_pages - mid, out);
            cases++;
            if (out != ref1) fails++;
        }

        // K2: a core range inside the pair pages, plus a range straddling P and
        // wholly-past-P pages (host_free over-provisions the split to a ceiling).
        const uint32_t p_pages = static_cast<uint32_t>((s.P + EPP - 1) / EPP) + 3;
        for (uint32_t lo : {0u, p_pages / 3}) {
            const uint32_t hi = p_pages;
            std::vector<int> rg(p_pages * EPP, -1), rt(p_pages * EPP, -1);
            run_scatter(s, lo, hi - lo, rg, rt);
            for (uint32_t mid = lo; mid <= hi; mid++) {
                std::vector<int> og(p_pages * EPP, -1), ot(p_pages * EPP, -1);
                run_scatter(s, lo, mid - lo, og, ot);
                run_scatter(s, mid, hi - mid, og, ot);
                cases++;
                if (og != rg || ot != rt) fails++;
                // Negative control: mover 1 starts from gaussian 0's cursor.
                if (mid > lo && mid * EPP < static_cast<uint32_t>(s.P) &&
                    s.offs[1] <= static_cast<int>(mid) * EPP) {
                    std::vector<int> cg(p_pages * EPP, -1), ct(p_pages * EPP, -1);
                    run_scatter(s, lo, mid - lo, cg, ct);
                    run_scatter(s, mid, hi - mid, cg, ct, /*stale_cursor=*/true);
                    control_cases++;
                    if (cg != rg || ct != rt) control_hits++;
                }
            }
        }
    }
    std::printf("tile_assign dual-mover split: %ld cases, %ld mismatches; "
                "stale-cursor control caught %ld/%ld\n",
                cases, fails, control_hits, control_cases);
    return (fails == 0 && control_cases > 0 && control_hits == control_cases) ? 0 : 1;
}
