// Model check for the dual-data-mover split of tile_assign K1 / K2
// (render/kernels/dataflow/tile_assign_bbox.cpp, tile_assign_scatter.cpp; host
// side in render/host/tile_assign_device.cpp). Standalone:
//
//   c++ -O2 -std=c++17 tests/unit/test_tile_assign_dual_mover.cpp -o /tmp/t_ta && /tmp/t_ta
//
// Replays each kernel's page loop for one core's page range on random inputs:
// once on a single mover over [lo, hi), and once split at every page boundary
// mid (BRISC = [lo, mid), NCRISC = [mid, hi)). Both movers write into the same
// output array (scatter movers interleaved pair by pair, as they run
// concurrently); the split must leave exactly the single-mover bytes. A
// negative control gives both scatter movers the same scratch CBs (no private
// CB copies) and must mismatch. Returns non-zero on failure.
#include <algorithm>
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

// tile_assign_scatter.cpp for one mover, stepped one pair at a time so two
// movers can be interleaved. `l1` is the mover's scratch (CB_PX..RY attribute
// page + CB_GID/CB_TID staging); the dual-mover build gives each mover its own.
struct L1 {
    int attr_page = -1;  // page index held by the attribute CBs
    float px[EPP], py[EPP], rx[EPP], ry[EPP];
    int gid[EPP], tid[EPP];
};

struct ScatterMover {
    const Scene* s;
    L1* l1;
    int p = 0, p_end = 0, g = 0, out_idx = 0, attr_page = -1;
    Box b{};

    Box load(int gg) {  // load_attrs: refetch the page only on a page change
        const int pg = gg / EPP, ip = gg % EPP;
        if (pg != attr_page) {
            for (int i = 0; i < EPP; i++) {
                const int x = std::min(pg * EPP + i, s->M - 1);
                l1->px[i] = s->px[x]; l1->py[i] = s->py[x];
                l1->rx[i] = s->rx[x]; l1->ry[i] = s->ry[x];
            }
            attr_page = l1->attr_page = pg;
        }
        Scene one;
        one.tiles_x = s->tiles_x; one.tiles_y = s->tiles_y; one.inv_tsf = s->inv_tsf;
        one.px = {l1->px[ip]}; one.py = {l1->py[ip]}; one.rx = {l1->rx[ip]}; one.ry = {l1->ry[ip]};
        return aabb(one, 0);
    }
    void start(uint32_t page_start, uint32_t count) {
        p = static_cast<int>(page_start) * EPP;
        p_end = p + static_cast<int>(count) * EPP;
        if (count == 0 || p >= s->P) { p_end = p; return; }  // kernel early-outs
        int lo = 0, hi = s->M - 1;  // binary search: largest g with offs[g] <= p_start
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (s->offs[mid] <= p) lo = mid; else hi = mid - 1;
        }
        g = lo;
        b = load(g);
    }
    bool done() const { return p >= p_end; }
    void step(std::vector<int>& gid, std::vector<int>& tid) {
        int og = 0, ot = 0;
        if (p < s->P) {
            while (p >= s->offs[g + 1]) b = load(++g);
            const int local = p - s->offs[g];
            const int dy = local / b.w;
            const int dx = local - dy * b.w;
            og = g;
            ot = (b.miny + dy) * s->tiles_x + (b.minx + dx);
        }
        l1->gid[out_idx] = og;
        l1->tid[out_idx] = ot;
        if (++out_idx == EPP) {  // write out the full page
            const int base = p + 1 - EPP;
            for (int i = 0; i < EPP; i++) { gid[base + i] = l1->gid[i]; tid[base + i] = l1->tid[i]; }
            out_idx = 0;
        }
        p++;
    }
};

// Movers run concurrently on the device: interleave them one pair at a time.
void run_scatter_split(const Scene& s, uint32_t lo, uint32_t mid, uint32_t hi, bool shared_l1,
                       std::vector<int>& gid, std::vector<int>& tid) {
    L1 a, b;
    ScatterMover m0{&s, &a}, m1{&s, shared_l1 ? &a : &b};
    m0.start(lo, mid - lo);
    m1.start(mid, hi - mid);
    while (!m0.done() || !m1.done()) {
        if (!m0.done()) m0.step(gid, tid);
        if (!m1.done()) m1.step(gid, tid);
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

        // K2: a core range from pair page 0 and one from inside the pairs, both
        // running past P (host_free over-provisions the split to a ceiling).
        // Compared over the pages that hold pairs < P: sort never reads past P,
        // and pages wholly past P are left unwritten by any mover whose range
        // starts past P (single-mover cores past P already skip them).
        const uint32_t p_pages = static_cast<uint32_t>((s.P + EPP - 1) / EPP) + 3;
        const size_t live = static_cast<size_t>((s.P + EPP - 1) / EPP) * EPP;
        auto same = [&](const std::vector<int>& a, const std::vector<int>& b) {
            return std::equal(a.begin(), a.begin() + live, b.begin());
        };
        for (uint32_t lo : {0u, p_pages / 3}) {
            const uint32_t hi = p_pages;
            std::vector<int> rg(p_pages * EPP, -1), rt(p_pages * EPP, -1);
            run_scatter_split(s, lo, hi, hi, false, rg, rt);  // single mover
            for (uint32_t mid = lo; mid <= hi; mid++) {
                std::vector<int> og(p_pages * EPP, -1), ot(p_pages * EPP, -1);
                run_scatter_split(s, lo, mid, hi, false, og, ot);
                cases++;
                if (!same(og, rg) || !same(ot, rt)) fails++;
                // Negative control: both movers on one set of scratch CBs.
                if (mid > lo && mid < hi && static_cast<int>(mid) * EPP < s.P) {
                    std::vector<int> cg(p_pages * EPP, -1), ct(p_pages * EPP, -1);
                    run_scatter_split(s, lo, mid, hi, true, cg, ct);
                    control_cases++;
                    if (!same(cg, rg) || !same(ct, rt)) control_hits++;
                }
            }
        }
    }
    std::printf("tile_assign dual-mover split: %ld cases, %ld mismatches; "
                "shared-scratch control caught %ld/%ld\n",
                cases, fails, control_hits, control_cases);
    return (fails == 0 && control_cases > 0 && control_hits == control_cases) ? 0 : 1;
}
