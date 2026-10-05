// Host checks for lever 2 (task #99, GSPLAT_TT_SFPU_VIS). Standalone:
//
//   c++ -O2 -ffp-contract=off -std=c++17 -Irender/kernels/dataflow \
//       tests/unit/test_vis_lever2.cpp -o /tmp/t_vis && /tmp/t_vis
//
// 1. sfpu_model() is the per-lane operation sequence of pfwc_vis_one() in
//    render/kernels/compute/project_pfwc_compute.cpp (PFWC_VIS), in host fp32
//    (round-to-nearest-even, like SFPMAD). After the pfwc writer's
//    classify_tile() (which resolves RECHECK words), every word must equal
//    vis_tile::exact_words(), i.e. gather_pred::visible_bits (the proj_count
//    predicate) and the tile_assign_bbox.cpp K1 rectangle. Inputs include
//    ties at the image edges, at tile borders, at k_near / min_opacity /
//    max_radius, +-0, inf and NaN.
// 2. The new pipeline (pfwc words -> writer mask/counts -> gather_vis_scan ->
//    gather_vis_scatter -> tile_assign K2 on the packed AABB) gives the same
//    M, P, compact order, offs[0..M] and (gid, tid) pair list as the legacy one
//    (proj_count + gather_scan_bases + proj_scatter, K1 + scans + K2), for the
//    balanced cut and for the legacy per-core halves, and every compact slot
//    is written exactly once.
// Must not be built with -ffast-math (NaN, -0 and rounding semantics).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "vis_tile.h"

namespace {

using vis_tile::Params;

float f(uint32_t b) { float x; std::memcpy(&x, &b, 4); return x; }
uint32_t u(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }

// SFPU compare semantics used by pfwc_vis_one: `d <= 0.0f` is true for -0, +0
// and negatives (sign bit or zero); NaN lanes never reach a decision (RECHECK).
bool le0(float d) { const uint32_t b = u(d); return (b & 0x80000000u) || (b & 0x7FFFFFFFu) == 0; }
bool gt0(float d) { return !le0(d); }

struct SfpuParams {
    float kn, mo, W, H, R, inv, tx1, ty1;
};

SfpuParams sfpu_params(const Params& p) {
    return {f(p.k_near), f(p.min_opacity), f(p.img_w), f(p.img_h), f(p.max_radius), p.inv_tile,
            static_cast<float>(p.tiles_x - 1), static_cast<float>(p.tiles_y - 1)};
}

// vec_min_max(lo, hi) on the SFPU leaves min in lo and max in hi.
float vmax(float a, float b) { return a > b ? a : b; }
float vmin(float a, float b) { return a < b ? a : b; }

// Round-toward-zero fp32 add, to model an SFPMAD that is faithful but not
// nearest-even (exact for the operand ranges that can be visible).
float add_rtz(float a, float b) {
    const double x = static_cast<double>(a) + static_cast<double>(b);
    float r = static_cast<float>(x);
    if (static_cast<double>(r) != x && std::fabs(static_cast<double>(r)) > std::fabs(x))
        r = std::nextafter(r, 0.0f);
    return r;
}

bool g_rtz = false;   // model the fl(m + r) adds with add_rtz
float g_tau = 1.0f / 4096.0f;

// pfwc_vis_one, one lane, in the kernel's order. Keep in step with the kernel.
void sfpu_model(float tz, float op, float mx, float my, float rx, float ry, const SfpuParams& P,
                uint32_t* tpg_word, uint32_t* aabb_word) {
    constexpr float TWO23 = 8388608.0f;
    // 1. Non-finite input: (biased exponent + 1) has bit 8 set only for 255.
    auto e1 = [](float x) { return ((u(x) >> 23) & 0xFFu) + 1u; };
    bool rck = ((e1(tz) | e1(op) | e1(mx) | e1(my) | e1(rx) | e1(ry)) & 0x100u) != 0;
    auto cell = [&](float q, float hi) {
        q = vmax(q, 0.0f);
        q = vmin(q, hi);
        float r = (q + TWO23) - TWO23;  // nearest integer (any faithful rounding works)
        if (gt0(r - q)) r = r - 1.0f;   // -> floor(q) == trunc(q) for q >= 0
        return r;
    };
    // 2. Predicate, then the cells; x first, then y.
    bool fail = false;
    if (le0(tz - P.kn)) fail = true;  // tz <= k_near
    if (gt0(P.mo - op)) fail = true;  // op < min_opacity
    if (le0(rx)) fail = true;
    if (gt0(rx - P.R)) fail = true;
    const float sx = g_rtz ? add_rtz(mx, rx) : mx + rx;
    const float dx = mx - rx;
    if (le0(sx)) fail = true;          // !(-rx < mx)
    if (le0(P.W - dx)) fail = true;    // !(fl(mx - rx) < W)
    const float x0 = cell(dx * P.inv, P.tx1);
    const float qx = sx * P.inv;
    const float x1 = cell(qx, P.tx1);
    if (!fail && !gt0(qx - x1 - g_tau)) rck = true;
    if (!fail && !(u(x1 + 1.0f - qx) & 0x80000000u) && !gt0(x1 + 1.0f - qx - g_tau)) rck = true;
    if (le0(ry)) fail = true;
    if (gt0(ry - P.R)) fail = true;
    const float sy = g_rtz ? add_rtz(my, ry) : my + ry;
    const float dy = my - ry;
    if (le0(sy)) fail = true;
    if (le0(P.H - dy)) fail = true;
    const float y0 = cell(dy * P.inv, P.ty1);
    const float qy = sy * P.inv;
    const float y1 = cell(qy, P.ty1);
    if (!fail && !gt0(qy - y1 - g_tau)) rck = true;
    if (!fail && !(u(y1 + 1.0f - qy) & 0x80000000u) && !gt0(y1 + 1.0f - qy - g_tau)) rck = true;
    // 3. Words: u(x + 2^23) == 0x4B000000 + x for an integer x in [0, 2^23).
    const float wm1 = x1 - x0, hm1 = y1 - y0;
    const float low = x0 + y0 * 1024.0f;
    *aabb_word = (u(low + TWO23) - 0x0B000000u) + ((u(wm1 + TWO23) - 0x4B000000u) << 20);
    const float tpg = (wm1 + 1.0f) * (hm1 + 1.0f);
    uint32_t w = u(tpg + TWO23) - 0x0B000000u;
    if (fail) w = 0;
    if (rck) w = vis_tile::RECHECK;
    *tpg_word = w;
}

Params make_params(uint32_t W, uint32_t H, uint32_t tile, float max_r, float min_op) {
    Params p;
    p.k_near = u(0.2f);
    p.min_opacity = u(min_op);
    p.img_w = u(static_cast<float>(W));
    p.img_h = u(static_cast<float>(H));
    p.max_radius = u(max_r);
    p.tiles_x = (W + tile - 1) / tile;
    p.tiles_y = (H + tile - 1) / tile;
    p.tile_shift = dm_fp32::pow2_shift(tile);
    p.inv_tile = 1.0f / static_cast<float>(tile);
    return p;
}

struct Scene {
    uint32_t N = 0;
    std::vector<uint32_t> tz, op, mx, my, rx, ry;
};

// Random pfwc outputs with many exact ties. rx/ry are ceil(k sqrt(a)) like pfwc.
Scene make_scene(std::mt19937& rng, uint32_t N, const Params& p, bool nasty, bool ties = true) {
    Scene s;
    s.N = N;
    for (auto* v : {&s.tz, &s.op, &s.mx, &s.my, &s.rx, &s.ry}) v->resize(N);
    const float W = f(p.img_w), H = f(p.img_h), R = f(p.max_radius);
    const float tile = static_cast<float>(1u << p.tile_shift);
    auto uni = [&](float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(rng() & 0xFFFFFF) / 16777216.0f;
    };
    auto nudge = [&](float x) {  // a few ulps either way
        const int k = static_cast<int>(rng() % 7) - 3;
        uint32_t b = u(x);
        if (x != 0.0f) b = static_cast<uint32_t>(static_cast<int32_t>(b) + k);
        return f(b);
    };
    const float specials[] = {INFINITY, -INFINITY, NAN, -NAN, 0.0f, -0.0f};
    for (uint32_t i = 0; i < N; i++) {
        float tz = uni(-1.0f, 40.0f);
        float op = uni(0.0f, 1.0f);
        float rx = std::ceil(3.0f * std::sqrt(uni(0.0f, 1.0f) < 0.9f ? uni(0.3f, 400.0f)
                                                                     : uni(0.3f, 40000.0f)));
        float ry = std::ceil(3.0f * std::sqrt(uni(0.3f, 400.0f)));
        float mx = uni(-700.0f, W + 700.0f);
        float my = uni(-700.0f, H + 700.0f);
        switch (ties ? rng() % 16 : 15u) {
            case 0: mx = nudge(std::floor(uni(-2.0f, W / tile + 2.0f)) * tile + rx); break;
            case 1: mx = nudge(std::floor(uni(-2.0f, W / tile + 2.0f)) * tile - rx); break;
            case 2: my = nudge(std::floor(uni(-2.0f, H / tile + 2.0f)) * tile + ry); break;
            case 3: mx = nudge(W + rx); break;
            case 4: mx = nudge(-rx); break;
            case 5: my = nudge(H + ry); break;
            case 6: tz = nudge(0.2f); break;
            case 7: op = nudge(f(p.min_opacity)); break;
            case 8: rx = std::floor(R) + static_cast<float>(rng() % 3) - 1.0f; break;
            case 9: rx = static_cast<float>(rng() % 2); ry = 0.0f; break;
            case 10: mx = uni(-1.0f, 1.0f) * 1e30f; break;
            default: break;
        }
        if (nasty && rng() % 64 == 0) {
            const float sp = specials[rng() % 6];
            switch (rng() % 6) {
                case 0: tz = sp; break;
                case 1: op = sp; break;
                case 2: mx = sp; break;
                case 3: my = sp; break;
                case 4: rx = sp; break;
                default: ry = sp; break;
            }
        }
        s.tz[i] = u(tz); s.op[i] = u(op); s.mx[i] = u(mx);
        s.my[i] = u(my); s.rx[i] = u(rx); s.ry[i] = u(ry);
    }
    return s;
}

struct Tiles {  // pfwc words + writer outputs
    std::vector<uint32_t> tpg, aabb;    // N-indexed, padded to whole tiles
    std::vector<uint32_t> mask;         // 32 words per tile
    std::vector<uint32_t> counts;       // [visible, pairs] per tile
};

Tiles run_pfwc(const Scene& s, const Params& p, uint64_t* recheck) {
    const uint32_t nt = (s.N + 1023) / 1024;
    Tiles t;
    t.tpg.assign(nt * 1024, 0x12345678u);  // padding lanes: garbage the writer must ignore
    t.aabb.assign(nt * 1024, 0x12345678u);
    t.mask.assign(nt * 32, 0);
    t.counts.assign(nt * 2, 0);
    const SfpuParams P = sfpu_params(p);
    for (uint32_t i = 0; i < s.N; i++) {
        sfpu_model(f(s.tz[i]), f(s.op[i]), f(s.mx[i]), f(s.my[i]), f(s.rx[i]), f(s.ry[i]), P,
                   &t.tpg[i], &t.aabb[i]);
        if (t.tpg[i] == vis_tile::RECHECK) (*recheck)++;
    }
    for (uint32_t k = 0; k < nt; k++) {
        const uint32_t base = k * 1024, n_el = std::min<uint32_t>(1024, s.N - base);
        auto get = [&](uint32_t il, uint32_t& tz, uint32_t& op, uint32_t& mx, uint32_t& my,
                       uint32_t& rx, uint32_t& ry) {
            const uint32_t i = base + il;
            tz = s.tz[i]; op = s.op[i]; mx = s.mx[i]; my = s.my[i]; rx = s.rx[i]; ry = s.ry[i];
        };
        uint32_t vc, pc;
        vis_tile::classify_tile(t.tpg.data() + base, t.aabb.data() + base, n_el,
                                t.mask.data() + k * 32, p, get, &vc, &pc);
        t.counts[2 * k] = vc;
        t.counts[2 * k + 1] = pc;
    }
    return t;
}

uint64_t g_bad = 0;
void fail(const char* what, uint64_t a, uint64_t b) {
    if (g_bad++ < 20) std::printf("MISMATCH %s: %llu vs %llu\n", what, (unsigned long long)a,
                                  (unsigned long long)b);
}

// Check 1: per-gaussian words after the writer == exact_words.
void check_words(const Scene& s, const Params& p, const Tiles& t, uint64_t* visible) {
    for (uint32_t i = 0; i < s.N; i++) {
        uint32_t ab = 0;
        const uint32_t ref = vis_tile::exact_words(s.tz[i], s.op[i], s.mx[i], s.my[i], s.rx[i],
                                                   s.ry[i], p, &ab);
        if (t.tpg[i] != ref) {
            fail("tpg word", t.tpg[i], ref);
            if (g_bad < 20)
                std::printf("  i=%u tz=%a op=%a mx=%a my=%a rx=%a ry=%a\n", i, f(s.tz[i]),
                            f(s.op[i]), f(s.mx[i]), f(s.my[i]), f(s.rx[i]), f(s.ry[i]));
        }
        if (ref != 0) {
            (*visible)++;
            if (t.aabb[i] != ab) fail("aabb word", t.aabb[i], ab);
        }
        const bool bit = (t.mask[(i / 1024) * 32 + (i % 1024) / 32] >> (i % 32)) & 1u;
        if (bit != (ref != 0)) fail("mask bit", bit, ref != 0);
    }
}

struct Result {
    uint32_t M = 0, P = 0;
    std::vector<uint32_t> src;   // compact g -> source index
    std::vector<uint32_t> offs;  // M + 1 entries
    std::vector<uint32_t> gid, tid;
};

// tile_assign K2 (tile_assign_scatter.cpp) over offs + a per-g rectangle.
void k2(Result& r, const std::vector<uint32_t>& minx, const std::vector<uint32_t>& miny,
        const std::vector<uint32_t>& w, uint32_t tiles_x) {
    r.gid.assign(r.P, 0);
    r.tid.assign(r.P, 0);
    for (uint32_t g = 0; g < r.M; g++)
        for (uint32_t pp = r.offs[g]; pp < r.offs[g + 1]; pp++) {
            const uint32_t local = pp - r.offs[g], dy = local / w[g], dx = local - dy * w[g];
            r.gid[pp] = g;
            r.tid[pp] = (miny[g] + dy) * tiles_x + (minx[g] + dx);
        }
}

// Legacy: proj_count + gather_scan_bases + proj_scatter (strided tiles, slot =
// 2 * core + mover, mover 0 = first half), then K1 + exclusive scan + K2.
Result run_legacy(const Scene& s, const Params& p, uint32_t cores) {
    const uint32_t nt = (s.N + 1023) / 1024;
    Result r;
    for (uint32_t c = 0; c < cores; c++) {
        const uint32_t n_all = c < nt ? (nt - 1 - c) / cores + 1 : 0;
        const uint32_t h0 = (n_all * 500 + 500) / 1000;
        for (uint32_t mv = 0; mv < 2; mv++) {
            const uint32_t first = c + (mv == 0 ? 0 : h0 * cores), cnt = mv == 0 ? h0 : n_all - h0;
            for (uint32_t k = 0, t = first; k < cnt; k++, t += cores)
                for (uint32_t il = 0; il < 1024 && t * 1024 + il < s.N; il++) {
                    const uint32_t i = t * 1024 + il;
                    if (gather_pred::visible_bits(s.tz[i], s.op[i], s.mx[i], s.my[i], s.rx[i],
                                                  s.ry[i], p.k_near, p.min_opacity, p.img_w,
                                                  p.img_h, p.max_radius))
                        r.src.push_back(i);
                }
        }
    }
    r.M = static_cast<uint32_t>(r.src.size());
    std::vector<uint32_t> minx(r.M), miny(r.M), w(r.M);
    r.offs.assign(r.M + 1, 0);
    uint32_t acc = 0;
    for (uint32_t g = 0; g < r.M; g++) {  // K1 on the compact px/py/rx/ry
        const uint32_t i = r.src[g];
        const int tx1 = static_cast<int>(p.tiles_x) - 1, ty1 = static_cast<int>(p.tiles_y) - 1;
        const uint32_t sh = p.tile_shift;
        const int x0 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(s.mx[i], s.rx[i] ^ dm_fp32::SIGN, sh, p.inv_tile), 0, tx1);
        const int x1 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(s.mx[i], s.rx[i], sh, p.inv_tile), 0, tx1);
        const int y0 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(s.my[i], s.ry[i] ^ dm_fp32::SIGN, sh, p.inv_tile), 0, ty1);
        const int y1 = vis_tile::clampi(dm_fp32::add_mul_pow2_to_int(s.my[i], s.ry[i], sh, p.inv_tile), 0, ty1);
        minx[g] = x0; miny[g] = y0; w[g] = x1 - x0 + 1;
        r.offs[g] = acc;
        acc += (x1 - x0 + 1) * (y1 - y0 + 1);
    }
    r.offs[r.M] = acc;
    r.P = acc;
    k2(r, minx, miny, w, p.tiles_x);
    return r;
}

// New: gather_vis_scan + gather_vis_scatter (+ the scan's offs tail page) + K2
// on the packed AABB.
Result run_new(const Scene& s, const Params& p, const Tiles& t, const vis_tile::ScanArgs& a) {
    const uint32_t S = a.num_cores * a.movers;
    std::vector<uint32_t> slots(S * vis_tile::SLOT_WORDS, 0xDEADBEEFu);
    Result r;
    vis_tile::scan_slots(t.counts.data(), a, slots.data(), &r.M, &r.P);
    const uint32_t cap = (r.M / 16 + 1) * 16;
    std::vector<uint32_t> offs(cap, 0xFFFFFFFFu), aabb(cap, 0), writes(cap, 0);
    r.src.assign(r.M, 0xFFFFFFFFu);
    for (uint32_t k = 0; k < 16; k++) offs[(r.M / 16) * 16 + k] = r.P;  // scan: offs tail page
    vis_tile::SeqMap sm;
    sm.init(a.num_tiles, a.num_cores);
    uint32_t qcover = 0;
    for (uint32_t sl = 0; sl < S; sl++) {
        const uint32_t* sp = &slots[sl * vis_tile::SLOT_WORDS];
        uint32_t g = sp[vis_tile::S_BASE], pr = sp[vis_tile::S_PBASE];
        const uint32_t q0 = sp[vis_tile::S_Q0], qn = sp[vis_tile::S_QN];
        if (q0 != qcover) fail("slot ranges not contiguous", q0, qcover);
        qcover = q0 + qn;
        uint32_t c = 0, k = 0;
        if (qn) sm.locate(q0, &c, &k);
        for (uint32_t q = 0; q < qn; q++) {
            const uint32_t tile = c + k * a.num_cores;
            for (uint32_t wd = 0; wd < 32; wd++)
                for (uint32_t bits = t.mask[tile * 32 + wd]; bits; bits &= bits - 1) {
                    const uint32_t i = tile * 1024 + wd * 32 + __builtin_ctz(bits);
                    if (g >= r.M) { fail("g past M", g, r.M); break; }
                    r.src[g] = i;
                    offs[g] = pr;
                    aabb[g] = t.aabb[i] & vis_tile::PAYLOAD;
                    writes[g]++;
                    pr += t.tpg[i] & vis_tile::PAYLOAD;
                    g++;
                }
            if (++k == sm.count(c)) { c++; k = 0; }
        }
        if (sp[vis_tile::S_IS_LAST] && g % 16)
            for (uint32_t x = g; x % 16; x++) { offs[x] = pr; writes[x]++; }
    }
    if (qcover != a.num_tiles) fail("slots do not cover the sequence", qcover, a.num_tiles);
    for (uint32_t g = 0; g < cap; g++)
        if (writes[g] > 1) fail("slot written twice", g, writes[g]);
    for (uint32_t g = 0; g < r.M; g++)
        if (writes[g] != 1) fail("compact slot not written", g, writes[g]);
    r.offs.assign(offs.begin(), offs.begin() + r.M + 1);
    std::vector<uint32_t> minx(r.M), miny(r.M), w(r.M);
    for (uint32_t g = 0; g < r.M; g++) {
        minx[g] = vis_tile::aabb_min_x(aabb[g]);
        miny[g] = vis_tile::aabb_min_y(aabb[g]);
        w[g] = vis_tile::aabb_w(aabb[g]);
    }
    k2(r, minx, miny, w, p.tiles_x);
    return r;
}

void compare(const Result& a, const Result& b) {
    if (a.M != b.M) { fail("M", a.M, b.M); return; }
    if (a.P != b.P) { fail("P", a.P, b.P); return; }
    for (uint32_t g = 0; g < a.M; g++)
        if (a.src[g] != b.src[g]) { fail("compact order", g, a.src[g]); return; }
    for (uint32_t g = 0; g <= a.M; g++)
        if (a.offs[g] != b.offs[g]) { fail("offs", g, a.offs[g]); return; }
    for (uint32_t q = 0; q < a.P; q++)
        if (a.gid[q] != b.gid[q] || a.tid[q] != b.tid[q]) { fail("pair", q, a.tid[q]); return; }
}

}  // namespace

int main() {
    std::mt19937 rng(99);
    uint64_t gaussians = 0, visible = 0, recheck = 0, pipelines = 0;
    struct Cfg { uint32_t W, H, tile; float maxr, minop; };
    const Cfg cfgs[] = {
        {1024, 1024, 32, 512.0f, 1.0f / 255.0f},   // bicycle bench
        {1024, 1024, 32, 2147483648.0f, 1.0f / 255.0f},  // radius cap off
        {1000, 700, 16, 350.0f, 0.005f},
        {1920, 1080, 32, 540.0f, 0.0f},
    };
    for (int trial = 0; trial < 40; trial++) {
        const Cfg& c = cfgs[trial % 4];
        const Params p = make_params(c.W, c.H, c.tile, c.maxr, c.minop);
        const uint32_t N = 1 + rng() % (1024 * (trial < 4 ? 300 : 40));
        const Scene s = make_scene(rng, N, p, trial % 2 == 1);
        g_rtz = (trial % 3 == 2);  // also model a round-toward-zero SFPMAD
        const Tiles t = run_pfwc(s, p, &recheck);
        check_words(s, p, t, &visible);
        g_rtz = false;
        gaussians += N;
        const uint32_t nt = (N + 1023) / 1024;
        for (uint32_t cores : {1u, 3u, 13u, 110u}) {
            const Result ref = run_legacy(s, p, cores);
            for (uint32_t bal = 0; bal < 3; bal++) {
                vis_tile::ScanArgs a;
                a.num_tiles = nt;
                a.num_cores = cores;
                a.movers = 2;
                a.balance = bal ? 1u : 0u;
                a.tile_weight = bal == 2 ? 0u : 24u;
                a.empty_weight = bal == 2 ? 0u : 1u;
                compare(run_new(s, p, t, a), ref);
                pipelines++;
            }
        }
    }
    // Recheck rate on a scene without planted ties (the writer's extra work).
    {
        uint64_t rk = 0, vis2 = 0;
        const Params p = make_params(1024, 1024, 32, 512.0f, 1.0f / 255.0f);
        const Scene s = make_scene(rng, 1024 * 400, p, false, false);
        const Tiles t = run_pfwc(s, p, &rk);
        check_words(s, p, t, &vis2);
        std::printf("natural scene: %llu visible, %llu recheck (%.3f%% of visible)\n",
                    (unsigned long long)vis2, (unsigned long long)rk, 100.0 * rk / (vis2 ? vis2 : 1));
    }
    // Sensitivity: without the edge recheck a round-toward-zero adder must
    // break some rectangles (otherwise the RTZ runs above prove nothing).
    {
        const uint64_t bad_before = g_bad;
        g_rtz = true;
        g_tau = 0.0f;
        uint64_t rk = 0, vis2 = 0;
        const Params p = make_params(1024, 1024, 32, 512.0f, 1.0f / 255.0f);
        const Scene s = make_scene(rng, 1024 * 400, p, false);
        const Tiles t = run_pfwc(s, p, &rk);
        check_words(s, p, t, &vis2);
        const uint64_t caught = g_bad - bad_before;
        std::printf("sensitivity (RTZ adder, tau = 0): %llu word mismatches (expected > 0)\n",
                    (unsigned long long)caught);
        g_bad = bad_before + (caught == 0 ? 1 : 0);
        g_rtz = false;
        g_tau = 1.0f / 4096.0f;
    }
    std::printf("vis lever2: %llu gaussians (%llu visible, %llu recheck), %llu pipelines, "
                "%llu mismatches\n",
                (unsigned long long)gaussians, (unsigned long long)visible,
                (unsigned long long)recheck, (unsigned long long)pipelines,
                (unsigned long long)g_bad);
    return g_bad == 0 ? 0 : 1;
}
