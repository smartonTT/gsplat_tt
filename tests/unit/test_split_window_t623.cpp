// Host checks for the single-frame 2-chip split (tasks #623/#624, GSPLAT_TT_SPLIT).
//
//   tests/unit/run_cpp.sh tests/unit/test_split_window_t623.cpp \
//     (CXXFLAGS="-Irender/host -Irender/kernels/dataflow")
//
// 1. Window clip (vis_tile::clip_window through classify_tile): for random
//    visible rectangles, the per-tile pair count of every tile inside a window
//    equals the one-chip (full screen) count, tiles outside get none, the
//    window-local ids map back to the global tile through the origin arg the
//    sort/mat/blend kernels decode, and left + right halves sum to the one-chip
//    pair total.
// 2. Window off: decode of the plain pfwc args 18/19 and of the plain tiles_x
//    slot is the full screen with no window, split_window passes every arg
//    through unchanged, and classify_tile gives identical words, mask and counts
//    with no window and with a full-screen window.
// 3. GSPLAT_TT_SPLIT / GSPLAT_TT_SPLIT_WINDOW parsing.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "split_window.h"
#include "tile_window.h"
#include "vis_tile.h"

namespace {

int g_fail = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++g_fail;                                                     \
        }                                                                 \
    } while (0)

constexpr uint32_t TX = 32, TY = 32, N = 1024;

struct Scene {
    std::vector<uint32_t> tpg, aabb;
};

// Random visible rectangles inside the TX x TY screen (some 1x1, some wide, a
// few invisible), as the pfwc writes them before classify_tile.
Scene make_scene(uint32_t seed) {
    std::mt19937 rng(seed);
    Scene s;
    s.tpg.assign(N, 0u);
    s.aabb.assign(N, 0u);
    for (uint32_t i = 0; i < N; ++i) {
        if (rng() % 8 == 0) continue;  // not visible
        const uint32_t x0 = rng() % TX, y0 = rng() % TY;
        const uint32_t big = rng() % 4 == 0;
        const uint32_t w = 1 + rng() % (big ? (TX - x0) : std::min<uint32_t>(4, TX - x0));
        const uint32_t h = 1 + rng() % (big ? (TY - y0) : std::min<uint32_t>(4, TY - y0));
        s.tpg[i] = vis_tile::TAG | (w * h);
        s.aabb[i] = vis_tile::TAG | vis_tile::aabb_pack(x0, y0, w);
    }
    return s;
}

vis_tile::Params params(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1, bool on) {
    vis_tile::Params p;
    p.tiles_x = TX;
    p.tiles_y = TY;
    // The writer's decode of the host's pfwc args 18/19 (pfwc_device.cpp).
    const uint32_t a18 = on ? tile_window::pack_pfwc_x(TX, x0, x1 - x0) : TX;
    const uint32_t a19 = on ? tile_window::pack_pfwc_y(TY, y0, y1 - y0) : TY;
    const tile_window::PfwcWin w = tile_window::decode_pfwc(a18, a19);
    CHECK(w.tiles_x == TX && w.tiles_y == TY);
    p.win = (w.wx != w.tiles_x || w.wy != w.tiles_y) ? 1u : 0u;
    p.win_x0 = w.x0;
    p.win_y0 = w.y0;
    p.win_w = w.wx;
    p.win_h = w.wy;
    return p;
}

struct Out {
    std::vector<uint32_t> tpg, aabb, mask;
    uint32_t vis = 0, pairs = 0;
};

Out classify(const Scene& s, const vis_tile::Params& p) {
    Out o;
    o.tpg = s.tpg;
    o.aabb = s.aabb;
    o.mask.assign(vis_tile::MASK_WORDS, 0u);
    auto no_inputs = [](uint32_t, uint32_t&, uint32_t&, uint32_t&, uint32_t&, uint32_t&, uint32_t&) {
        std::abort();  // no RECHECK words in these scenes
    };
    vis_tile::classify_tile(o.tpg.data(), o.aabb.data(), N, o.mask.data(), p, no_inputs, &o.vis,
                            &o.pairs);
    return o;
}

// Per-tile pair histogram of the emitted (rectangle -> tile id) pairs, mapped
// back to global tiles with the origin arg the downstream kernels decode.
std::vector<uint32_t> global_hist(const Out& o, uint32_t origin_arg) {
    const tile_window::Origin org = tile_window::decode(origin_arg);
    std::vector<uint32_t> h(TX * TY, 0u);
    for (uint32_t i = 0; i < N; ++i) {
        const uint32_t t = o.tpg[i];
        if (!(t & vis_tile::TAG)) continue;
        const uint32_t a = o.aabb[i];
        const uint32_t w = vis_tile::aabb_w(a), hh = (t & vis_tile::PAYLOAD) / w;
        CHECK(w * hh == (t & vis_tile::PAYLOAD));
        for (uint32_t y = 0; y < hh; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                const uint32_t lid = (vis_tile::aabb_min_y(a) + y) * org.w + vis_tile::aabb_min_x(a) + x;
                const uint32_t gx = lid % org.w + org.x0, gy = lid / org.w + org.y0;
                CHECK(gx < TX && gy < TY);
                if (gx < TX && gy < TY) h[gy * TX + gx]++;
            }
    }
    return h;
}

void test_window_clip() {
    for (uint32_t seed = 1; seed <= 20; ++seed) {
        const Scene s = make_scene(seed);
        const Out full = classify(s, params(0, TX, 0, TY, false));
        const std::vector<uint32_t> hf = global_hist(full, TX);
        uint32_t total = 0;
        for (uint32_t v : hf) total += v;
        CHECK(total == full.pairs);

        struct W { uint32_t x0, x1, y0, y1; };
        const W wins[] = {{0, 16, 0, 32}, {16, 32, 0, 32}, {0, 32, 0, 16}, {0, 32, 16, 32},
                          {5, 20, 7, 13}, {31, 32, 0, 1}, {0, 1, 31, 32}};
        uint32_t lr_pairs = 0;
        for (const W& w : wins) {
            const Out o = classify(s, params(w.x0, w.x1, w.y0, w.y1, true));
            const uint32_t origin = tile_window::pack(w.x1 - w.x0, w.x0, w.y0);
            const std::vector<uint32_t> hw = global_hist(o, origin);
            uint32_t in_pairs = 0;
            for (uint32_t ty = 0; ty < TY; ++ty)
                for (uint32_t tx = 0; tx < TX; ++tx) {
                    const bool in = tx >= w.x0 && tx < w.x1 && ty >= w.y0 && ty < w.y1;
                    CHECK(hw[ty * TX + tx] == (in ? hf[ty * TX + tx] : 0u));
                    if (in) in_pairs += hf[ty * TX + tx];
                }
            CHECK(o.pairs == in_pairs);
            // Visible on this chip = the rectangle meets the window; the mask
            // keeps storage order (a subset of the one-chip mask).
            uint32_t vis = 0;
            for (uint32_t i = 0; i < N; ++i) {
                const uint32_t t = s.tpg[i];
                bool meets = false;
                if (t & vis_tile::TAG) {
                    const uint32_t a = s.aabb[i], aw = vis_tile::aabb_w(a);
                    const uint32_t ah = (t & vis_tile::PAYLOAD) / aw;
                    const uint32_t ax = vis_tile::aabb_min_x(a), ay = vis_tile::aabb_min_y(a);
                    meets = ax < w.x1 && ax + aw > w.x0 && ay < w.y1 && ay + ah > w.y0;
                }
                const bool bit = (o.mask[i / 32] >> (i % 32)) & 1u;
                CHECK(bit == meets);
                CHECK(!bit || ((full.mask[i / 32] >> (i % 32)) & 1u));
                vis += meets ? 1u : 0u;
            }
            CHECK(o.vis == vis);
            if (w.y0 == 0 && w.y1 == TY && (w.x1 - w.x0) == 16) lr_pairs += o.pairs;
        }
        CHECK(lr_pairs == full.pairs);
    }
}

void test_window_off() {
    // Plain args decode to the full screen, no window.
    const tile_window::PfwcWin w = tile_window::decode_pfwc(TX, TY);
    CHECK(w.tiles_x == TX && w.tiles_y == TY && w.x0 == 0 && w.y0 == 0 && w.wx == TX && w.wy == TY);
    const tile_window::Origin o = tile_window::decode(TX);
    CHECK(o.w == TX && o.x0 == 0 && o.y0 == 0);
    // Pack / decode round trip of a window.
    const tile_window::Origin r = tile_window::decode(tile_window::pack(16, 16, 3));
    CHECK(r.w == 16 && r.x0 == 16 && r.y0 == 3);
    const tile_window::PfwcWin pw =
        tile_window::decode_pfwc(tile_window::pack_pfwc_x(TX, 16, 16), tile_window::pack_pfwc_y(TY, 2, 30));
    CHECK(pw.tiles_x == TX && pw.tiles_y == TY && pw.x0 == 16 && pw.wx == 16 && pw.y0 == 2 && pw.wy == 30);

    // split_window off: every host arg is the one-chip value.
    namespace sw = gsplat_tt::split_window;
    sw::current() = sw::Window{};
    CHECK(sw::origin_arg(TX) == TX);
    CHECK(sw::image_tiles_x(TX) == TX && sw::image_tiles_y(TY) == TY);
    CHECK(sw::grid_tiles_x(TX) == TX && sw::grid_tiles_y(TY) == TY);
    // On: the window's grid, full-image pitch, packed origin.
    sw::current() = sw::parse("1", "16,32,0,32", TX, TY);
    CHECK(sw::origin_arg(TX) == tile_window::pack(16, 16, 0));
    CHECK(sw::image_tiles_x(TX) == TX && sw::image_tiles_y(TY) == TY);
    CHECK(sw::grid_tiles_x(TX) == 16 && sw::grid_tiles_y(TY) == TY);
    sw::current() = sw::Window{};

    // classify_tile: no window == full-screen window, word for word.
    for (uint32_t seed = 100; seed < 105; ++seed) {
        const Scene s = make_scene(seed);
        const Out a = classify(s, params(0, TX, 0, TY, false));
        vis_tile::Params pf = params(0, TX, 0, TY, false);
        pf.win = 1;  // force the clip path on a full-screen window
        pf.win_w = TX;
        pf.win_h = TY;
        const Out b = classify(s, pf);
        CHECK(a.tpg == b.tpg && a.aabb == b.aabb && a.mask == b.mask);
        CHECK(a.vis == b.vis && a.pairs == b.pairs);
        CHECK(a.tpg == s.tpg && a.aabb == s.aabb);  // off leaves the words as written
    }
}

void test_parse() {
    namespace sw = gsplat_tt::split_window;
    CHECK(!sw::parse(nullptr, "0,16,0,32", TX, TY).on);
    CHECK(!sw::parse("", nullptr, TX, TY).on);
    CHECK(!sw::parse("0", "0,16,0,32", TX, TY).on);
    const sw::Window w = sw::parse("1", "0,16,0,32", TX, TY);
    CHECK(w.on && w.x0 == 0 && w.w == 16 && w.y0 == 0 && w.h == 32 && w.screen_tiles_x == TX &&
          w.screen_tiles_y == TY);
    CHECK(sw::same(w, sw::parse("1", "0,16,0,32", TX, TY)));
    CHECK(!sw::same(w, sw::parse("1", "16,32,0,32", TX, TY)));
    CHECK(!sw::same(w, sw::Window{}));
    CHECK(sw::same(sw::Window{}, sw::Window{}));
    const char* bad[] = {nullptr, "", "0,16,0", "0,16,0,32,1", "16,16,0,32", "0,33,0,32",
                         "-1,16,0,32", "0,16,0,0", "a,16,0,32", "0;16;0;32"};
    for (const char* b : bad) {
        bool threw = false;
        try {
            (void)sw::parse("1", b, TX, TY);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

}  // namespace

int main() {
    test_window_clip();
    test_window_off();
    test_parse();
    if (g_fail) {
        std::fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
