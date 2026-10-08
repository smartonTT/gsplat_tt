// Task #374: the zero-copy output ring (render/host/out_ring.h).
//
//   tests/unit/run_cpp.sh tests/unit/test_out_ring.cpp
//
// Checks: a caller that keeps the previous frame while rendering the next
// cycles through two slots and never gets a slot it still holds; a slot the
// caller dropped is reused; a held frame (the bench's hero) costs one extra
// slot; past the cap the ring replaces a slot but the caller's old lease (and
// its bytes) stay valid; a cap-1 ring (GSPLAT_TT_OUT_PINNED copy path) reuses
// its one slot.
#include <cstdio>
#include <memory>
#include <vector>

#include "render/host/out_ring.h"

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

struct Img {
    std::vector<int> px;
};
struct Map {
    int id = -1;
};
using Ring = gsplat_tt::OutRing<Img, Map>;

static int g_maps = 0;
static void make(Ring::Slot& s) {
    s.lease = std::make_shared<Img>();
    s.lease->px.assign(4, 0);
    s.extra.id = g_maps++;
}

// "Render" frame f into the next slot and hand its lease to the caller.
static std::shared_ptr<Img> frame(Ring& r, int f) {
    const std::size_t i = r.acquire(make);
    for (int& p : r.at(i).lease->px) p = f;
    return r.at(i).lease;
}

int main() {
    {  // Bench/viewer: keep frame N while N+1 renders.
        Ring r(4);
        std::shared_ptr<Img> held = frame(r, 0);
        for (int f = 1; f < 20; f++) {
            std::shared_ptr<Img> next = frame(r, f);
            CHECK(next.get() != held.get());
            CHECK(held->px[0] == f - 1);  // the kept frame was not overwritten
            held = next;
        }
        CHECK(r.size() == 2);
        CHECK(r.grows() == 2);
        CHECK(r.replaced() == 0);
    }
    {  // Dropped frames: the same slot comes back, no growth.
        Ring r(4);
        for (int f = 0; f < 5; f++) (void)frame(r, f);
        CHECK(r.size() == 1);
    }
    {  // Bench hero: one frame kept for the whole run costs one more slot.
        Ring r(4);
        std::shared_ptr<Img> hero = frame(r, 100);
        std::shared_ptr<Img> held;
        for (int f = 0; f < 10; f++) {
            std::shared_ptr<Img> next = frame(r, f);
            CHECK(next.get() != hero.get());
            held = next;
        }
        CHECK(hero->px[0] == 100);
        CHECK(r.size() == 3);
    }
    {  // Hoarding caller past the cap: slots are replaced, every kept frame survives.
        Ring r(3);
        const int maps0 = g_maps;
        std::vector<std::shared_ptr<Img>> kept;
        for (int f = 0; f < 8; f++) kept.push_back(frame(r, f));
        for (int f = 0; f < 8; f++) CHECK(kept[f]->px[0] == f);
        CHECK(r.size() == 3);
        CHECK(r.replaced() == 5);
        CHECK(g_maps - maps0 == 8);
        kept.clear();  // all free again: no more replacements
        for (int f = 0; f < 4; f++) (void)frame(r, f);
        CHECK(r.replaced() == 5);
    }
    {  // Cap 1 (copy path): the one slot is always reused.
        Ring r(1);
        for (int f = 0; f < 3; f++) {
            const std::size_t i = r.acquire(make);
            CHECK(i == 0);
        }
        CHECK(r.size() == 1);
        CHECK(r.replaced() == 0);
    }
    {  // clear(): a lease the caller holds outlives the ring's slots.
        Ring r(2);
        std::shared_ptr<Img> held = frame(r, 7);
        r.clear();
        CHECK(r.size() == 0);
        CHECK(held->px[0] == 7);
        (void)frame(r, 8);
        CHECK(held->px[0] == 7);
    }
    if (fails) std::fprintf(stderr, "%d check(s) failed\n", fails);
    return fails ? 1 : 0;
}
