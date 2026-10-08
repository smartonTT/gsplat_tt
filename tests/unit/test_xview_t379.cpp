// Task #379: the cross-view prefetch key (render/host/xview.h).
//   tests/unit/run_cpp.sh tests/unit/test_xview_t379.cpp
#include "render/host/xview.h"

#include <cassert>
#include <cstdio>

using namespace gsplat_tt::xview;

int main() {
    float means[3], cov[6], col[3], op[1];
    float extr[16], intr[9];
    for (int i = 0; i < 16; i++) extr[i] = 0.1f * i;
    for (int i = 0; i < 9; i++) intr[i] = 1.0f + i;
    auto key = [&](const float* e) {
        return make_key(means, cov, e, intr, col, op, 0.004f, 6000000, 1037, 1600, 1024, 16,
                        1.0f / 255, false);
    };
    const Key k0 = key(extr);
    assert(same(k0, key(extr)));

    // Each field breaks the match.
    {
        Key k = k0; k.means = means + 1; assert(!same(k0, k));
        k = k0; k.cov3d = cov + 1; assert(!same(k0, k));
        k = k0; k.colors = col + 1; assert(!same(k0, k));
        k = k0; k.opacities = nullptr; assert(!same(k0, k));
        k = k0; k.n = 5; assert(!same(k0, k));
        for (int i = 0; i < 16; i++) { k = k0; k.extr[i] += 1e-6f; assert(!same(k0, k)); }
        for (int i = 0; i < 9; i++) { k = k0; k.intr[i] += 1e-3f; assert(!same(k0, k)); }
        k = k0; k.min_opacity = 0.005f; assert(!same(k0, k));
        k = k0; k.image_height = 1038; assert(!same(k0, k));
        k = k0; k.image_width = 1601; assert(!same(k0, k));
        k = k0; k.max_radius = 512; assert(!same(k0, k));
        k = k0; k.tile_size = 32; assert(!same(k0, k));
        k = k0; k.mb_contrib_floor = 1.0f / 128; assert(!same(k0, k));
        k = k0; k.cull_disabled = true; assert(!same(k0, k));
    }

    Prefetch p;
    // Nothing queued: run pfwc.
    assert(!p.consume(k0));
    // Hit, then consumed once only.
    p.set(k0);
    assert(p.pending());
    assert(p.consume(key(extr)));
    assert(!p.pending());
    assert(!p.consume(k0));
    // Wrong hint (other pose): miss, pfwc runs for the real view.
    float extr2[16];
    for (int i = 0; i < 16; i++) extr2[i] = extr[i];
    extr2[3] = 9.0f;
    p.set(key(extr2));
    assert(!p.consume(k0));
    assert(!p.pending());
    // Overflow retry of view N at a coarser floor after N+1 was prefetched.
    p.set(key(extr2));
    Key retry = k0;
    retry.mb_contrib_floor = 1.0f / 64;
    assert(!p.consume(retry));
    // ... and the next view (N+1) then runs its own pfwc.
    assert(!p.consume(key(extr2)));
    // clear() drops a queued prefetch.
    p.set(k0);
    p.clear();
    assert(!p.consume(k0));
    assert(p.hits() == 1 && p.misses() == 2);
    std::printf("ok\n");
    return 0;
}
