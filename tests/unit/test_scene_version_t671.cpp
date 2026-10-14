// Task #671: version-keyed scene upload cache (render/host/scene_version.h)
// and the scene version in the xview prefetch key (render/host/xview.h).
//   tests/unit/run_cpp.sh tests/unit/test_scene_version_t671.cpp
#include "render/host/scene_version.h"
#include "render/host/xview.h"

#include <cassert>
#include <cstdio>

namespace sv = gsplat_tt::scene_version;
using namespace gsplat_tt::xview;

int main() {
    float means[6] = {0, 0, 0, 1, 1, 1};
    float other[6] = {};

    // Nothing uploaded yet: miss.
    sv::UploadKey c;
    const uint64_t v0 = sv::current();
    assert(v0 != 0);
    assert(!c.hit(means, 2, v0));
    c.record(means, 2, v0);
    assert(c.hit(means, 2, sv::current()));

    // In-place update (same pointer, same N, new values): the caller bumps,
    // the cache misses and the upload runs again.
    means[0] = 5.0f;
    const uint64_t v1 = sv::bump();
    assert(v1 == v0 + 1 && sv::current() == v1);
    assert(!c.hit(means, 2, sv::current()));
    c.record(means, 2, sv::current());
    assert(c.hit(means, 2, sv::current()));

    // A different tensor or N still misses without a bump.
    assert(!c.hit(other, 2, sv::current()));
    assert(!c.hit(means, 1, sv::current()));

    // Buffer regrow drops the cached state.
    c.invalidate();
    assert(!c.hit(means, 2, sv::current()));
    // A null pointer with N 0 at version 0 never counts as uploaded.
    assert(!sv::UploadKey{}.hit(nullptr, 0, 0));

    // xview: a pfwc prefetched before an in-place update must not be consumed
    // after it (it read the old parameters).
    float cov[12] = {}, col[6] = {}, op[2] = {};
    float extr[16] = {}, intr[9] = {};
    auto key = [&] {
        return make_key(means, cov, extr, intr, col, op, 0.004f, 2, 822, 1237, 1024, 16,
                        1.0f / 255, false, sv::current());
    };
    Prefetch p;
    p.set(key());
    assert(p.consume(key()));  // same version: hit
    p.set(key());
    means[1] = 7.0f;
    sv::bump();
    assert(!p.consume(key()));  // bumped: miss, pfwc runs again
    assert(p.misses() == 1 && p.hits() == 1);

    std::printf("test_scene_version_t671: ok\n");
    return 0;
}
