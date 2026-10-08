// Review #391: GSPLAT_TT_OUT_ZEROCOPY passes no image buffer to the sort->blend
// continuation, yet the frame still has an output (the pinned ring slot). The
// fused mat+blend gate (sort_device.cpp) once checked image_out alone, so the
// zero-copy arm silently ran the unfused materialize then blend.
//
//   CXXFLAGS=-Isrc tests/unit/run_cpp.sh tests/unit/test_sort_blend_has_output.cpp
#include <cstdint>
#include <cstdio>

#include "render/host/sort.h"

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "%s:%d CHECK(%s)\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

int main() {
    using gsplat_tt::SortBlendContinuation;
    using gsplat_tt::sort_blend_has_output;
    uint8_t buf[3] = {};
    SortBlendContinuation with_buf;
    with_buf.image_out = buf;
    SortBlendContinuation zc;  // zero-copy: no buffer
    CHECK(sort_blend_has_output(&with_buf, false));
    CHECK(sort_blend_has_output(&with_buf, true));
    CHECK(sort_blend_has_output(&zc, true));    // the review #391 case
    CHECK(!sort_blend_has_output(&zc, false));  // no buffer, no zero-copy
    CHECK(!sort_blend_has_output(nullptr, true));
    CHECK(!sort_blend_has_output(nullptr, false));
    if (fails == 0) std::printf("ok\n");
    return fails == 0 ? 0 : 1;
}
