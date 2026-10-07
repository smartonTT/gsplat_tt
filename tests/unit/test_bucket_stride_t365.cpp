// Host checks for task #365 (bucket tile stride vs DRAM bank count). Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_bucket_stride_t365.cpp
//
// The bucket is interleaved by 2 KB page over the DRAM banks, so tile t's page k
// sits in bank (t * stride + k) % nbanks. With the auto stride, page 0 of
// consecutive tiles must cover every bank on 7 (p100a) and 8 (p150) banks alike;
// the pre-#365 512-page stride put every tile's page 0 in bank 0 on 8 banks.
#include <cstdint>
#include <cstdio>
#include <set>

#include "render/host/sort_onelaunch_layout.h"

namespace so = gsplat_tt::sort_onelaunch;
int failures = 0;
void fail(const char* what, long a, long b) {
    if (failures++ < 20) std::fprintf(stderr, "FAIL %s: %ld %ld\n", what, a, b);
}

int main() {
    for (uint32_t nb : {6u, 7u, 8u, 12u, 16u}) {
        const uint32_t cap = so::bucket_tile_cap(so::kTileCap, nb, -1);
        if (cap % 64u != 0u || cap < so::kTileCap || cap > 43690u) fail("cap range", cap, nb);
        const uint32_t pages = cap / 64u;
        std::set<uint32_t> banks;
        for (uint32_t t = 0; t < 64u; t++) banks.insert((t * pages) % nb);
        if (banks.size() != nb) fail("page 0 of 64 tiles covers all banks", static_cast<long>(banks.size()), nb);
    }
    if (so::bucket_tile_cap(so::kTileCap, 7u, -1) != so::kTileCap) fail("p100a unchanged", 0, 7);
    if (so::bucket_tile_cap(so::kTileCap, 8u, -1) != so::kTileCap + 64u) fail("p150 513 pages", 0, 8);
    if (so::bucket_tile_cap(so::kTileCap, 8u, 0) != so::kTileCap) fail("pad 0 = pre-#365", 0, 8);
    if (so::bucket_tile_cap(so::kTileCap, 8u, 3) != so::kTileCap + 192u) fail("pad 3", 0, 8);
    if (so::bucket_tile_cap(so::kTileCapBig, 8u, -1) != so::kTileCapBig) fail("big cap odd already", 0, 8);
    if (failures == 0) std::printf("ok\n");
    return failures == 0 ? 0 : 1;
}
