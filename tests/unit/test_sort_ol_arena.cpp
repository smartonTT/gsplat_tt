// SPDX-License-Identifier: Apache-2.0
// Task #555: render/host/sort_ol_arena.h places the solo sort's scratch in the
// fused program's alias-cut tails: no overlap, inside the regions, aligned,
// and a clean failure when it does not fit.
//   c++ -O2 -std=c++17 -Irender/host tests/unit/test_sort_ol_arena.cpp
#include <cstdio>
#include <vector>

#include "sort_ol_arena.h"

using namespace gsplat_tt::sort_ol_arena;

namespace {

int check(const std::vector<Region>& reg, const std::vector<Buf>& bufs, bool expect_fit, const char* tag) {
    const std::vector<uint32_t> p = place(reg, bufs);
    if (p.empty() != !expect_fit) {
        std::printf("%s: fit %d, expected %d\n", tag, !p.empty(), expect_fit);
        return 1;
    }
    if (p.empty()) return 0;
    int bad = 0;
    for (size_t i = 0; i < bufs.size(); ++i) {
        const uint32_t r = region_of(p[i]), o = offset_of(p[i]);
        if (r >= reg.size() || o % kAlign != 0u || o + bufs[i].bytes > reg[r].bytes) {
            if (bad++ < 5) std::printf("%s: buf %u outside region %u (+%u)\n", tag, bufs[i].id, r, o);
        }
        for (size_t j = 0; j < i; ++j) {
            const uint32_t rj = region_of(p[j]), oj = offset_of(p[j]);
            if (r == rj && o < oj + bufs[j].bytes && oj < o + bufs[i].bytes) {
                if (bad++ < 5) std::printf("%s: bufs %u and %u overlap\n", tag, bufs[i].id, bufs[j].id);
            }
        }
    }
    return bad;
}

}  // namespace

int main() {
    int bad = 0;
    // The bh-33 fused program's tails (docs/b4-step1-t544: CB5 123,904, CB6
    // 251,904, CB20 193,536 B) and the solo sort's buffers at R=8, window 1024,
    // OL_PB=8, town (sort_device.cpp build_program_sort_onelaunch, mover 0).
    const std::vector<Region> reg = {{5, 8192, 123904}, {6, 12288, 251904}, {20, 3072, 193536}};
    const uint32_t pg = 64;
    const std::vector<Buf> solo = {
        {16, 3 * 8 * pg}, {17, 3 * 8 * pg}, {18, 3 * 8 * pg}, {19, pg}, {20, 4096}, {21, 4096},
        {22, 2 * 32 * pg}, {23, 4 * 256 * pg}, {24, 16 * 32}, {28, 1024 * 3 * pg}, {29, 1024 * (8 * 32 + 4)},
        {30, 7680}, {10, 4096}, {11, (2 * 120 + 2) * pg}};
    bad += check(reg, solo, false, "solo-569K");  // 593 KB > 569 KB with these tails
    // Finding: the R=8 run ring (CB 13, 266,240 B) is larger than every single
    // tail (CB6's 251,904 B is the largest), so it fits nowhere even with the
    // dead CB21/CB22 (B4a K2 done; CB_SLAB 6144 x 32 B) added; R=7 (233,472 B) fits.
    std::vector<Region> more = reg;
    more.push_back({21, 0, (2 * 6144 + 256) * 4});
    more.push_back({22, 0, 6144 * 32});
    bad += check(more, solo, false, "solo-R8+cb21+cb22");
    std::vector<Buf> r7 = solo;
    for (auto& b : r7) if (b.id == 29) b.bytes = 1024 * (7 * 32 + 4);
    bad += check(more, r7, true, "solo-R7+cb21+cb22");
    bad += check({{6, 100, 1 << 20}}, solo, false, "unaligned-skip");
    bad += check({{6, 0, 1 << 20}}, solo, true, "one-region");
    std::printf(bad == 0 ? "PASS\n" : "FAIL (%d)\n", bad);
    return bad == 0 ? 0 : 1;
}
