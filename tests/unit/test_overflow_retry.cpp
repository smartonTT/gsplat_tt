// Task #270: the tile-overflow floor ladder (render/host/overflow_retry.h).
//
//   tests/unit/run_cpp.sh tests/unit/test_overflow_retry.cpp
//
// Checks: every step at least doubles the floor and never passes kMaxFloor;
// a small overflow takes a small step (the t257 viewer case, tile 370 with
// 34672 records at 1/16384, retries at a floor still finer than 1/1024); any
// overflow 10x the cap reaches kMaxFloor within kMaxRetries steps.
#include <cstdio>

#include "render/host/overflow_retry.h"

namespace orr = gsplat_tt::overflow_retry;

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

int main() {
    constexpr uint32_t cap = 32768;
    const float floors[] = {1.0f / 16384, 1.0f / 1024, 1.0f / 255, 1.0f / 16, 0.4f};
    const uint32_t ns[] = {cap + 1, 34672, 40000, 2 * cap, 100 * cap};
    for (float f : floors) {
        for (uint32_t n : ns) {
            const float g = orr::next_floor(f, n, cap);
            CHECK(g <= orr::kMaxFloor);
            CHECK(g >= 2.0f * f || g == orr::kMaxFloor);
        }
    }
    // t257: one step from 1/16384 lands between 1/8192 and 1/1024.
    const float t257 = orr::next_floor(1.0f / 16384, 34672, cap);
    CHECK(t257 >= 1.0f / 8192 && t257 < 1.0f / 1024);
    // Bench floor, 10% over: stays finer than 1/32.
    CHECK(orr::next_floor(1.0f / 255, 36045, cap) < 1.0f / 32);
    // The ladder ends: from any floor and a huge overflow, kMaxFloor within
    // kMaxRetries steps.
    for (float f : {1.0f / 1048576, 1.0f / 16384, 1.0f / 255}) {
        float g = f;
        for (int i = 0; i < orr::kMaxRetries; ++i) g = orr::next_floor(g, 10 * cap, cap);
        CHECK(g == orr::kMaxFloor);
    }
    if (fails == 0) std::printf("ok\n");
    return fails == 0 ? 0 : 1;
}
