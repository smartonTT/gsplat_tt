// Task #213: render/host/pair_guard.h.
//
//   tests/unit/run_cpp.sh tests/unit/test_pair_guard.cpp
//
// Checks: drain_on_throw drains exactly once and rethrows the same exception
// when f throws, and never drains when f returns; pair_alloc_bytes keeps the
// static ceiling by default, uses a nonzero test cap instead (whole 1024-pair
// pages), and never allocates less than asked for.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "render/host/pair_guard.h"

namespace pg = gsplat_tt::pair_guard;

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

int main() {
    int drains = 0;
    auto drain = [&] { ++drains; };
    CHECK(pg::drain_on_throw([] { return true; }, drain));
    CHECK(drains == 0);
    bool caught = false;
    try {
        (void)pg::drain_on_throw([]() -> bool { throw std::runtime_error("enqueue"); }, drain);
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "enqueue";
    }
    CHECK(caught);
    CHECK(drains == 1);

    constexpr uint32_t ceil = 4718592u;
    CHECK(pg::pair_alloc_bytes(0, ceil, 0) == std::size_t{ceil} * 4);
    CHECK(pg::pair_alloc_bytes(std::size_t{5000000} * 4, ceil, 0) == std::size_t{5000000} * 4);
    CHECK(pg::pair_alloc_bytes(0, ceil, 1000000) == std::size_t{1000448} * 4);
    CHECK(pg::pair_alloc_bytes(std::size_t{3000000} * 4, ceil, 1000000) == std::size_t{3000000} * 4);
    CHECK(pg::pair_alloc_bytes(0, ceil, 1) == 4096);
    if (fails) std::fprintf(stderr, "%d failure(s)\n", fails);
    return fails ? 1 : 0;
}
