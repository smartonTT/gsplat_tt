// Host checks for task #358 (per-board mover speed tables). Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_mover_table_t358.cpp
//
// - auto picks the p100a table on every board, p150 cards (7, 8, 9, 13)
//   included: the p150 table measured slower on bh-30 (#358);
// - GSPLAT_TT_MOVER_TABLE parsing: unset/""/auto = detected, p100a, p150,
//   anything else = detected and not ok;
// - both tables cover the 110 sort cores (physical x 1-6, 11-15, y 2-11) once
//   each with non-zero speeds, and mover_speeds returns the chosen table's row.
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include "render/host/sort_mover_speed.h"

namespace {

using gsplat_tt::sort_split::MoverBoard;
using gsplat_tt::sort_split::MoverSpeed;
int failures = 0;
void fail(const char* what, long a, long b) {
    if (failures++ < 20) std::fprintf(stderr, "FAIL %s: %ld %ld\n", what, a, b);
}

template <std::size_t N>
void check_table(const MoverSpeed (&tab)[N], const char* name) {
    if (N != 110u) fail(name, static_cast<long>(N), 110);
    std::set<uint32_t> seen;
    for (const MoverSpeed& m : tab) {
        const bool x_ok = (m.x >= 1 && m.x <= 6) || (m.x >= 11 && m.x <= 15);
        if (!x_ok || m.y < 2 || m.y > 11) fail(name, m.x, m.y);
        if (!seen.insert(m.x | (m.y << 16)).second) fail("duplicate core", m.x, m.y);
        if (m.brisc == 0 || m.ncrisc == 0) fail("zero speed", m.x, m.y);
    }
}

}  // namespace

int main() {
    namespace ss = gsplat_tt::sort_split;
    for (int ct : {0, 1, 6, 7, 8, 9, 11, 13, 14, 16, 18})
        if (ss::mover_board_for_cluster(ct) != MoverBoard::P100a) fail("p100a cluster", ct, 0);

    bool ok = false;
    for (const MoverBoard d : {MoverBoard::P100a, MoverBoard::P150}) {
        if (ss::mover_board_from_env(nullptr, d, &ok) != d || !ok) fail("env unset", 0, 0);
        if (ss::mover_board_from_env("", d, &ok) != d || !ok) fail("env empty", 0, 0);
        if (ss::mover_board_from_env("auto", d, &ok) != d || !ok) fail("env auto", 0, 0);
        if (ss::mover_board_from_env("p100a", d, &ok) != MoverBoard::P100a || !ok) fail("p100a", 0, 0);
        if (ss::mover_board_from_env("p150", d, &ok) != MoverBoard::P150 || !ok) fail("p150", 0, 0);
        if (ss::mover_board_from_env("p300", d, &ok) != d || ok) fail("bad env", 0, 0);
    }

    check_table(ss::kMoverSpeedP100a, "p100a table");
    check_table(ss::kMoverSpeedP150, "p150 table");

    // Core (13, 3): p100a {913, 1026}, p150 {716, 738}; a core outside both = 1000.
    const std::vector<uint32_t> xy = {13u | (3u << 16), 7u | (2u << 16)};
    const auto a = ss::mover_speeds(xy, MoverBoard::P100a);
    const auto b = ss::mover_speeds(xy, MoverBoard::P150);
    if (a[0] != 913 || a[1] != 1026) fail("p100a row", a[0], a[1]);
    if (b[0] != 716 || b[1] != 738) fail("p150 row", b[0], b[1]);
    if (a[2] != 1000 || a[3] != 1000 || b[2] != 1000 || b[3] != 1000) fail("missing core", a[2], b[2]);
    std::printf("%s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
