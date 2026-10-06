// Host check of the GSPLAT_TT_MATCULL_TRISC_FILL default (task #315,
// render/host/matcull_trisc_fill.h): unset = on, =0 = off, and never on without
// the fused cull and the one-launch sort.
//
//   tests/unit/run_cpp.sh tests/unit/test_matcull_trisc_fill.cpp
#include <cstdio>

#include "render/host/matcull_trisc_fill.h"

namespace {
int failures = 0;
void check(bool got, bool want, const char* what) {
    if (got != want) {
        std::printf("FAIL %s: got %d want %d\n", what, got, want);
        ++failures;
    }
}
}  // namespace

int main() {
    using gsplat_tt::matcull_trisc_fill_on;
    check(matcull_trisc_fill_on(nullptr, true, true), true, "unset = on");
    check(matcull_trisc_fill_on("1", true, true), true, "=1 on");
    check(matcull_trisc_fill_on("0", true, true), false, "=0 off");
    check(matcull_trisc_fill_on(nullptr, false, true), false, "needs fused cull");
    check(matcull_trisc_fill_on(nullptr, true, false), false, "needs one-launch sort");
    if (failures == 0) std::printf("ok\n");
    return failures == 0 ? 0 : 1;
}
