// Host check of the kernel config buffer sizing (task #258, render/host/kcfg_size.h).
// Standalone:
//
//   tests/unit/run_cpp.sh tests/unit/test_kcfg_size.cpp
//
// The split pfwc (fused writer + split + SFPU cov_cam) needs +24 KB untraced, +32 KB with
// the BRISC reader (the default since t232). The device profiler adds 8 KB, except that the
// split stays at +32 KB (t302: the fused mat+blend CBs overflow L1 at +40). An explicit
// GSPLAT_TT_KCFG_EXTRA_KB still wins, and untraced sizes must not change.
#include <cstdio>

#include "render/host/kcfg_size.h"

namespace {
int failures = 0;
void check(long got, long want, const char* what) {
    if (got != want) {
        std::printf("FAIL %s: got %ld want %ld\n", what, got, want);
        ++failures;
    }
}
}  // namespace

int main() {
    using gsplat_tt::kcfg_extra_kb;
    // Untraced: unchanged from tasks #221 / #232.
    check(kcfg_extra_kb(nullptr, true, true, true, false), 32, "split + BRISC reader, untraced");
    check(kcfg_extra_kb(nullptr, true, false, true, false), 24, "split, untraced");
    check(kcfg_extra_kb(nullptr, false, true, true, false), 8, "covcam only, untraced");
    check(kcfg_extra_kb(nullptr, false, false, false, false), 0, "neither, untraced");
    // Device profiler on: +8 KB; the split is capped at 32 (t302, fused mat+blend CBs).
    check(kcfg_extra_kb(nullptr, true, true, true, true), 32, "split + BRISC reader, profiler");
    check(kcfg_extra_kb(nullptr, true, false, true, true), 32, "split, profiler");
    check(kcfg_extra_kb(nullptr, false, true, true, true), 16, "covcam only, profiler");
    check(kcfg_extra_kb(nullptr, false, false, false, true), 8, "neither, profiler");
    // Explicit override wins either way.
    check(kcfg_extra_kb("40", true, true, true, true), 40, "override 40");
    check(kcfg_extra_kb("0", true, true, true, false), 0, "override 0");
    // Profiler detection.
    check(gsplat_tt::kcfg_profiler_on(nullptr), 0, "profiler unset");
    check(gsplat_tt::kcfg_profiler_on(""), 0, "profiler empty");
    check(gsplat_tt::kcfg_profiler_on("0"), 0, "profiler 0");
    check(gsplat_tt::kcfg_profiler_on("1"), 1, "profiler 1");
    return failures ? 1 : 0;
}
