// Task #643: render/host/fold_sort.h.
//
//   tests/unit/run_cpp.sh tests/unit/test_fold_sort_t643.cpp
//
// Checks: the after-hook enqueues the next view's sort only with the switch
// on, the fold run, the early sort wanted, no verify rerun pending and CQ1;
// the next K2 takes the hook's sort only when it skips its own K2 too; a
// buffer grows from the hook only after a drain, and a new one needs none.
#include <cstdio>

#include "render/host/fold_sort.h"

namespace fs = gsplat_tt::fold_sort;

static int fails = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                     \
        }                                                                \
    } while (0)

int main() {
    CHECK(fs::hook_enqueues(true, true, true, false, true));
    CHECK(!fs::hook_enqueues(false, true, true, false, true));  // default off
    CHECK(!fs::hook_enqueues(true, false, true, false, true));  // fold did not run
    CHECK(!fs::hook_enqueues(true, true, false, false, true));  // no early sort
    CHECK(!fs::hook_enqueues(true, true, true, true, true));    // verify rerun pending
    CHECK(!fs::hook_enqueues(true, true, true, false, false));  // no CQ1

    CHECK(fs::use_hook_sort(true, true));
    CHECK(!fs::use_hook_sort(false, true));  // xview miss / fold cancelled: K2 reruns
    CHECK(!fs::use_hook_sort(true, false));

    CHECK(fs::must_drain(100, 200, true));
    CHECK(!fs::must_drain(200, 200, true));
    CHECK(!fs::must_drain(0, 200, false));
    if (fails == 0) std::printf("ok\n");
    return fails == 0 ? 0 : 1;
}
