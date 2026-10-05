// SPDX-License-Identifier: Apache-2.0
//
// Out-of-line definition of the per-stage accumulator. See stage_timers.h.

#include "stage_timers.h"

#include <cstdlib>

namespace gsplat_tt::stagetimers {

Acc& acc() {
    static Acc a;
    return a;
}

void reset() { acc() = Acc{}; }

bool split_blend() {
    static const bool on = [] {
        const char* e = std::getenv("GSPLAT_TT_SPLIT_BLEND");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

}  // namespace gsplat_tt::stagetimers
