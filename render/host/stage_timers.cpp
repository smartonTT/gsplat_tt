// SPDX-License-Identifier: Apache-2.0
//
// Out-of-line definition of the per-stage accumulator. See stage_timers.h.

#include "stage_timers.h"

namespace gsplat_tt::stagetimers {

Acc& acc() {
    static Acc a;
    return a;
}

void reset() { acc() = Acc{}; }

}  // namespace gsplat_tt::stagetimers
