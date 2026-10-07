// Task #306/#315: GSPLAT_TT_MATCULL_TRISC_FILL policy. On by default since #315
// (iter 207); =0 turns it off. Needs the fused cull and the one-launch sort.
// Header-only so tests/unit/test_matcull_trisc_fill.cpp can check it without tt-metal.
#pragma once

namespace gsplat_tt {

inline bool matcull_trisc_fill_on(const char* env, bool fused_cull, bool onelaunch) {
    return !(env != nullptr && env[0] == '0') && fused_cull && onelaunch;
}

}  // namespace gsplat_tt
