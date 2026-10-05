// SPDX-License-Identifier: Apache-2.0
// Task #280/#285 (GSPLAT_TT_MATBLEND_FUSE=1): fused mat+blend program, NCRISC.
// Runs the mat mover 1 body (sort_subchunk_materialize.cpp, MAT_CB_BASE 0) and
// then the blend reader body (reader_alpha_blend_mb_devcull.cpp) on the same
// RISC, so a core's blend starts as soon as its own mat items are done; the
// blend reader waits per (tile, subchunk) on the ready flags the mat movers
// write. Blend CB ids / compile-time args / runtime args are shifted by
// BLEND_CB_BASE / BLEND_CTA_BASE / BLEND_RTA_BASE (host: matblend_fuse.h).
// Every header either body uses is included here first at global scope (all
// are #pragma once), so the namespace-wrapped bodies only add their own code.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"
#include "sort_radix_tile_algo.h"

#ifndef BLEND_RTA_BASE
#define BLEND_RTA_BASE 32
#endif

namespace matk {
#include "sort_subchunk_materialize.cpp"
}  // namespace matk

namespace blendk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + BLEND_RTA_BASE);
}
#include "reader_alpha_blend_mb_devcull.cpp"
}  // namespace blendk

void kernel_main() {
    matk::kernel_main();
    noc_async_write_barrier();
    blendk::kernel_main();
}
