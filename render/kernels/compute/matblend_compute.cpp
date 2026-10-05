// SPDX-License-Identifier: Apache-2.0
// Task #280/#285 (GSPLAT_TT_MATBLEND_FUSE=1): fused mat+blend program, TRISCs.
// The mat cull compute body (mat_cull_compute.cpp, runtime args 0-2) serves
// both mat movers until both streams end, then the blend compute body
// (alpha_blend_compute_mb.cpp, runtime args from BLEND_RTA_BASE, CB ids from
// BLEND_CB_BASE) runs. See dataflow/matblend_ncrisc.cpp.
#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "../dataflow/dm_fp32.h"
#include "blend_t_live.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

#if defined(GSPLAT_TT_MB_STATS) || defined(GSPLAT_TT_MB_TILECYC)
#include "api/debug/dprint.h"
#endif

#ifndef BLEND_RTA_BASE
#define BLEND_RTA_BASE 4
#endif

namespace matk {
#include "mat_cull_compute.cpp"
}  // namespace matk

namespace blendk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + BLEND_RTA_BASE);
}
#include "alpha_blend_compute_mb.cpp"
}  // namespace blendk

void kernel_main() {
    matk::kernel_main();
    blendk::kernel_main();
}
