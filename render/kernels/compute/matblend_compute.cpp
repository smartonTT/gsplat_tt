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
#include "../blend_alias_fifo.h"

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

#if defined(SORT_IN_BLEND) && SORT_IN_BLEND
// Task #558 (GSPLAT_TT_SORT_IN_BLEND=1): PACK runs the town of BRISC's solo
// sort(N+1) (sort_ol_town_compute.cpp, OL_TOWN_EMBED) while it waits for MATH,
// then drains it after the blend. Compute arg SIB_TOWN_RTA: the mailbox's
// placement (cb << 24 | byte offset in that CB's storage), 0 = no sort.
#define OL_TOWN_EMBED 1
namespace olt {
#include "sort_ol_town_compute.cpp"
}  // namespace olt
// PACK only (blend_t_readback calls it on every thread). No tensix_sync: a
// stale nonzero read only ends the poll early; tile_regs_wait still waits.
inline void sib_mathwait() {
    PACK(({
        while (ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) == 0) {
            if (olt::town_poll()) break;
        }
    }));
}
#define T2_MATHWAIT() sib_mathwait()
#endif

namespace blendk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + BLEND_RTA_BASE);
}
#include "alpha_blend_compute_mb.cpp"
}  // namespace blendk

void kernel_main() {
#if defined(SORT_IN_BLEND) && SORT_IN_BLEND
    PACK(({
        const uint32_t tw = ::get_arg_val<uint32_t>(SIB_TOWN_RTA);
        const auto& c = get_local_cb_interface(tw >> 24);
        olt::town_init(tw == 0u ? 0u : ((c.fifo_limit - c.fifo_size) << cb_addr_shift) + (tw & 0xFFFFFFu));
    }));
#endif
    matk::kernel_main();
#ifdef BLEND_ALIAS_CUT
    // Every TRISC keeps its own copy: unpack pops CB_MB_COUNTS, pack fills CB_OUT.
    blend_fifo_shrink(BLEND_CB_BASE + 3, BLEND_MB_COUNTS_BYTES, BLEND_MB_COUNTS_PAGE);
    blend_fifo_shrink(BLEND_CB_BASE + 16, BLEND_OUT_BYTES, BLEND_OUT_PAGE);
#endif
    blendk::kernel_main();
#if defined(SORT_IN_BLEND) && SORT_IN_BLEND
    PACK(({ olt::town_drain(); }));
#endif
}
