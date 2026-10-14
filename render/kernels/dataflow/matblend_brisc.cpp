// SPDX-License-Identifier: Apache-2.0
// Task #280/#285 (GSPLAT_TT_MATBLEND_FUSE=1): fused mat+blend program, BRISC.
// Mat mover 0 body (sort_subchunk_materialize.cpp, MAT_CB_BASE 16), then the
// blend writer body (writer_alpha_blend.cpp). See matblend_ncrisc.cpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "img_pack_u8.h"
#include "sort_bin_fp32.h"
#include "sort_radix_tile_algo.h"
#include "tile_window.h"
#if defined(BLEND_B1) && BLEND_B1
#include "subtile_unit.h"
#endif

#ifndef BLEND_RTA_BASE
#define BLEND_RTA_BASE 32
#endif

namespace matk {
#include "sort_subchunk_materialize.cpp"
}  // namespace matk

#if defined(B4A_K2) && B4A_K2
// Task #542 (B4a): after its mat part, BRISC also runs a segment K2
// (tile_assign_scatter_seg.cpp, mover-only diet path, both movers' ranges in
// turn: runtime args at K2_RTA_BASE and K2_RTA_BASE + K2_NARGS) while it
// serves the blend writer once per pair page. Its scratch is CB TA_CB_OFFSET.
#ifndef K2_RTA_BASE
#define K2_RTA_BASE 48
#endif
#define K2_NARGS 21
#define K2_DIET 1
#define K2_CNT16 1  // task #577: 2 KB count row, not 4 KB (stack ran into .bss under ETH)
void b4a_yield();
#define K2_YIELD() b4a_yield()
#include "pfwc_fuse.h"
namespace k2k {
uint32_t g_rta = K2_RTA_BASE;
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + g_rta);
}
#include "tile_assign_scatter_seg.cpp"
}  // namespace k2k

void (*g_b4a_fn)(void*) = nullptr;
void* g_b4a_ctx = nullptr;
void b4a_yield() {
    if (g_b4a_fn != nullptr) g_b4a_fn(g_b4a_ctx);
}
template <class F>
void b4a_run_k2(F& poll) {
    g_b4a_ctx = &poll;
    g_b4a_fn = [](void* p) { (*static_cast<F*>(p))(); };
    for (uint32_t m = 0; m < 2u; m++) {
        k2k::g_rta = K2_RTA_BASE + m * K2_NARGS;
        if (::get_arg_val<uint32_t>(k2k::g_rta + 7) != 0) k2k::kernel_main();  // nseg != 0
    }
    g_b4a_fn = nullptr;
}
#define BLEND_WRITER_PRE(poll) b4a_run_k2(poll)
#endif

namespace blendk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + BLEND_RTA_BASE);
}
#include "writer_alpha_blend.cpp"
}  // namespace blendk

void kernel_main() {
    matk::kernel_main();
    noc_async_write_barrier();
    blendk::kernel_main();
}
