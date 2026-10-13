// SPDX-License-Identifier: Apache-2.0
// Task #280/#285 (GSPLAT_TT_MATBLEND_FUSE=1): fused mat+blend program, BRISC.
// Mat mover 0 body (sort_subchunk_materialize.cpp, MAT_CB_BASE 16), then the
// blend writer body (writer_alpha_blend.cpp). See matblend_ncrisc.cpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "img_pack_u8.h"
#include "sort_bin_fp32.h"
#include "sort_radix_tile_algo.h"
// Global, before matk: its #pragma once would otherwise put sort_ol in matk (#565).
#include "sort_onelaunch_algo.h"
#include "../blend_alias_fifo.h"
#include "mv_prof.h"

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

#if defined(SORT_IN_BLEND) && SORT_IN_BLEND
// Task #558 (GSPLAT_TT_SORT_IN_BLEND=1): after K2(N+1), BRISC runs the solo
// sort(N+1) (sort_bin_onelaunch.cpp, OL_SOLO, OL_ARENA) in the same writer
// pre-phase; every sort wait (OL_YIELD) serves the blend writer once. Its
// scratch lives in the dead tails of the mat CBs (host-placed) and its town
// runs on PACK (matblend_compute.cpp). A global barrier first: every core's
// mat(N) (NCRISC and BRISC) and K2(N+1) are done before any sort(N+1) write.
// Runtime args at SIB_RTA_BASE: 0 run, 1 arrive sem (core 0), 2 release sem,
// 3 cores, 4-5 core 0 NoC x/y, 6-19 arena placements (cb << 24 | byte offset)
// of kSibIds, 20-49 the sort's args (core 0: 50- the NoC xy list).
#include "sort_onelaunch_algo.h"
// Task #565: the sort part at -Os. At -O3 it put the fused program at 122640 B,
// needing a +51 KB kernel config buffer, where the static CBs fit only up to +34 KB.
#pragma GCC push_options
#pragma GCC optimize("Os")
#include "sort_ol_town.h"
#define OL_YIELD() b4a_yield()
namespace olk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + SIB_RTA_BASE + 20);
}
#include "sort_bin_onelaunch.cpp"
}  // namespace olk

constexpr uint32_t kSibIds[14] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 30, 10, 11};
template <class F>
void sib_run(F& poll) {
    constexpr uint32_t A = SIB_RTA_BASE;
    if (::get_arg_val<uint32_t>(A) == 0u) return;
    for (uint32_t i = 0; i < 14u; i++) {
        const uint32_t pk = ::get_arg_val<uint32_t>(A + 6u + i);
        const auto& c = get_local_cb_interface(pk >> 24);
        olk::g_ol_arena[kSibIds[i]] = ((c.fifo_limit - c.fifo_size) << cb_addr_shift) + (pk & 0xFFFFFFu);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    const uint32_t arrive_id = ::get_arg_val<uint32_t>(A + 1u), release_id = ::get_arg_val<uint32_t>(A + 2u);
    const uint32_t nc = ::get_arg_val<uint32_t>(A + 3u);
    noc_semaphore_inc(get_noc_addr(::get_arg_val<uint32_t>(A + 4u), ::get_arg_val<uint32_t>(A + 5u),
                                   get_semaphore(arrive_id)), 1u);
    if (::get_arg_val<uint32_t>(A + 20u + 14u) == 0u) {  // sort arg 14: core id
        auto arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_id));
        while (*arrive < 2u * nc) {
            poll();
            invalidate_l1_cache();
        }
        for (uint32_t c = 0; c < nc; c++) {
            const uint32_t xy = ::get_arg_val<uint32_t>(A + 50u + c);
            noc_semaphore_inc(get_noc_addr(xy & 0xFFFFu, xy >> 16, get_semaphore(release_id)), 1u);
        }
        noc_async_atomic_barrier();
    }
    auto release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_id));
    while (*release == 0u) {
        poll();
        invalidate_l1_cache();
    }
#if defined(SIB_DBG) && SIB_DBG == 4u
    {  // t571 debug: 4 = PACK town handshake only (NB = 0 mailbox, no sort body)
        namespace tw = sort_ol_town;
        volatile uint32_t* const twh = reinterpret_cast<volatile uint32_t*>(olk::g_ol_arena[tw::CB_TWN + 16u]);
        twh[tw::H_NB] = 0u;
        twh[tw::H_READY] = 0u;
        for (uint32_t i = 0; i < tw::NT; i++) twh[tw::H_FIN + i] = 0u;
        asm volatile("fence" ::: "memory");
        twh[tw::H_GO2] = tw::MAGIC2;
        asm volatile("fence" ::: "memory");
        twh[tw::H_GO] = tw::MAGIC;
        for (;;) {
            invalidate_l1_cache();
            if (twh[tw::H_FIN] == tw::MAGIC) break;
            poll();
        }
        twh[tw::H_GO] = 0u;
        twh[tw::H_GO2] = 0u;
        twh[tw::H_FIN] = 0u;
        asm volatile("fence" ::: "memory");
    }
#elif defined(SIB_DBG) && SIB_DBG >= 5u && SIB_DBG <= 9u
    // t571 debug: 5 = the arm-4 mailbox writes, PACK town off (no handshake, no FIN wait);
    // 6 = the same writes at CB21's base (BRISC's dead mat CB_BSORT), 7 = at CB5's base + 8192
    // (the start of the CB5 arena region past the blend alias cut); t573: 8 = arm 5 compiled in
    // but skipped at runtime (code size only), 9 = arm 5 with K2(N+1) skipped (b4a_run_k2)
    if (SIB_DBG != 8u || ::get_arg_val<uint32_t>(A + 3u) == 0xD0D0u) {  // nc never 0xD0D0
        namespace tw = sort_ol_town;
        auto cb_base = [](uint32_t cb) {
            const auto& c = get_local_cb_interface(cb);
            return (c.fifo_limit - c.fifo_size) << cb_addr_shift;
        };
        const uint32_t twa = SIB_DBG == 6u ? cb_base(21) : SIB_DBG == 7u ? cb_base(5) + 8192u
                                                                          : olk::g_ol_arena[tw::CB_TWN + 16u];
        volatile uint32_t* const twh = reinterpret_cast<volatile uint32_t*>(twa);
        twh[tw::H_NB] = 0u;
        twh[tw::H_READY] = 0u;
        for (uint32_t i = 0; i < tw::NT; i++) twh[tw::H_FIN + i] = 0u;
        asm volatile("fence" ::: "memory");
        twh[tw::H_GO2] = tw::MAGIC2;
        twh[tw::H_GO] = tw::MAGIC;
        asm volatile("fence" ::: "memory");
        twh[tw::H_GO] = 0u;
        twh[tw::H_GO2] = 0u;
        asm volatile("fence" ::: "memory");
    }
#elif !defined(SIB_DBG) || SIB_DBG != 1u  // t568 debug: 1 = barrier only
    olk::kernel_main();
#endif
    noc_async_write_barrier();
}
#pragma GCC pop_options
#endif
template <class F>
void b4a_run_k2(F& poll) {
    g_b4a_ctx = &poll;
    g_b4a_fn = [](void* p) { (*static_cast<F*>(p))(); };
    for (uint32_t m = 0; m < 2u; m++) {
#if defined(SORT_IN_BLEND) && SORT_IN_BLEND && defined(SIB_DBG) && SIB_DBG == 9u
        break;  // t573 debug arm 9: no K2(N+1) in this blend
#endif
        k2k::g_rta = K2_RTA_BASE + m * K2_NARGS;
        if (::get_arg_val<uint32_t>(k2k::g_rta + 7) != 0) k2k::kernel_main();  // nseg != 0
    }
#if defined(SORT_IN_BLEND) && SORT_IN_BLEND
    sib_run(poll);
#endif
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
    MV_BEGIN();  // task #553: GSPLAT_TT_MV_PROF=1 mover counters (profiler builds)
    matk::kernel_main();
    noc_async_write_barrier();
    MV_MARK_PRE();
#ifdef BLEND_ALIAS_CUT
    blend_fifo_shrink(BLEND_CB_BASE + 16, BLEND_OUT_BYTES, BLEND_OUT_PAGE);        // writer pops
    blend_fifo_shrink(BLEND_CB_BASE + 8, BLEND_IMG_U8_BYTES, BLEND_IMG_U8_BYTES);  // writer scratch
#endif
    blendk::kernel_main();
    MV_END();
}
