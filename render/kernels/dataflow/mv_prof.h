// SPDX-License-Identifier: Apache-2.0
// Task #553 (GSPLAT_TT_MV_PROF=1, profiler builds only): BRISC/NCRISC (mover) cycle counters in
// the fused mat+blend program, the mover side of T2_PROF (alpha_blend_compute_mb.cpp). Reported
// once per launch as "mv_pc" timestamped data, value = (index << 32) | cycles (or count).
//   TOT  kernel start -> blend end; PRE: kernel start -> blend start (the mat phase)
//   WCB  waits on a CB for compute or the other mover (reader: bulk ring / MB counts / tile queue
//        space; writer: tile queue / colour tiles): the RISC idles, nothing of its own in flight
//   WNOC small-read / atomic / claim / flush latency waits (NoC round trips)
//   WBULK bulk slab read barrier (the reader's real data movement)
//   WRDY reader poll for the mat movers' ready flag
//   L1K/L4K/L16K: cycles of WCB waits >= 1K/4K/16K cycles (idle long enough for foreign work)
//   NW   number of waits
// Busy = TOT - PRE - all waits. Off: every macro is empty.
#pragma once
#include <cstdint>

#if defined(MV_PROF) && MV_PROF && defined(PROFILE_KERNEL)
enum : uint32_t { MV_TOT, MV_PRE, MV_WCB, MV_WNOC, MV_WBULK, MV_WRDY, MV_L1K, MV_L4K, MV_L16K, MV_NW, MV_N };
uint32_t g_mv[MV_N];
uint32_t g_mv_t0;
inline uint32_t mv_now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }
inline void mv_wait(uint32_t i, uint32_t t0) {
    const uint32_t d = mv_now() - t0;
    g_mv[i] += d;
    ++g_mv[MV_NW];
    if (i == MV_WCB) {
        if (d >= 1024u) g_mv[MV_L1K] += d;
        if (d >= 4096u) g_mv[MV_L4K] += d;
        if (d >= 16384u) g_mv[MV_L16K] += d;
    }
}
#define MV_BEGIN() (g_mv_t0 = mv_now())
#define MV_MARK_PRE() (g_mv[MV_PRE] = mv_now() - g_mv_t0)
#define MV_T0(v) const uint32_t v = mv_now()
#define MV_W(i, t0) mv_wait(i, t0)
#define MV_END()                                                                          \
    do {                                                                                  \
        g_mv[MV_TOT] = mv_now() - g_mv_t0;                                                \
        for (uint32_t i = 0; i < MV_N; i++) DeviceTimestampedData("mv_pc", (uint64_t(i) << 32) | g_mv[i]); \
    } while (0)
// Wrap one blocking call: MV_WRAP(MV_WCB, cb_reserve_back(cb, n));
#define MV_WRAP(i, call) \
    do {                 \
        MV_T0(mvt_);     \
        call;            \
        MV_W(i, mvt_);   \
    } while (0)
#else
#define MV_BEGIN() ((void)0)
#define MV_MARK_PRE() ((void)0)
#define MV_T0(v)
#define MV_W(i, t0) ((void)0)
#define MV_END() ((void)0)
#define MV_WRAP(i, call) call
#endif
