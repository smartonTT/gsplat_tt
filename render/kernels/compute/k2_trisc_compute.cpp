// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #274 (GSPLAT_TT_K2_TRISC=1): the segment K2's pair loop on the TRISCs
// (protocol: ../dataflow/k2_trisc.h). TRISC 0 runs BRISC's job 0, TRISC 2
// NCRISC's job 0, TRISC 1 both jobs 1. L1 only: no LLK, no CB traffic. UNPACK
// sends the two job CB addresses to MATH and PACK (MATH has no CB interfaces).
#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "../dataflow/pfwc_fuse.h"
#include "../dataflow/k2_trisc.h"

namespace {
using namespace k2_trisc;

// Orders this RISC's stores and invalidates its L1 read cache (Blackhole).
inline void l1_fence() { asm volatile("fence" ::: "memory"); }

#ifndef K2_PROF
#define K2_PROF 0
#endif
inline uint32_t now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }

void run_job(uint32_t w) {
    volatile uint32_t* h = reinterpret_cast<volatile uint32_t*>(w);
    for (;;) {
        l1_fence();
        if (h[H_GO] == MAGIC && h[H_GO2] == MAGIC2) break;
        for (uint32_t i = 0; i < 8u; i++) asm volatile("nop");
    }
    l1_fence();
#if K2_PROF
    const uint32_t ts = now();
#endif
    const uint32_t pg0 = h[H_PG0], npg = h[H_NPG], c = h[H_C], lo = h[H_LO];
    const uint32_t n_in = h[H_NIN], P_pub = h[H_PPUB], tiles_x = h[H_TX], nseg = h[H_NSEG];
    const auto tab = reinterpret_cast<const volatile uint32_t*>(h[H_TAB]);
    const uint32_t span = h[H_SPAN];
    uint32_t* cnt = reinterpret_cast<uint32_t*>(w + CNT_OFF);
    for (uint32_t t = 0; t < span; t++) cnt[t] = 0;
    if (npg != 0) {
        DeviceZoneScopedN("k2_trisc");
        const uint32_t* lofs = reinterpret_cast<const uint32_t*>(w + LOFS_OFF);
        pfwc_fuse::WinIo io{lofs, reinterpret_cast<const uint32_t*>(w + BOX_OFF), lofs, n_in,
                            reinterpret_cast<uint32_t*>(w + GID_OFF),
                            reinterpret_cast<uint32_t*>(w + TID_OFF)};
        pfwc_fuse::emit_pairs_diet_from<true>(tab, nseg, P_pub, tiles_x, pg0, npg, c, lo, io, cnt);
    }
#if K2_PROF
    h[H_TS] = ts;
    h[H_TE] = now();
#endif
    l1_fence();  // pair pages and counts are in L1 before DONE
    h[H_DONE] = MAGIC;
}

}  // namespace

void kernel_main() {
    uint32_t ab = 0, an = 0;
    UNPACK(({
        ab = get_local_cb_interface(CB_JOB + 16u).fifo_rd_ptr << 4;  // mover 0 (BRISC)
        an = get_local_cb_interface(CB_JOB).fifo_rd_ptr << 4;        // mover 1 (NCRISC)
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, ab);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, an);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, ab);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, an);
        run_job(job_addr(ab, 0));
    }));
    MATH(({
        ab = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        an = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        run_job(job_addr(ab, 1));
        run_job(job_addr(an, 1));
    }));
    PACK(({
        ab = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        an = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        run_job(job_addr(an, 0));
    }));
}
