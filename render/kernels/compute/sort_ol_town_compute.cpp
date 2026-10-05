// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #202 (on by default, GSPLAT_TT_OL_EMIT_TOWN=0 turns it off;
// docs/emit-trisc-own-t200): the one-launch emit's per-record loop on the 3
// TRISCs, tile-owned (protocol: ../dataflow/sort_ol_town.h). TRISC i walks its
// list of every batch of both movers in batch order: the tile's cursor
// read-modify-write, the same 32 B record as the mover's fast loop
// (sort_bin_onelaunch.cpp) into the mover's ring, and a run word to the mover
// when a run is full. L1 only: no LLK, no CB traffic. UNPACK sends the two
// mailbox addresses to MATH and PACK (task #165: MATH has no CB interfaces,
// PACK no read pointers).
//
// GSPLAT_TT_OL_EMIT_PROF=1 (profiling only): cycle and count totals per launch
// as Tracy timestamped data "town_pc", value (index << 32) | total, index
// order as in TP_* below.

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#include "../dataflow/sort_bin_fp32.h"
#include "../dataflow/sort_ol_town.h"

#ifndef OL_RING
#define OL_RING 8u
#endif
#ifndef OL_EMIT_PROF
#define OL_EMIT_PROF 0
#endif

namespace {

using namespace sort_ol_town;

constexpr uint32_t REC_BYTES = 32;
constexpr uint32_t L1_TILE_SIZE = 32u;
constexpr uint32_t R = OL_RING;
static_assert(R >= 2u && R <= 16u && (R & (R - 1u)) == 0u, "OL_RING");

// Counters (OL_EMIT_PROF): wall cycles from kernel start to the end, waiting
// for the first GO, in process() (all of it), of which waiting for a tile's
// previous run to leave L1, of which waiting for queue space, waiting for a
// published batch; counts: records, runs pushed, batches, blendrec page loads.
enum : uint32_t { TP_TOT, TP_GO, TP_PROC, TP_WFL, TP_WQ, TP_RDY, TP_REC, TP_RUN, TP_NB, TP_NG, TP_ME, TP_N };
#if OL_EMIT_PROF
uint32_t g_tp[TP_N];
inline uint32_t tp_now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }
#define TP_T0(v) const uint32_t v = tp_now()
#define TP_ADD(i, t0) g_tp[i] += tp_now() - (t0)
#define TP_CNT(i, n) g_tp[i] += (n)
#else
#define TP_T0(v)
#define TP_ADD(i, t0)
#define TP_CNT(i, n)
#endif

__attribute__((noinline)) uint32_t sub_int_cold(uint32_t abits, uint32_t k) {
    uint32_t out;
    if (!sort_bin_fp32::sub_int(abits, k, &out)) {
        float a;
        __builtin_memcpy(&a, &abits, 4);
        const float r = a - static_cast<float>(k);
        __builtin_memcpy(&out, &r, 4);
    }
    return out;
}

// Orders this RISC's stores and invalidates its L1 read cache (Blackhole: a
// small write-through cache; == invalidate_l1_cache()). Every poll of a word
// another RISC writes fences first.
inline void l1_fence() { asm volatile("fence" ::: "memory"); }
inline void spin_pause() {
    for (uint32_t i = 0; i < 8u; i++) asm volatile("nop");
}

struct Stream {
    volatile uint32_t* h;  // mailbox header
    uint32_t base;         // mailbox address
    uint32_t nb, k, qwp, st;  // st: 0 wait for GO, 1 active, 2 done
    uint32_t qlim;            // qwp may not reach it (consumed count + QCAP, as last read)
};

// Batch slot s of stream m: this TRISC's records, in stream order.
void process(Stream& sm, uint32_t me, uint32_t s) {
    const uint32_t base = sm.base;
    const volatile uint32_t* desc = reinterpret_cast<const volatile uint32_t*>(base + desc_off(s));
    const uint32_t n = desc[me];
    const uint32_t bbase = desc[3];
    volatile uint32_t* h = sm.h;
    const uint32_t cap = h[H_CAP], msk = h[H_MSK], sh = h[H_SH], ring = h[H_RING];
    uint32_t* const cur = reinterpret_cast<uint32_t*>(h[H_CUR]);
    volatile uint32_t* const fl = reinterpret_cast<volatile uint32_t*>(base + FL_OFF);
    uint32_t* const q = reinterpret_cast<uint32_t*>(base + q_off(me));
    volatile uint32_t* const qwp_w = h + H_QWP + 4u * me;
    volatile uint32_t* const qrd_w = h + H_QRD + 4u * me;
    const uint32_t* const list = reinterpret_cast<const uint32_t*>(base + list_off(s, me));
    uint32_t qwp = sm.qwp, qlim = sm.qlim;
    asm volatile("" ::: "memory");  // the list and blendrec pages are complete (READY read, fence)
    uint32_t off_c = 0xFFFFFFFFu, ty_c = 0xFFFFFFFFu;
    uint32_t cov0 = 0, cov1 = 0, cov2 = 0, dep = 0, mxb = 0, myb = 0, opr = 0, cgb = 0, myt = 0;
    for (uint32_t j = 0; j < n; j++) {
        const uint32_t e = list[j];
        const uint32_t t = e >> 16;
        const uint32_t off = e & 0xFFFFu;
        if (off != off_c) {
            off_c = off;
            const uint32_t* cp = reinterpret_cast<const uint32_t*>(bbase + off);
            cov0 = cp[0];
            cov1 = cp[1];
            cov2 = cp[2];
            mxb = cp[3];
            myb = cp[4];
            opr = cp[10];
            cgb = cp[11];
            dep = cp[12];
            ty_c = 0xFFFFFFFFu;
            TP_CNT(TP_NG, 1u);
        }
        const uint32_t c = cur[t];
        cur[t] = c + 1u;
        TP_CNT(TP_REC, 1u);
        if (c >= cap) continue;  // past capacity: dropped, host fails the frame
        const uint32_t ri = c & (R - 1u);
        if (ri == 0u && fl[t] != c) {  // the tile's previous run has not left L1 yet
            TP_T0(t0);
            do {
                spin_pause();
                l1_fence();
            } while (fl[t] != c);
            TP_ADD(TP_WFL, t0);
        }
        const uint32_t kx = (t & msk) * L1_TILE_SIZE;
        uint32_t mx;
        if (!sort_bin_fp32::sub_int32(mxb, kx, &mx)) mx = sub_int_cold(mxb, kx);
        const uint32_t tyi = t >> sh;
        if (tyi != ty_c) {
            ty_c = tyi;
            const uint32_t ky = tyi * L1_TILE_SIZE;
            if (!sort_bin_fp32::sub_int32(myb, ky, &myt)) myt = sub_int_cold(myb, ky);
        }
        auto d = reinterpret_cast<volatile uint32_t*>(ring + (t * R + ri) * REC_BYTES);
        d[0] = cov0;
        d[1] = cov1;
        d[2] = cov2;
        d[3] = dep;
        d[4] = mx;
        d[5] = myt;
        d[6] = opr;
        d[7] = cgb;
        if (ri == R - 1u) {
            if (qwp == qlim) {
                l1_fence();
                qlim = *qrd_w + QCAP;
                if (qwp == qlim) {
                    TP_T0(t0);
                    do {
                        spin_pause();
                        l1_fence();
                    } while ((qlim = *qrd_w + QCAP) == qwp);
                    TP_ADD(TP_WQ, t0);
                }
            }
            q[qwp & (QCAP - 1u)] = run_word(t, c);
            l1_fence();  // the run's records and the word are in L1 before the count
            *qwp_w = ++qwp;
            TP_CNT(TP_RUN, 1u);
        }
    }
    sm.qwp = qwp;
    sm.qlim = qlim;
}

}  // namespace

void kernel_main() {
    uint32_t me = 0, a0 = 0, a1 = 0;
    UNPACK(({
        me = 0u;
        a0 = get_local_cb_interface(CB_TWN + 16u).fifo_rd_ptr << 4;  // mover 0 (BRISC)
        a1 = get_local_cb_interface(CB_TWN).fifo_rd_ptr << 4;        // mover 1 (NCRISC)
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, a0);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, a1);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, a0);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, a1);
    }));
    MATH(({
        me = 1u;
        a0 = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        a1 = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
    }));
    PACK(({
        me = 2u;
        a0 = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
        a1 = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
    }));
#if OL_EMIT_PROF
    for (uint32_t i = 0; i < TP_N; i++) g_tp[i] = 0u;
    g_tp[TP_ME] = me;
    const uint32_t tp_start = tp_now();
    bool tp_go = false;
#endif
    Stream sm[2] = {{reinterpret_cast<volatile uint32_t*>(a0), a0, 0u, 0u, 0u, 0u, QCAP},
                    {reinterpret_cast<volatile uint32_t*>(a1), a1, 0u, 0u, 0u, 0u, QCAP}};
    // Before GO: no zone (the movers' count, prefix and barriers run first).
    for (;;) {
        bool any = false;
        l1_fence();
        for (uint32_t m = 0; m < 2u; m++) {
            Stream& s = sm[m];
            if (s.st != 0u) continue;
            if (s.h[H_GO] != MAGIC || s.h[H_GO2] != MAGIC2) continue;
            l1_fence();
            s.nb = s.h[H_NB];
            s.st = 1u;
            any = true;
        }
        if (sm[0].st != 0u || sm[1].st != 0u) break;
        if (!any) spin_pause();
    }
#if OL_EMIT_PROF
    g_tp[TP_GO] = tp_now() - tp_start;
    tp_go = true;
#endif
    {
        DeviceZoneScopedN("sort_ol_town");
        uint32_t idle_t0 = 0;
        bool idle = false;
        while (sm[0].st != 2u || sm[1].st != 2u) {
            bool any = false;
            l1_fence();
            for (uint32_t m = 0; m < 2u; m++) {
                Stream& s = sm[m];
                if (s.st == 0u) {
                    if (s.h[H_GO] != MAGIC || s.h[H_GO2] != MAGIC2) continue;
                    l1_fence();
                    s.nb = s.h[H_NB];
                    s.st = 1u;
                }
                if (s.st != 1u) continue;
                if (s.k < s.nb) {
                    if (s.h[H_READY] <= s.k) continue;
                    l1_fence();
#if OL_EMIT_PROF
                    if (idle) {
                        g_tp[TP_RDY] += tp_now() - idle_t0;
                        idle = false;
                    }
#endif
                    TP_T0(t0);
                    process(s, me, s.k % SLOTS);
                    TP_ADD(TP_PROC, t0);
                    TP_CNT(TP_NB, 1u);
                    l1_fence();  // records, cursors and run words before the batch count
                    s.h[H_DONE + 4u * me] = ++s.k;
                    any = true;
                }
                if (s.k >= s.nb) {
                    l1_fence();
                    s.h[H_FIN + me] = MAGIC;
                    s.st = 2u;
                    any = true;
                }
            }
            if (!any) {
#if OL_EMIT_PROF
                if (!idle) {
                    idle_t0 = tp_now();
                    idle = true;
                }
#endif
                spin_pause();
            }
        }
        (void)idle_t0;
        (void)idle;
    }
#if OL_EMIT_PROF
    (void)tp_go;
    g_tp[TP_TOT] = tp_now() - tp_start;
    // One marker per counter, (index << 32) | value: one call site keeps the binary small.
    for (uint32_t i = 0; i < TP_N; i++) DeviceTimestampedData("town_pc", (uint64_t(i) << 32) | g_tp[i]);
#endif
}
