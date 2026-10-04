// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #165 (host knob GSPLAT_TT_OL_EMIT_TPACK): record pack of the one-launch
// emit on the 3 TRISCs. Each TRISC packs its third of every segment both movers
// publish (protocol: ../dataflow/sort_ol_tpack.h): the same 32 B records as the
// mover's fast loop (sort_bin_onelaunch.cpp), into the mover's ring. L1 only:
// no LLK, no CB traffic. UNPACK sends the two mailbox addresses to MATH and
// PACK (MATH has no CB interfaces, PACK no read pointers).

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "../dataflow/sort_bin_fp32.h"
#include "../dataflow/sort_ol_tpack.h"

#ifndef OL_RING
#define OL_RING 8u
#endif

namespace {

using namespace sort_ol_tpack;

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t REC_BYTES = 32;
constexpr uint32_t L1_TILE_SIZE = 32u;
constexpr uint32_t R = OL_RING;
constexpr uint32_t LOG2R = R == 2u ? 1u : R == 4u ? 2u : 3u;
static_assert(R == 2u || R == 4u || R == 8u, "OL_RING");

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

inline void l1_fence() { asm volatile("fence" ::: "memory"); }

void pack_chunk(volatile uint32_t* mb, uint32_t me, uint32_t n) {
    const uint32_t lo = me == 0u ? 0u : mb[MB_B1 + me - 1u];
    const uint32_t hi = me == 2u ? n : mb[MB_B1 + me];
    const uint32_t* items = reinterpret_cast<const uint32_t*>(mb[MB_ITEMS]);
    const uint32_t brec = mb[MB_BREC], ring = mb[MB_RING], msk = mb[MB_MSK], sh = mb[MB_SH];
    uint32_t s_c = 0xFFFFFFFFu, ty_c = 0xFFFFFFFFu;
    uint32_t cov0 = 0, cov1 = 0, cov2 = 0, dep = 0, mxb = 0, myb = 0, opr = 0, cgb = 0, myt = 0;
    for (uint32_t i = lo; i < hi; i++) {
        const uint32_t it = items[i];
        const uint32_t s = it & 0xFFu;
        const uint32_t dst = it >> 8;
        if (s != s_c) {
            s_c = s;
            const uint32_t* cp = reinterpret_cast<const uint32_t*>(brec + s * PAGE_BYTES);
            cov0 = cp[0];
            cov1 = cp[1];
            cov2 = cp[2];
            mxb = cp[3];
            myb = cp[4];
            opr = cp[10];
            cgb = cp[11];
            dep = cp[12];
            ty_c = 0xFFFFFFFFu;
        }
        const uint32_t t = dst >> LOG2R;
        const uint32_t kx = (t & msk) * L1_TILE_SIZE;
        uint32_t mx;
        if (!sort_bin_fp32::sub_int32(mxb, kx, &mx)) mx = sub_int_cold(mxb, kx);
        const uint32_t tyi = t >> sh;
        if (tyi != ty_c) {
            ty_c = tyi;
            const uint32_t ky = tyi * L1_TILE_SIZE;
            if (!sort_bin_fp32::sub_int32(myb, ky, &myt)) myt = sub_int_cold(myb, ky);
        }
        auto d = reinterpret_cast<volatile uint32_t*>(ring + dst * REC_BYTES);
        d[0] = cov0;
        d[1] = cov1;
        d[2] = cov2;
        d[3] = dep;
        d[4] = mx;
        d[5] = myt;
        d[6] = opr;
        d[7] = cgb;
    }
}

}  // namespace

void kernel_main() {
    uint32_t me = 0, a0 = 0, a1 = 0;
    UNPACK(({
        me = 0u;
        a0 = get_local_cb_interface(CB_TPK + 16u).fifo_rd_ptr << 4;  // mover 0 (BRISC)
        a1 = get_local_cb_interface(CB_TPK).fifo_rd_ptr << 4;        // mover 1 (NCRISC)
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
    volatile uint32_t* mbs[2] = {reinterpret_cast<volatile uint32_t*>(a0), reinterpret_cast<volatile uint32_t*>(a1)};
    uint32_t expect[2] = {1u, 1u};
    bool fin[2] = {false, false};
    while (!(fin[0] && fin[1])) {
        for (uint32_t m = 0; m < 2u; m++) {
            if (fin[m]) continue;
            volatile uint32_t* mb = mbs[m];
            l1_fence();
            if (mb[MB_MAGIC] != MAGIC || mb[MB_SEQ] != expect[m]) continue;
            l1_fence();
            const uint32_t n = mb[MB_N];
            if (n == TPK_FINAL) {
                fin[m] = true;
            } else {
                pack_chunk(mb, me, n);
                l1_fence();
            }
            mb[MB_DONE + 4u * me] = expect[m];
            expect[m]++;
        }
    }
}
