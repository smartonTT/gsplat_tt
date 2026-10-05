// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Task #90: the band-extent SFPU microblock cull, fused into the
// sort_subchunk_mat program.
//
// Both data movers run sort_subchunk_materialize. Each one depth-sorts its
// records into an L1 slab, transposes every 128-record batch into an fp32
// coefficient tile (the layout reader_tile_l1_cull.cpp used) and pushes it on
// ITS OWN stream: CB_COEFF_S0 / CB_KEEP_S0 for NCRISC, CB_COEFF_S1 /
// CB_KEEP_S1 for BRISC. This kernel serves both streams in arrival order, so
// a mover that is reading or radix-sorting never holds up the other one, and
// the SFPU runs while the movers work instead of in a separate program.
//
// UNPACK polls the two coefficient CBs and picks the next stream; MATH and
// PACK learn it through the TRISC mailboxes (as get_tile_address does). The
// DEST is in full-sync mode, so UNPACK is at most one batch ahead of PACK and a
// mailbox never holds more than two messages. A coefficient tile whose word
// END_WORD is non-zero carries no data and ends that stream; the loop ends when
// both streams have ended.

#define BAND_CULL_NO_MAIN
#include "microblock_band_cull_compute.cpp"

namespace {

constexpr uint32_t CB_COEFF_S0 = 8;    // NCRISC (MAT_CB_BASE 0) coefficient tiles
constexpr uint32_t CB_KEEP_S0 = 9;     // -> NCRISC mask tiles
constexpr uint32_t CB_COEFF_S1 = 24;   // BRISC (MAT_CB_BASE 16)
constexpr uint32_t CB_KEEP_S1 = 25;
constexpr uint32_t END_WORD = 1023u;   // == sort_subchunk_materialize.cpp
constexpr uint32_t MSG_DONE = 2u;

#ifdef TRISC_UNPACK
inline bool coeff_ready(uint32_t cb) {
    volatile tt_l1_ptr uint32_t* rp = get_cb_tiles_received_ptr(cb);
    const uint16_t recv = static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(rp)));
    return static_cast<uint16_t>(recv - get_local_cb_interface(cb).tiles_acked) != 0u;
}

// Next stream with a data tile at its front (0 or 1), or MSG_DONE once both
// streams have ended. End tiles are popped here. `prefer` alternates the
// streams when both have data.
inline uint32_t pick_stream(uint32_t& live, uint32_t& prefer) {
    while (live != 0u) {
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t s = prefer ^ i;
            if ((live & (1u << s)) == 0u) continue;
            const uint32_t cb = s ? CB_COEFF_S1 : CB_COEFF_S0;
            if (!coeff_ready(cb)) continue;
            asm volatile("fence" ::: "memory");
            const uint32_t tile = get_local_cb_interface(cb).fifo_rd_ptr << 4;
            if (reinterpret_cast<volatile uint32_t*>(tile)[END_WORD] != 0u) {
                llk_pop_tiles(cb, 1);
                live &= ~(1u << s);
                continue;
            }
            prefer = s ^ 1u;
            return s;
        }
    }
    return MSG_DONE;
}
#endif

// Task #297: profiler builds only (PROFILE_KERNEL): per-core cycle split of the
// mat-phase cull, reported at the end as "mc_*" timestamped-data markers
// (opt/profiler/postl1_cores.py). UNPACK: mc_uw = spin in pick_stream (waiting for
// a mover's coefficient tile). MATH: mc_mw = wait in mailbox_read, mc_act = copy +
// SFPU + commit per batch, mc_sf = band_batch alone, mc_nb = batches, mc_tot = all.
#if defined(PROFILE_KERNEL)
#define MC_PROF 1
inline uint32_t mc_now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }
#else
#define MC_PROF 0
#endif

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("mat_cull_mask");
    const uint32_t floor_bits = get_arg_val<uint32_t>(0);
    const bool cull_disabled = get_arg_val<uint32_t>(1) != 0;
    const uint32_t live_init = get_arg_val<uint32_t>(2);  // bit s: stream s sends tiles
    float floor_f;
    __builtin_memcpy(&floor_f, &floor_bits, 4);
    const float inv_floor_f = (floor_f > 0.0f) ? (1.0f / floor_f) : 0.0f;
    uint32_t inv_floor_bits;
    __builtin_memcpy(&inv_floor_bits, &inv_floor_f, 4);
    (void)inv_floor_bits;
    (void)cull_disabled;

    init_sfpu(CB_COEFF_S0, CB_KEEP_S0);
    uint32_t live = live_init;
    uint32_t prefer = 0;
    (void)live;
    (void)prefer;
#if MC_PROF
    uint32_t mc_uw = 0, mc_mw = 0, mc_act = 0, mc_sf = 0, mc_nb = 0;
    const uint32_t mc_t0 = mc_now();
    (void)mc_uw; (void)mc_mw; (void)mc_act; (void)mc_sf;
#endif
    for (;;) {
        uint32_t msg = MSG_DONE;
#if MC_PROF
        const uint32_t mc_w0 = mc_now();
        (void)mc_w0;
#endif
        UNPACK(({
            msg = pick_stream(live, prefer);
#if MC_PROF
            mc_uw += mc_now() - mc_w0;
#endif
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, msg);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, msg);
        }));
        MATH((msg = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((msg = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
#if MC_PROF
        const uint32_t mc_a0 = mc_now();
        MATH((mc_mw += mc_a0 - mc_w0));
#endif
        if (msg == MSG_DONE) {
            break;
        }
        const uint32_t cb_in = msg ? CB_COEFF_S1 : CB_COEFF_S0;
        const uint32_t cb_out = msg ? CB_KEEP_S1 : CB_KEEP_S0;
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, DR_IN / 32);
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
#if MC_PROF
        const uint32_t mc_s0 = mc_now();
        (void)mc_s0;
#endif
        if (cull_disabled) {
            MATH((band_keep_all()));
        } else {
            MATH((band_batch(inv_floor_bits)));
        }
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
#if MC_PROF
        MATH((mc_sf += mc_now() - mc_s0));
#endif
        tile_regs_commit();
#if MC_PROF
        MATH((mc_act += mc_now() - mc_a0));
        mc_nb++;
#endif
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(DR_OUT / 32, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
#if MC_PROF
    const uint32_t mc_tot = mc_now() - mc_t0;
#ifdef TRISC_UNPACK
    DeviceTimestampedData("mc_uw", mc_uw);
    DeviceTimestampedData("mc_utot", mc_tot);
#endif
#ifdef TRISC_MATH
    DeviceTimestampedData("mc_mw", mc_mw);
    DeviceTimestampedData("mc_act", mc_act);
    DeviceTimestampedData("mc_sf", mc_sf);
    DeviceTimestampedData("mc_nb", mc_nb);
    DeviceTimestampedData("mc_tot", mc_tot);
#endif
#endif
}
