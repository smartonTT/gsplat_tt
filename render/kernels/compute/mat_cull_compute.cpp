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

#if defined(MATCULL_TRISC_FILL) && MATCULL_TRISC_FILL
// Task #306: the movers post one job per slab on CB_JOB_S0 / CB_JOB_S1
// ({slab, n, done, end}, == sort_subchunk_materialize.cpp). TRISC0 fills the
// coefficient tiles from the slab into CB_COEFF (TRISC-owned here: filled and
// popped by TRISC0 alone), TRISC2 patches each mask tile from CB_KEEP into
// word3 of the slab records, acks the page itself and, after the job's last
// batch, sets the job's done word. MATH and PACK get one message per job.
constexpr uint32_t CB_JOB_S0 = 7;
constexpr uint32_t CB_JOB_S1 = 23;
constexpr uint32_t JOB_SLAB = 0u;
constexpr uint32_t JOB_N = 1u;
constexpr uint32_t DONE_WORD = 2u;
constexpr uint32_t JOB_END = 3u;
constexpr uint32_t COEFF_BATCH = 128u;
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t OPQ_BIAS = 0x4B000000u;
constexpr uint32_t JOB_DONE_MSG = 0xffffffffu;

inline uint32_t n_batches(uint32_t n) { return (n + COEFF_BATCH - 1u) / COEFF_BATCH; }

// Task #319: profiler builds only (PROFILE_KERNEL): per-TRISC cycle counters of
// the fill path, reported once at the end as "fz_*" timestamped-data markers
// (opt/profiler/fill_zones.py). UNPACK: pick_job wait, fill_batch CB wait and
// copy, records. MATH: mailbox wait, copy_tile, band_batch. PACK: mailbox wait,
// tile_regs_wait, pack, patch_batch wait on the packer and patch loop.
#if defined(PROFILE_KERNEL)
#define FZ_PROF 1
inline uint32_t fz_now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }
uint32_t fz_a, fz_b;  // fill_batch: CB wait / copy; patch_batch: packer wait / patch
#else
#define FZ_PROF 0
#endif
#endif

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

#if defined(MATCULL_TRISC_FILL) && MATCULL_TRISC_FILL
// Next job page (address | stream), or JOB_DONE_MSG once both streams ended.
// The page is popped here: the mover does not reuse it before its done word.
inline uint32_t pick_job(uint32_t& live, uint32_t& prefer, uint32_t& slab, uint32_t& n) {
    while (live != 0u) {
        for (uint32_t i = 0; i < 2u; ++i) {
            const uint32_t s = prefer ^ i;
            if ((live & (1u << s)) == 0u) continue;
            const uint32_t cb = s ? CB_JOB_S1 : CB_JOB_S0;
            if (!coeff_ready(cb)) continue;
            asm volatile("fence" ::: "memory");
            const uint32_t page = get_local_cb_interface(cb).fifo_rd_ptr << 4;
            auto p = reinterpret_cast<volatile uint32_t*>(page);
            const bool end = p[JOB_END] != 0u;
            slab = p[JOB_SLAB];
            n = p[JOB_N];
            llk_pop_tiles(cb, 1);
            if (end) {
                live &= ~(1u << s);
                continue;
            }
            prefer = s ^ 1u;
            return page | s;
        }
    }
    return JOB_DONE_MSG;
}

// == sort_subchunk_materialize.cpp fill_coeff_tile (END_WORD is unused here).
// Fills the page at the CB_COEFF read pointer once the unpacker has finished
// with its previous contents (the L1 acked count trails the local one by less
// than the CB depth).
inline void fill_batch(uint32_t cb, uint32_t slab, uint32_t base, uint32_t n) {
    auto& ci = get_local_cb_interface(cb);
    const uint32_t depth = ci.fifo_size / ci.fifo_page_size;
    volatile tt_l1_ptr uint32_t* ap = get_cb_tiles_acked_ptr(cb);
#if FZ_PROF
    const uint32_t fz_t0 = fz_now();
#endif
    while (static_cast<uint16_t>(ci.tiles_acked - static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(ap)))) >=
           depth) {
    }
#if FZ_PROF
    const uint32_t fz_t1 = fz_now();
    fz_a += fz_t1 - fz_t0;
#endif
    const uint32_t tile = ci.fifo_rd_ptr << 4;
    for (uint32_t i = 0; i < n; ++i) {
        auto src = reinterpret_cast<volatile uint32_t*>(slab + (base + i) * L1_SPLAT_BYTES);
        auto dst = reinterpret_cast<volatile uint32_t*>(tile) + 192u * (i >> 5) + 2u * (i & 31u);
        const uint32_t w0 = src[0], w1 = src[1], w2 = src[2];
        const uint32_t w4 = src[4], w5 = src[5], w6 = src[6];
        dst[0] = w0;
        dst[1] = w1;
        dst[64] = w2;
        dst[65] = OPQ_BIAS | (w6 & 0xffffu);
        dst[128] = w4;
        dst[129] = w5;
    }
    asm volatile("fence" ::: "memory");  // tile words reach L1 before the unpacker reads them
#if FZ_PROF
    fz_b += fz_now() - fz_t1;
#endif
}
#endif
#endif

#if defined(TRISC_PACK) && defined(MATCULL_TRISC_FILL) && MATCULL_TRISC_FILL
// == sort_subchunk_materialize.cpp patch_batch, on the page at `keep` once the
// packer has written it (its L1 received count caught up); then frees the page.
inline void patch_batch(uint32_t cb, uint32_t keep_addr, uint32_t slab, uint32_t base, uint32_t n) {
    const uint16_t want = get_local_cb_interface(cb).tiles_received;
    volatile tt_l1_ptr uint32_t* rp = get_cb_tiles_received_ptr(cb);
#if FZ_PROF
    const uint32_t fz_t0 = fz_now();
#endif
    while (static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(rp))) != want) {
    }
#if FZ_PROF
    const uint32_t fz_t1 = fz_now();
    fz_a += fz_t1 - fz_t0;
#endif
    asm volatile("fence" ::: "memory");
    auto keep = reinterpret_cast<volatile uint32_t*>(keep_addr);
    auto rec = reinterpret_cast<volatile uint32_t*>(slab + base * L1_SPLAT_BYTES);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t o = 64u * (i >> 5) + 2u * (i & 31u);
        const uint32_t lo = keep[o], hi = keep[o + 1u];
        rec[i * 8u + 3u] = (lo & 0xffffu) | (hi << 16);
    }
    asm volatile("fence" ::: "memory");  // word3 and keep reads done before the page is freed
    *get_cb_tiles_acked_ptr(cb) = want;
#if FZ_PROF
    fz_b += fz_now() - fz_t1;
#endif
}
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
#if defined(MATCULL_TRISC_FILL) && MATCULL_TRISC_FILL
#if FZ_PROF
    // w: pick_job / mailbox wait; c: MATH copy_tile, PACK tile_regs_wait;
    // k: MATH band_batch, PACK pack + push; nb / nr / nj: batches, records, jobs.
    uint32_t fz_w = 0, fz_c = 0, fz_k = 0, fz_nb = 0, fz_nr = 0, fz_nj = 0;
    fz_a = fz_b = 0;
    const uint32_t fz_t0 = fz_now();
    (void)fz_c;
    (void)fz_k;
#endif
    for (;;) {
        uint32_t job = JOB_DONE_MSG;
        uint32_t slab = 0, n = 0;
        (void)slab;
        (void)n;
#if FZ_PROF
        const uint32_t fz_w0 = fz_now();
#endif
        UNPACK(({
            job = pick_job(live, prefer, slab, n);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, job);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, job);
        }));
        MATH((job = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((job = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
#if FZ_PROF
        fz_w += fz_now() - fz_w0;
#endif
        if (job == JOB_DONE_MSG) {
            break;
        }
        const uint32_t s = job & 1u;
        const uint32_t page = job & ~1u;
        const uint32_t cb_in = s ? CB_COEFF_S1 : CB_COEFF_S0;
        const uint32_t cb_out = s ? CB_KEEP_S1 : CB_KEEP_S0;
        MATH(({
            asm volatile("fence" ::: "memory");
            n = reinterpret_cast<volatile uint32_t*>(page)[JOB_N];
        }));
        PACK(({
            asm volatile("fence" ::: "memory");
            slab = reinterpret_cast<volatile uint32_t*>(page)[JOB_SLAB];
            n = reinterpret_cast<volatile uint32_t*>(page)[JOB_N];
        }));
        const uint32_t nb = n_batches(n);
#if FZ_PROF
        fz_nj++;
        fz_nb += nb;
        fz_nr += n;
#endif
        for (uint32_t b = 0; b < nb; ++b) {
            const uint32_t base = b * COEFF_BATCH;
            const uint32_t cnt = (n - base < COEFF_BATCH) ? (n - base) : COEFF_BATCH;
            (void)cnt;
            UNPACK((fill_batch(cb_in, slab, base, cnt)));
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_in);
#if FZ_PROF
            const uint32_t fz_c0 = fz_now();
            (void)fz_c0;
#endif
            copy_tile(cb_in, 0, DR_IN / 32);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
#if FZ_PROF
            const uint32_t fz_k0 = fz_now();
            (void)fz_k0;
            MATH((fz_c += fz_k0 - fz_c0));
#endif
            if (cull_disabled) {
                MATH((band_keep_all()));
            } else {
                MATH((band_batch(inv_floor_bits)));
            }
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
#if FZ_PROF
            MATH((fz_k += fz_now() - fz_k0));
#endif
            tile_regs_commit();
#if FZ_PROF
            const uint32_t fz_r0 = fz_now();
            (void)fz_r0;
#endif
            tile_regs_wait();
#if FZ_PROF
            const uint32_t fz_r1 = fz_now();
            (void)fz_r1;
            PACK((fz_c += fz_r1 - fz_r0));
#endif
            cb_reserve_back(cb_out, 1);
            uint32_t keep = 0;
            (void)keep;
            PACK((keep = get_local_cb_interface(cb_out).fifo_wr_ptr << 4));
            pack_tile(DR_OUT / 32, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
#if FZ_PROF
            PACK((fz_k += fz_now() - fz_r1));
#endif
            PACK((patch_batch(cb_out, keep, slab, base, cnt)));
            cb_pop_front(cb_in, 1);
        }
        PACK(({
            asm volatile("fence" ::: "memory");  // all word3 stores before done
            reinterpret_cast<volatile uint32_t*>(page)[DONE_WORD] = 1u;
            asm volatile("fence" ::: "memory");
        }));
    }
#if FZ_PROF
    const uint32_t fz_tot = fz_now() - fz_t0;
#ifdef TRISC_UNPACK
    DeviceTimestampedData("fz_u_tot", fz_tot);
    DeviceTimestampedData("fz_u_pick", fz_w);
    DeviceTimestampedData("fz_u_fillw", fz_a);
    DeviceTimestampedData("fz_u_fill", fz_b);
    DeviceTimestampedData("fz_u_nb", fz_nb);
    DeviceTimestampedData("fz_u_nr", fz_nr);
    DeviceTimestampedData("fz_u_nj", fz_nj);
#endif
#ifdef TRISC_MATH
    DeviceTimestampedData("fz_m_tot", fz_tot);
    DeviceTimestampedData("fz_m_mbw", fz_w);
    DeviceTimestampedData("fz_m_copy", fz_c);
    DeviceTimestampedData("fz_m_band", fz_k);
#endif
#ifdef TRISC_PACK
    DeviceTimestampedData("fz_p_tot", fz_tot);
    DeviceTimestampedData("fz_p_mbw", fz_w);
    DeviceTimestampedData("fz_p_regw", fz_c);
    DeviceTimestampedData("fz_p_pack", fz_k);
    DeviceTimestampedData("fz_p_patchw", fz_a);
    DeviceTimestampedData("fz_p_patch", fz_b);
#endif
#endif
#else
    for (;;) {
        uint32_t msg = MSG_DONE;
        UNPACK(({
            msg = pick_stream(live, prefer);
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, msg);
            ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, msg);
        }));
        MATH((msg = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
        PACK((msg = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId)));
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
        if (cull_disabled) {
            MATH((band_keep_all()));
        } else {
            MATH((band_batch(inv_floor_bits)));
        }
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(DR_OUT / 32, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
#endif
}
