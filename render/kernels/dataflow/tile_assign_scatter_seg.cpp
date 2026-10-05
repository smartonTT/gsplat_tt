// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign K2 over pfwc segments (task #125, lever B, GSPLAT_TT_PFWC_FUSE=1).
// tile_assign_scatter.cpp (TA_K2_AABB) for the fused pfwc writer's layout
// (pfwc_fuse.h): the visible gaussians sit in per-core segments with
// segment-local pair offsets and a per-core counts table instead of a dense
// compact stream with global offsets. Every mover reads the counts table, builds
// the segment table (pair / storage bases, M, P) on its own, takes its page
// range of the (P_pub + 15) / 16 pair pages and emits (gid = storage index,
// tid) in the legacy gaussian-major order. No scan kernel and no host P read
// sit between pfwc and this K2. Core 0's NCRISC publishes
//   ta_pairs_P = [P_pub, pad16(P_pub), overflow, P, 0...] (gather_vis_scan layout)
//   proj_M     = [M, P, overflow, 0...]
// with P_pub = min(P, p_cap); the host grows the pair buffers and reruns on
// overflow.
//
//
// K2_DIET (task #170, default; host GSPLAT_TT_K2_DIET=0 is the kill switch):
// the pairs come from pfwc_fuse::emit_pairs_diet (read-ahead ring of lofs / box
// pages, pair pages written without a flush each, no divide per pair). With
// row_pages != 0 (host GSPLAT_TT_K2_FOLD, default on when the sort's page split
// is this one) the mover also counts its pairs per tile in its local memory
// (an L1 counter is loaded right after its store, task #26) and writes them as
// row 2 * k + mover of the count-rows buffer (row_pages 64 B pages, zero past
// the screen tiles): the one-launch sort's per-mover count of the same pages,
// so the sort skips its count pass and first barrier (sort_bin_onelaunch.cpp
// fold). With arg 20 != 0 the mover takes the sort's speed-proportional range
// (task #174, pfwc_fuse::k2_range_speed) instead of k2_range.
//
// RUNTIME ARGS
//   0: proj_m_offs (segment-local)   1: proj_m_aabb   2: counts table
//   3: gids   4: tids   5: ta_pairs_P   6: proj_M        (64 B pages)
//   7: nseg (pfwc cores)   8: num_tiles   9: K2 core k   10: K2 cores
//   11: mover (0 BRISC, 1 NCRISC)   12: dual   13: BRISC permille
//   14: p_cap (pair buffer capacity, elements)   15: tiles_x
//   16: count rows (K2_DIET)   17: row_pages (0: no count rows)
//   18, 19: speed sums before / after this mover   20: speed total (0: k2_range)
//
// COMPILE-TIME ARGS: 8 TensorAccessorArgs in runtime-arg order 0..6, 16.
// CB: one raw scratch CB per mover (id 0 on NCRISC, TA_CB_OFFSET on BRISC):
// nseg + 6 pages of 64 B (+64 B alignment slack); K2_DIET adds the read-ahead
// ring and the pair staging (4 x 16 pages) and the count row (K2_FOLD_TILES /
// 16 pages).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwc_fuse.h"

#ifndef TA_CB_OFFSET
#define TA_CB_OFFSET 0
#endif
#ifndef K2_DIET
#define K2_DIET 0
#endif
#ifndef K2_TRISC
#define K2_TRISC 0  // task #274: TRISC jobs (host GSPLAT_TT_K2_TRISC=1, needs K2_DIET)
#endif
#if K2_TRISC
#include "k2_trisc.h"
#ifndef K2_TJ0
#define K2_TJ0 430u  // permille of the range for job 0 (TRISC 0 / 2)
#endif
#ifndef K2_TJ1
#define K2_TJ1 215u  // permille for job 1 (TRISC 1, runs both movers' jobs)
#endif
#endif
#ifndef K2_FOLD_TILES
#define K2_FOLD_TILES 1024u  // tiles a count row can hold (local memory)
#endif
// Task #274 (host knob GSPLAT_TT_K2_PROF=1, profiling only): each mover sums
// wall-clock cycles per part (K2P_* below) and records the totals as Tracy
// timestamped-data markers "k2p_*" at its end (value = cycles over the launch).
// Off: the macros are empty and the binary is the same as before.
#ifndef K2_PROF
#define K2_PROF 0
#endif
#if K2_PROF
enum { K2P_SETUP, K2P_RISS, K2P_RDW, K2P_WISS, K2P_WFL, K2P_PAIRS, K2P_ROWS, K2P_WBAR, K2P_TOT,
       // K2_TRISC only (task #291): job prep, own part, DONE waits, write-back,
       // per job TRISC start delay (post -> GO seen) and run time, pages.
       K2P_JPREP, K2P_OWN, K2P_JW0, K2P_JW1, K2P_JWB, K2P_TS0, K2P_TR0, K2P_TS1, K2P_TR1,
       K2P_PG0, K2P_PG1, K2P_PGOWN, K2P_N };
uint32_t g_k2p[K2P_N];
#define K2P_NOW() (reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0])
#define K2P_T0(v) const uint32_t v = K2P_NOW()
#define K2P_ADD(i, t0) g_k2p[i] += K2P_NOW() - (t0)
#else
#define K2P_T0(v)
#define K2P_ADD(i, t0)
#endif

#if K2_DIET
namespace {
// emit_pairs_diet's Io: NoC reads into the ring, NoC writes from the staging.
// Ring and staging are read / written with plain loads and stores; the
// compiler barriers order them against the NoC calls.
template <class LA, class BA, class GA, class TA>
struct DietIo {
    const LA& lofs_acc;
    const BA& box_acc;
    const GA& gids_acc;
    const TA& tids_acc;
    uint32_t ra_lofs, ra_box, st_gid, st_tid;
    void issue_lofs(uint32_t page, uint32_t slot) const {
        K2P_T0(t0);
        noc_async_read(get_noc_addr(page, lofs_acc), ra_lofs + slot * 64u, 64u);
        K2P_ADD(K2P_RISS, t0);
    }
    void issue(uint32_t page, uint32_t slot) const {
        K2P_T0(t0);
        noc_async_read(get_noc_addr(page, lofs_acc), ra_lofs + slot * 64u, 64u);
        noc_async_read(get_noc_addr(page, box_acc), ra_box + slot * 64u, 64u);
        K2P_ADD(K2P_RISS, t0);
    }
    void wait_reads() const {
        K2P_T0(t0);
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        K2P_ADD(K2P_RDW, t0);
    }
    const uint32_t* lofs_slot(uint32_t slot) const {
        return reinterpret_cast<const uint32_t*>(ra_lofs + slot * 64u);
    }
    const uint32_t* box_slot(uint32_t slot) const {
        return reinterpret_cast<const uint32_t*>(ra_box + slot * 64u);
    }
    uint32_t* gid_slot(uint32_t o) const { return reinterpret_cast<uint32_t*>(st_gid + o * 64u); }
    uint32_t* tid_slot(uint32_t o) const { return reinterpret_cast<uint32_t*>(st_tid + o * 64u); }
    void write_page(uint32_t page, uint32_t o) const {
        asm volatile("" ::: "memory");
        K2P_T0(t0);
        noc_async_write(st_gid + o * 64u, get_noc_addr(page, gids_acc), 64u);
        noc_async_write(st_tid + o * 64u, get_noc_addr(page, tids_acc), 64u);
        K2P_ADD(K2P_WISS, t0);
    }
    void writes_flushed() const {
        K2P_T0(t0);
        noc_async_writes_flushed();
        asm volatile("" ::: "memory");
        K2P_ADD(K2P_WFL, t0);
    }
};
}  // namespace
#endif

void kernel_main() {
    K2P_T0(t_tot);
#if K2_PROF
    for (uint32_t i = 0; i < K2P_N; i++) g_k2p[i] = 0;
#endif
    constexpr uint32_t PW = pfwc_fuse::PAGE_WORDS;
    constexpr uint32_t PB = PW * 4;
    uint32_t addr[7];
    for (uint32_t k = 0; k < 7; k++) addr[k] = get_arg_val<uint32_t>(k);
    const uint32_t nseg = get_arg_val<uint32_t>(7);
    const uint32_t num_tiles = get_arg_val<uint32_t>(8);
    const uint32_t kc = get_arg_val<uint32_t>(9);
    const uint32_t ncores = get_arg_val<uint32_t>(10);
    const uint32_t mover = get_arg_val<uint32_t>(11);
    const uint32_t dual = get_arg_val<uint32_t>(12);
    const uint32_t permille = get_arg_val<uint32_t>(13);
    const uint32_t p_cap = get_arg_val<uint32_t>(14);
    const uint32_t tiles_x = get_arg_val<uint32_t>(15);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    const auto lofs_acc = TensorAccessor(a0, addr[0], PB);
    const auto box_acc = TensorAccessor(a1, addr[1], PB);
    const auto cnt_acc = TensorAccessor(a2, addr[2], PB);
    const auto gids_acc = TensorAccessor(a3, addr[3], PB);
    const auto tids_acc = TensorAccessor(a4, addr[4], PB);
    const auto pp_acc = TensorAccessor(a5, addr[5], PB);
    const auto mp_acc = TensorAccessor(a6, addr[6], PB);

    const uint32_t l1_tab = (get_write_ptr(TA_CB_OFFSET) + 63u) & ~63u;
    const uint32_t l1_pub = l1_tab + nseg * PB;
    auto tab = reinterpret_cast<volatile uint32_t*>(l1_tab);
    auto pub = reinterpret_cast<volatile uint32_t*>(l1_pub);

    for (uint32_t c = 0; c < nseg; c++)
        noc_async_read(get_noc_addr(c, cnt_acc), l1_tab + c * PB, PB);
    noc_async_read_barrier();
    uint32_t M = 0, P = 0;
    pfwc_fuse::seg_table(tab, nseg, num_tiles, &M, &P);
    const uint32_t P_pub = P < p_cap ? P : p_cap;
    const uint32_t overflow = P > p_cap ? 1u : 0u;

    if (kc == 0 && mover == 1) {
        for (uint32_t w = 0; w < PW; w++) pub[w] = 0;
        pub[0] = P_pub;
        pub[1] = (P_pub + PW - 1) / PW * PW;
        pub[2] = overflow;
        pub[3] = P;
        noc_async_write(l1_pub, get_noc_addr(0, pp_acc), PB);
        noc_async_write_barrier();
        pub[0] = M;
        pub[1] = P;
        pub[2] = overflow;
        pub[3] = 0;
        noc_async_write(l1_pub, get_noc_addr(0, mp_acc), PB);
    }

    uint32_t pg0 = 0, npg = 0;
    pfwc_fuse::k2_range(P_pub, ncores, kc, mover, dual, permille, &pg0, &npg);
#if K2_DIET
    const uint32_t row_pages = get_arg_val<uint32_t>(17);
    const uint32_t sp_tot = get_arg_val<uint32_t>(20);  // fold under the sort's speed split
    if (sp_tot != 0)
        pfwc_fuse::k2_range_speed(P_pub, get_arg_val<uint32_t>(18), get_arg_val<uint32_t>(19),
                                  sp_tot, &pg0, &npg);
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    const auto row_acc = TensorAccessor(a7, get_arg_val<uint32_t>(16), PB);
    constexpr uint32_t RS = pfwc_fuse::RA_SLOTS, OS = pfwc_fuse::OUT_SLOTS;
    const uint32_t l1_ra_lofs = l1_pub + PB;
    const uint32_t l1_ra_box = l1_ra_lofs + RS * PB;
    const uint32_t l1_st_gid = l1_ra_box + RS * PB;
    const uint32_t l1_st_tid = l1_st_gid + OS * PB;
    const uint32_t l1_row = l1_st_tid + OS * PB;
    const DietIo<decltype(lofs_acc), decltype(box_acc), decltype(gids_acc), decltype(tids_acc)> io{
        lofs_acc, box_acc, gids_acc, tids_acc, l1_ra_lofs, l1_ra_box, l1_st_gid, l1_st_tid};
    const uint32_t span = row_pages * PW;  // >= screen tiles
#if K2_TRISC
    // Task #274: jobs for the idle TRISCs (k2_trisc.h). Header first, then GO2, GO.
    const uint32_t l1_cbj = get_write_ptr(k2_trisc::CB_JOB + TA_CB_OFFSET);
    auto job_post = [&](uint32_t j, uint32_t jpg0, uint32_t jn, uint32_t jc, uint32_t jlo,
                        uint32_t jnin, uint32_t jspan) {
        auto h = reinterpret_cast<volatile uint32_t*>(k2_trisc::job_addr(l1_cbj, j));
        h[k2_trisc::H_DONE] = 0;
        h[k2_trisc::H_PG0] = jpg0;
        h[k2_trisc::H_NPG] = jn;
        h[k2_trisc::H_C] = jc;
        h[k2_trisc::H_LO] = jlo;
        h[k2_trisc::H_NIN] = jnin;
        h[k2_trisc::H_PPUB] = P_pub;
        h[k2_trisc::H_TX] = tiles_x;
        h[k2_trisc::H_NSEG] = nseg;
        h[k2_trisc::H_TAB] = l1_tab;
        h[k2_trisc::H_SPAN] = jspan;
        asm volatile("fence" ::: "memory");
        h[k2_trisc::H_GO2] = k2_trisc::MAGIC2;
        asm volatile("fence" ::: "memory");
        h[k2_trisc::H_GO] = k2_trisc::MAGIC;
        asm volatile("fence" ::: "memory");
    };
    auto job_wait = [&](uint32_t j) -> uint32_t {
        const uint32_t w = k2_trisc::job_addr(l1_cbj, j);
        auto h = reinterpret_cast<volatile uint32_t*>(w);
        for (;;) {
            invalidate_l1_cache();
            if (h[k2_trisc::H_DONE] == k2_trisc::MAGIC) break;
        }
        invalidate_l1_cache();
        return w;
    };
    auto job_clear = [&](uint32_t j) {
        auto h = reinterpret_cast<volatile uint32_t*>(k2_trisc::job_addr(l1_cbj, j));
        h[k2_trisc::H_GO] = 0;
        h[k2_trisc::H_GO2] = 0;
        h[k2_trisc::H_DONE] = 0;
        asm volatile("fence" ::: "memory");
    };
#endif
    if (row_pages != 0 && span <= K2_FOLD_TILES) {
        uint32_t cnt[K2_FOLD_TILES];
        for (uint32_t t = 0; t < span; t++) cnt[t] = 0;
        K2P_ADD(K2P_SETUP, t_tot);
        {
            DeviceZoneScopedN("k2_pairs");
            K2P_T0(t_pairs);
#if K2_TRISC
            uint32_t n[2] = {0, 0}, st[3] = {pg0, pg0, pg0}, c[3] = {0, 0, 0}, lo[3] = {0, 0, 0};
            uint32_t n_in[2] = {0, 0};
            if (npg != 0) {
                pfwc_fuse::k2_jobs(npg, K2_TJ0, K2_TJ1, K2_TCAP, &n[0], &n[1]);
                st[1] = pg0 + n[0];
                st[2] = st[1] + n[1];
                for (uint32_t i = 0; i < 3; i++)
                    if (st[i] * PW < P_pub) pfwc_fuse::diet_start(tab, nseg, st[i] * PW, io, &c[i], &lo[i]);
                bool fit = true;
                for (uint32_t j = 0; j < 2 && fit; j++) {
                    if (n[j] == 0 || st[j] * PW >= P_pub) continue;
                    const bool to_end = st[j + 1] * PW >= P_pub;
                    const uint32_t w = k2_trisc::job_addr(l1_cbj, j);
                    n_in[j] = pfwc_fuse::diet_window(
                        tab, nseg, c[j], lo[j], to_end, c[j + 1], lo[j + 1], K2_TCAP,
                        [&](uint32_t k, uint32_t page) {
                            noc_async_read(get_noc_addr(page, lofs_acc), w + k2_trisc::LOFS_OFF + k * PB, PB);
                            noc_async_read(get_noc_addr(page, box_acc), w + k2_trisc::BOX_OFF + k * PB, PB);
                        });
                    if (n_in[j] > K2_TCAP) fit = false;
                }
                noc_async_read_barrier();
                if (!fit) {
                    n[0] = n[1] = 0;
                    st[1] = st[2] = pg0;
                    c[2] = c[0];
                    lo[2] = lo[0];
                }
            }
#if K2_PROF
            uint32_t t_post[2];
#endif
            for (uint32_t j = 0; j < 2; j++) {
                job_post(j, st[j], n[j], c[j], lo[j], n_in[j], n[j] ? span : 0u);
#if K2_PROF
                t_post[j] = K2P_NOW();
#endif
            }
            K2P_ADD(K2P_JPREP, t_pairs);
            K2P_T0(t_own);
            if (pg0 + npg > st[2])
                pfwc_fuse::emit_pairs_diet_from<true>(tab, nseg, P_pub, tiles_x, st[2], pg0 + npg - st[2],
                                                      c[2], lo[2], io, cnt);
            K2P_ADD(K2P_OWN, t_own);
            K2P_ADD(K2P_PAIRS, t_pairs);
            for (uint32_t j = 0; j < 2; j++) {
                K2P_T0(t_w);
                const uint32_t w = job_wait(j);
#if K2_PROF
                K2P_ADD(K2P_JW0 + j, t_w);
                K2P_T0(t_wb);
                if (n[j] != 0) {
                    auto h = reinterpret_cast<volatile uint32_t*>(w);
                    g_k2p[K2P_TS0 + 2 * j] += h[k2_trisc::H_TS] - t_post[j];
                    g_k2p[K2P_TR0 + 2 * j] += h[k2_trisc::H_TE] - h[k2_trisc::H_TS];
                    g_k2p[K2P_PG0 + j] += n[j];
                }
#endif
                for (uint32_t q = 0; q < n[j]; q++) {
                    noc_async_write(w + k2_trisc::GID_OFF + q * PB, get_noc_addr(st[j] + q, gids_acc), PB);
                    noc_async_write(w + k2_trisc::TID_OFF + q * PB, get_noc_addr(st[j] + q, tids_acc), PB);
                }
                if (n[j] != 0) {
                    auto jc = reinterpret_cast<volatile uint32_t*>(w + k2_trisc::CNT_OFF);
                    for (uint32_t t = 0; t < span; t++) cnt[t] += jc[t];
                }
                K2P_ADD(K2P_JWB, t_wb);
            }
            K2P_T0(t_fl);
            noc_async_writes_flushed();
            K2P_ADD(K2P_JWB, t_fl);
#if K2_PROF
            g_k2p[K2P_PGOWN] += pg0 + npg - st[2];
#endif
            for (uint32_t j = 0; j < 2; j++) job_clear(j);
#else
            pfwc_fuse::emit_pairs_diet<true>(tab, nseg, P_pub, tiles_x, pg0, npg, io, cnt);
            K2P_ADD(K2P_PAIRS, t_pairs);
#endif
        }
        DeviceZoneScopedN("k2_rows");
        K2P_T0(t_rows);
        auto rowp = reinterpret_cast<volatile uint32_t*>(l1_row);
        for (uint32_t t = 0; t < span; t++) rowp[t] = cnt[t];
        asm volatile("fence" ::: "memory");
        const uint32_t r0 = (kc * 2u + mover) * row_pages;
        for (uint32_t q = 0; q < row_pages; q++)
            noc_async_write(l1_row + q * PB, get_noc_addr(r0 + q, row_acc), PB);
        K2P_ADD(K2P_ROWS, t_rows);
    } else {
        DeviceZoneScopedN("k2_pairs");
        pfwc_fuse::emit_pairs_diet<false>(tab, nseg, P_pub, tiles_x, pg0, npg, io,
                                          static_cast<uint32_t*>(nullptr));
#if K2_TRISC
        for (uint32_t j = 0; j < 2; j++) job_post(j, 0, 0, 0, 0, 0, 0);
        for (uint32_t j = 0; j < 2; j++) job_wait(j);
        for (uint32_t j = 0; j < 2; j++) job_clear(j);
#endif
    }
#else
    const uint32_t l1_lofs = l1_pub + PB;
    const uint32_t l1_box = l1_lofs + PB;
    const uint32_t l1_gid = l1_box + PB;
    const uint32_t l1_tid = l1_gid + PB;
    auto lofs = reinterpret_cast<volatile uint32_t*>(l1_lofs);
    auto box = reinterpret_cast<volatile uint32_t*>(l1_box);
    auto gidp = reinterpret_cast<volatile uint32_t*>(l1_gid);
    auto tidp = reinterpret_cast<volatile uint32_t*>(l1_tid);
    if (npg != 0) {
        uint32_t lofs_page = 0xFFFFFFFFu, box_page = 0xFFFFFFFFu;
        auto read_lofs = [&](uint32_t s) -> uint32_t {
            const uint32_t pg = s / PW;
            if (pg != lofs_page) {
                noc_async_read(get_noc_addr(pg, lofs_acc), l1_lofs, PB);
                noc_async_read_barrier();
                lofs_page = pg;
            }
            return lofs[s % PW];
        };
        auto read_box = [&](uint32_t s) -> uint32_t {
            const uint32_t pg = s / PW;
            if (pg != box_page) {
                noc_async_read(get_noc_addr(pg, box_acc), l1_box, PB);
                noc_async_read_barrier();
                box_page = pg;
            }
            return box[s % PW];
        };
        uint32_t idx = 0, out_page = pg0;
        auto out = [&](uint32_t, uint32_t gid, uint32_t tid) {
            gidp[idx] = gid;
            tidp[idx] = tid;
            if (++idx == PW) {
                noc_async_write(l1_gid, get_noc_addr(out_page, gids_acc), PB);
                noc_async_write(l1_tid, get_noc_addr(out_page, tids_acc), PB);
                noc_async_writes_flushed();
                out_page++;
                idx = 0;
            }
        };
        pfwc_fuse::emit_pairs(tab, nseg, P_pub, tiles_x, pg0 * PW, (pg0 + npg) * PW, read_lofs,
                              read_box, out);
    }
#endif
    K2P_T0(t_wbar);
    noc_async_write_barrier();
#if K2_PROF
    K2P_ADD(K2P_WBAR, t_wbar);
    K2P_ADD(K2P_TOT, t_tot);
    DeviceTimestampedData("k2p_setup", g_k2p[K2P_SETUP]);
    DeviceTimestampedData("k2p_riss", g_k2p[K2P_RISS]);
    DeviceTimestampedData("k2p_rdw", g_k2p[K2P_RDW]);
    DeviceTimestampedData("k2p_wiss", g_k2p[K2P_WISS]);
    DeviceTimestampedData("k2p_wfl", g_k2p[K2P_WFL]);
    DeviceTimestampedData("k2p_pairs", g_k2p[K2P_PAIRS]);
    DeviceTimestampedData("k2p_rows", g_k2p[K2P_ROWS]);
    DeviceTimestampedData("k2p_wbar", g_k2p[K2P_WBAR]);
    DeviceTimestampedData("k2p_tot", g_k2p[K2P_TOT]);
    DeviceTimestampedData("k2p_npg", npg);
#if K2_TRISC
    DeviceTimestampedData("k2p_jprep", g_k2p[K2P_JPREP]);
    DeviceTimestampedData("k2p_own", g_k2p[K2P_OWN]);
    DeviceTimestampedData("k2p_jw0", g_k2p[K2P_JW0]);
    DeviceTimestampedData("k2p_jw1", g_k2p[K2P_JW1]);
    DeviceTimestampedData("k2p_jwb", g_k2p[K2P_JWB]);
    DeviceTimestampedData("k2p_ts0", g_k2p[K2P_TS0]);
    DeviceTimestampedData("k2p_tr0", g_k2p[K2P_TR0]);
    DeviceTimestampedData("k2p_ts1", g_k2p[K2P_TS1]);
    DeviceTimestampedData("k2p_tr1", g_k2p[K2P_TR1]);
    DeviceTimestampedData("k2p_pg0", g_k2p[K2P_PG0]);
    DeviceTimestampedData("k2p_pg1", g_k2p[K2P_PG1]);
    DeviceTimestampedData("k2p_pgown", g_k2p[K2P_PGOWN]);
#endif
#endif
}
