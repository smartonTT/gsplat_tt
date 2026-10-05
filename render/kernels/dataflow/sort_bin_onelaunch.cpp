// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN in one launch (task #106, lever 1; host knob GSPLAT_TT_SORT_ONELAUNCH=1).
//
// Replaces count (sort_bin.cpp mode 0) + hist D2H + host bin layout + layout
// upload + emit (mode 1) + radix + publish with ONE launch. Every tile owns a
// fixed-capacity bucket of `tile_cap` 32 B records at slot t * tile_cap of the
// bucket buffer (2 KB pages, 64 records each). The bucket offsets come from a
// device prefix sum between two semaphore barriers (no NoC atomic return values;
// task #24's fetch-and-add version returned wrong old values and hung):
//
//   1. count    each mover counts the kept pairs of its page range per tile
//               (BRISC = mover 0 [lo, mid), NCRISC = mover 1 [mid, hi)).
//               Mover 0 adds mover 1's row and writes the core's count row.
//   2. barrier  every core's mover 0 increments arrive1 on the coordinator
//               (core 0's mover 0), which releases all cores.
//   3. prefix   core c owns the count-row pages p = c, c + num_cores, ...: it
//               reads page p of every core's row, writes each core's exclusive
//               prefix (its base in the tile's bucket) to the base rows, and
//               the per-tile totals and padded totals (sum over cores of
//               round_up(count, 16), the legacy LPT cost) to the totals rows.
//   4. barrier  same as 2 (separate semaphores, so no reset race).
//   5. emit     both movers scatter their records: mover 0 from base[t],
//               mover 1 from base[t] + mover 0's count. That is the canonical
//               (core, BRISC pages, NCRISC pages) order of the prefix-sum
//               layout, so the materialize's stable depth radix of a bucket
//               gives the legacy output byte for byte.
//
// Records are packed exactly like sort_bin.cpp mode 1 (pack_invariants /
// pack_rec). A tile over tile_cap records drops the excess and the host fails
// the frame (the legacy MAX_TILE_ENTRIES limit is the same number). No (key, id)
// layout and no packoc write-back: in this mode nothing reads sort_sorted_ids or
// blendrec words 10, 11 (the over-cap gather they fed is replaced by the
// materialize's bucket path). EMIT_PUBOC (host GSPLAT_TT_EMIT_PUBOC, task #100):
// the gather published op/color UNORM16 (blendrec[10], [11]) and the depth key
// ([12]), so the emit copies them and reads no depth pages.
//
// Fold (task #170, arg 27 = 1; host GSPLAT_TT_K2_FOLD=0 is the kill switch):
// the K2 that wrote the pairs (tile_assign_scatter_seg.cpp K2_DIET) took the
// same page split and already counted each mover's pairs per tile into row
// 2 * core + mover of arg 6 (keep is all ones, so every pair counts). Steps 1,
// 1b and 2 go away: the prefix reads both movers' rows and adds them, NCRISC
// reads BRISC's row for its cursors. Each mover fills its pair window while it
// waits (NCRISC for the bases, BRISC in barrier 2) and reads no keep pages (an
// all-ones page stands in for them).
//
// RUNTIME ARGS
//   0 gids  1 tids  2 keep  3 depth  4 blendrec  5 bucket  6 count rows
//   (fold: the K2's per-mover rows)  7 base rows  8 totals rows  9 pg_lo
//   10 pg_hi (this mover's pages)  11 P  12 num_tiles  13 row_pages (a row =
//   row_pages 64 B pages)  14 core_id  15 num_cores  16 tile_cap  17 tiles_x
//   18 mover (0 BRISC, 1 NCRISC)
//   19..24 semaphores: counted, based, arrive1, release1, arrive2, release2
//   25 coordinator NoC x  26 coordinator NoC y  27 fold
//   28, 29 early launch only (below), else 0
//   30..34 gen only (fold = 2, task #298, see gen_window), else 0
//   35.. (core 0 mover 0 only) NoC x | y << 16 of logical core r, r < num_cores
// Gen (task #298, arg 27 = 2; host GSPLAT_TT_K2_FOLDED=0 is the kill switch):
// the fold with a count-only K2 that wrote the count rows, M and P but no
// pairs. Each mover makes its window's pairs in L1 with the K2's walk while it
// waits (gen_window) and writes only the pages past its window to DRAM.
// Early launch (task #198, host GSPLAT_TT_SORT_OL_EARLY): enqueued right behind
// the fold's K2, before the host knows P. 11 == 0xFFFFFFFF; 9 is then the
// ta_pairs_P page ([0] P_pub), 10 and 28 this mover's running mover-speed sums
// (the K2's args 18, 19) and 29 their total (K2 arg 20): the kernel takes the
// K2's own page range (pfwc_fuse::k2_range_speed), so the fold holds by
// construction.
// COMPILE-TIME ARGS: 9 TensorAccessorArgs (gids, tids, keep, depth, blendrec,
//   bucket, count rows, base rows, totals rows).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwc_fuse.h"
#include "sort_bin_fp32.h"
#include "sort_onelaunch_algo.h"
#if defined(OL_FILL_BULK) && OL_FILL_BULK == 2
#include "api/debug/dprint.h"
#endif

#ifndef EMIT_PUBOC
#define EMIT_PUBOC 0u
#endif
// v2 emit (task #124; host knobs GSPLAT_TT_OL_PB / _RING / _WIN_PAGES). The
// defaults here are the v1 emit's batching and write pattern.
#ifndef OL_PB
#define OL_PB 1u  // pair pages per emit batch
#endif
#ifndef OL_RING
#define OL_RING 0u  // records per coalesced run (0: one 32 B write per record)
#endif
#ifndef OL_RING_TILES
#define OL_RING_TILES 1024u  // tiles with a ring (CB_RING size); more: rings off
#endif
// Task #181 (host knob GSPLAT_TT_OL_FILL_BULK=0 turns it off): the fold's
// window fill reads one run per DRAM bank, see fill_window below.
#ifndef OL_FILL_BULK
#define OL_FILL_BULK 1
#endif
#ifndef OL_WIN_PAGES
#define OL_WIN_PAGES 1536u  // == sort_device.cpp onelaunch window
#endif
// Task #196 (host knob GSPLAT_TT_OL_BREC_BULK=0 turns it off): the fast fold
// emit reads a batch's blendrec pages as one run per DRAM bank, see issue_brec.
#ifndef OL_BREC_BULK
#define OL_BREC_BULK 1
#endif
#ifndef OL_BREC_HALF
#define OL_BREC_HALF (OL_PB * 16u)  // blendrec pages per ring half (== sort_device.cpp)
#endif
// Task #154 (host knob GSPLAT_TT_OL_EMIT_PROF=1, profiling only): the emit
// accumulates wall-clock cycles per part (EP_* below) and, at its end, records
// each total as a Tracy timestamped-data marker (zone names "ep_*", value =
// cycles summed over the launch for this mover). Off: the macros are empty.
#ifndef OL_EMIT_PROF
#define OL_EMIT_PROF 0
#endif
// Task #160 (host knob GSPLAT_TT_OL_EMIT_FAST=0 turns it off): the fast emit
// loop, see FAST_OK below.
#ifndef OL_EMIT_FAST
#define OL_EMIT_FAST 1
#endif
// Task #202 (on by default, GSPLAT_TT_OL_EMIT_TOWN=0 turns it off;
// docs/emit-trisc-own-t200): the fast fold emit's per-record loop runs on the 3
// TRISCs, tile-owned (sort_ol_town.h, ../compute/sort_ol_town_compute.cpp). The
// movers read the pairs and blendrec pages, build per-TRISC lists and write the
// runs the TRISCs fill. Off: no define, the same kernel binary as before.
#ifndef OL_EMIT_TOWN
#define OL_EMIT_TOWN 0
#endif
#if OL_EMIT_TOWN
#include "sort_ol_town.h"
#endif
#if OL_EMIT_PROF
#define EP_NOW() (reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0])
#define EP_T0(v) const uint32_t v = EP_NOW()
#define EP_ADD(acc, t0) (acc) += EP_NOW() - (t0)
#define EP_CNT(acc, n) (acc) += (n)
#else
#define EP_T0(v)
#define EP_ADD(acc, t0)
#define EP_CNT(acc, n)
#endif

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t L1_TILE_SIZE = 32u;
constexpr uint32_t REC_BYTES = 32u;
constexpr uint32_t REC_PAGE_BYTES = 2048u;  // == render_config::kRecPageBytes
constexpr uint32_t REC_PAGE_RECS = REC_PAGE_BYTES / REC_BYTES;  // 64
constexpr uint32_t CNT_BATCH = 32u;  // count-pass pages per read barrier
constexpr uint32_t REC_BATCH = 16u;
// BRISC (mover 0) uses its own copy of every private CB at id + 16.
constexpr uint32_t MOVER0_CB_OFFSET = 16;
// Private CBs (per mover).
constexpr uint32_t CB_GID = 0, CB_TID = 1, CB_KEEP = 2, CB_DEP = 3, CB_H = 4, CB_CUR = 5,
                   CB_BATCH = 6, CB_REC = 7, CB_SCRATCH = 8, CB_WIN = 12, CB_RING = 13;
// The first WIN_PAGES pages of a mover's range stay in L1 (gid, tid, keep
// planes) from the count pass to the emit; pages past the window are re-read
// by the emit, two batches ahead.
constexpr uint32_t WIN_PAGES = OL_WIN_PAGES;
// Shared CBs (one copy per core): CB_ROW holds the core's count row, then its
// base row; CB_PFX stages the prefix pass (num_cores count pages, num_cores
// base pages, the totals page and the padded-totals page).
constexpr uint32_t CB_ROW = 10, CB_PFX = 11;

inline void sem_inc(uint32_t x, uint32_t y, uint32_t sem_id) {
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1u);
}

// pack_rec's tile-local mean when sub_int32 can't answer: sub_int's 64-bit
// path, else the float expression. Out of line: rare on real frames.
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

// Gen (task #298, arg 27 = 2): emit_pairs_diet's Io for making this mover's
// pairs in place. The ring, the staging and the segment table sit in the
// window's keep plane (scratch under the fold until the emit's ones); pages
// below nwin go straight into the window's gid / tid planes, the rest through
// the staging to the pair pages, which the emit re-reads.
template <class LA, class BA, class GA, class TA>
struct GenIo {
    const LA& lofs_acc;
    const BA& box_acc;
    const GA& gids_acc;
    const TA& tids_acc;
    uint32_t ra_lofs, ra_box, st_gid, st_tid, win_gid, win_tid, pg_lo, nwin;
    uint32_t cur;  // page the next slot is for
    void issue_lofs(uint32_t page, uint32_t slot) const {
        noc_async_read(get_noc_addr(page, lofs_acc), ra_lofs + slot * 64u, 64u);
    }
    void issue(uint32_t page, uint32_t slot) const {
        noc_async_read(get_noc_addr(page, lofs_acc), ra_lofs + slot * 64u, 64u);
        noc_async_read(get_noc_addr(page, box_acc), ra_box + slot * 64u, 64u);
    }
    void wait_reads() const {
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
    }
    const uint32_t* lofs_slot(uint32_t slot) const {
        return reinterpret_cast<const uint32_t*>(ra_lofs + slot * 64u);
    }
    const uint32_t* box_slot(uint32_t slot) const {
        return reinterpret_cast<const uint32_t*>(ra_box + slot * 64u);
    }
    uint32_t* gid_slot(uint32_t o) const {
        const uint32_t w = cur - pg_lo;
        return reinterpret_cast<uint32_t*>(w < nwin ? win_gid + w * 64u : st_gid + o * 64u);
    }
    uint32_t* tid_slot(uint32_t o) const {
        const uint32_t w = cur - pg_lo;
        return reinterpret_cast<uint32_t*>(w < nwin ? win_tid + w * 64u : st_tid + o * 64u);
    }
    void write_page(uint32_t page, uint32_t o) {
        asm volatile("" ::: "memory");
        if (page - pg_lo >= nwin) {
            noc_async_write(st_gid + o * 64u, get_noc_addr(page, gids_acc), 64u);
            noc_async_write(st_tid + o * 64u, get_noc_addr(page, tids_acc), 64u);
        }
        cur = page + 1u;
    }
    void writes_flushed() const {
        noc_async_writes_flushed();
        asm volatile("" ::: "memory");
    }
};

}  // namespace

void kernel_main() {
    const uint32_t gids_addr = get_arg_val<uint32_t>(0);
    const uint32_t tids_addr = get_arg_val<uint32_t>(1);
    const uint32_t keep_addr = get_arg_val<uint32_t>(2);
    const uint32_t depth_addr = get_arg_val<uint32_t>(3);
    const uint32_t blendrec_addr = get_arg_val<uint32_t>(4);
    const uint32_t bucket_addr = get_arg_val<uint32_t>(5);
    const uint32_t counts_addr = get_arg_val<uint32_t>(6);
    const uint32_t bases_addr = get_arg_val<uint32_t>(7);
    const uint32_t totals_addr = get_arg_val<uint32_t>(8);
    uint32_t pg_lo = get_arg_val<uint32_t>(9);
    uint32_t pg_hi = get_arg_val<uint32_t>(10);
    uint32_t P = get_arg_val<uint32_t>(11);
    const uint32_t num_tiles = get_arg_val<uint32_t>(12);
    const uint32_t row_pages = get_arg_val<uint32_t>(13);
    const uint32_t core_id = get_arg_val<uint32_t>(14);
    const uint32_t num_cores = get_arg_val<uint32_t>(15);
    const uint32_t tile_cap = get_arg_val<uint32_t>(16);
    const uint32_t tiles_x = get_arg_val<uint32_t>(17);
    const uint32_t mover = get_arg_val<uint32_t>(18);
    const uint32_t sem_counted_id = get_arg_val<uint32_t>(19);
    const uint32_t sem_based_id = get_arg_val<uint32_t>(20);
    const uint32_t sem_arrive1_id = get_arg_val<uint32_t>(21);
    const uint32_t sem_release1_id = get_arg_val<uint32_t>(22);
    const uint32_t sem_arrive2_id = get_arg_val<uint32_t>(23);
    const uint32_t sem_release2_id = get_arg_val<uint32_t>(24);
    const uint32_t coord_x = get_arg_val<uint32_t>(25);
    const uint32_t coord_y = get_arg_val<uint32_t>(26);
    const uint32_t fold_mode = get_arg_val<uint32_t>(27);
    const bool fold = fold_mode != 0u;
    const bool gen = fold_mode == 2u;

    constexpr auto gids_args = TensorAccessorArgs<0>();
    constexpr auto tids_args = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto depth_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto brec_args = TensorAccessorArgs<depth_args.next_compile_time_args_offset()>();
    constexpr auto bucket_args = TensorAccessorArgs<brec_args.next_compile_time_args_offset()>();
    constexpr auto cnt_args = TensorAccessorArgs<bucket_args.next_compile_time_args_offset()>();
    constexpr auto base_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();
    constexpr auto tot_args = TensorAccessorArgs<base_args.next_compile_time_args_offset()>();
    const auto gids_acc = TensorAccessor(gids_args, gids_addr, PAGE_BYTES);
    const auto tids_acc = TensorAccessor(tids_args, tids_addr, PAGE_BYTES);
    const auto keep_acc = TensorAccessor(keep_args, keep_addr, PAGE_BYTES);
    const auto depth_acc = TensorAccessor(depth_args, depth_addr, PAGE_BYTES);
    const auto brec_acc = TensorAccessor(brec_args, blendrec_addr, PAGE_BYTES);
    const auto bucket_acc = TensorAccessor(bucket_args, bucket_addr, REC_PAGE_BYTES);
    const auto cnt_acc = TensorAccessor(cnt_args, counts_addr, PAGE_BYTES);
    const auto base_acc = TensorAccessor(base_args, bases_addr, PAGE_BYTES);
    const auto tot_acc = TensorAccessor(tot_args, totals_addr, PAGE_BYTES);

    const uint32_t cbo = (mover == 0) ? MOVER0_CB_OFFSET : 0u;
    auto hp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_H + cbo));
    auto curp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CUR + cbo));
    const uint32_t row_l1 = get_write_ptr(CB_ROW);
    auto rowp = reinterpret_cast<volatile uint32_t*>(row_l1);
    auto h0p = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_H + MOVER0_CB_OFFSET));
    auto h1p = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_H));

    if (P == 0xFFFFFFFFu) {
        // Early launch: the ta_pairs_P page lands in this mover's H row (the
        // fold reads no H row before NCRISC's count-row read overwrites it).
        const InterleavedAddrGen<true> pctrl{pg_lo, PAGE_BYTES};
        noc_async_read(get_noc_addr(0, pctrl), get_write_ptr(CB_H + cbo), PAGE_BYTES);
        noc_async_read_barrier();
        invalidate_l1_cache();
        P = hp[0];
        uint32_t start = 0, count = 0;
        pfwc_fuse::k2_range_speed(P, pg_hi, get_arg_val<uint32_t>(28), get_arg_val<uint32_t>(29),
                                  &start, &count);
        pg_lo = start;
        pg_hi = start + count;
    }
    const uint32_t npages = pg_hi - pg_lo;
    // Window size rounded down to whole count batches (a batch is all-window
    // or all-fallback).
    const uint32_t nwin_raw = (npages < WIN_PAGES) ? npages : WIN_PAGES;
    const uint32_t nwin = (nwin_raw == npages) ? npages : (nwin_raw / CNT_BATCH) * CNT_BATCH;
    const uint32_t win_gid = get_write_ptr(CB_WIN + cbo);
    const uint32_t win_tid = win_gid + WIN_PAGES * PAGE_BYTES;
    const uint32_t win_keep = win_tid + WIN_PAGES * PAGE_BYTES;
    // Fold: the window's gid and tid planes, read while this mover waits.
    auto fill_window_pages = [&]() {
        for (uint32_t w = 0; w < nwin; w++) {
            noc_async_read(get_noc_addr(pg_lo + w, gids_acc), win_gid + w * PAGE_BYTES, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg_lo + w, tids_acc), win_tid + w * PAGE_BYTES, PAGE_BYTES);
        }
    };
    // OL_FILL_BULK (task #181): an interleaved buffer puts page p + nb right
    // after page p in the same DRAM bank (nb = bank count). So the window is nb
    // runs (pages r, r + nb, r + 2 nb, ...), one read each per plane instead of
    // one 64 B read per page. The runs land bank-major in a staging plane and a
    // copy puts each page at its window slot. Staging: the gid runs go to the
    // tid plane and the tid runs to the keep plane (the fold reads no keep
    // pages; the emit sets its first PB keep pages to ones afterwards), so gid
    // is unpacked first, then tid. Blocking: the read barrier is inside.
    // Returns false (nothing read) when no stride nb <= 16 matches.
    auto fill_window_bulk = [&]() -> bool {
        const uint64_t a0g = get_noc_addr(pg_lo, gids_acc);
        const uint64_t a0t = get_noc_addr(pg_lo, tids_acc);
        uint32_t nb = 0;
        for (uint32_t q = 1; q <= 16u; q++) {
            if (get_noc_addr(pg_lo + q, gids_acc) == a0g + PAGE_BYTES &&
                get_noc_addr(pg_lo + q, tids_acc) == a0t + PAGE_BYTES) {
                nb = q;
                break;
            }
        }
        if (nb == 0u || nwin < nb) return false;
        uint32_t s = 0;
        for (uint32_t r = 0; r < nb; r++) {
            const uint32_t c = (nwin - r + nb - 1u) / nb;
            noc_async_read(get_noc_addr(pg_lo + r, gids_acc), win_tid + s * PAGE_BYTES, c * PAGE_BYTES);
            noc_async_read(get_noc_addr(pg_lo + r, tids_acc), win_keep + s * PAGE_BYTES, c * PAGE_BYTES);
            s += c;
        }
        noc_async_read_barrier();
        auto unpack = [&](uint32_t src, uint32_t dst) {
            auto sp = reinterpret_cast<volatile uint32_t*>(src);
            uint32_t i = 0;
            for (uint32_t r = 0; r < nb; r++) {
                for (uint32_t w = r; w < nwin; w += nb, i++) {
                    auto dp = reinterpret_cast<volatile uint32_t*>(dst + w * PAGE_BYTES);
                    const volatile uint32_t* q = sp + i * ELEMS_PER_PAGE;
                    const uint32_t v0 = q[0], v1 = q[1], v2 = q[2], v3 = q[3], v4 = q[4], v5 = q[5],
                                   v6 = q[6], v7 = q[7];
                    dp[0] = v0; dp[1] = v1; dp[2] = v2; dp[3] = v3;
                    dp[4] = v4; dp[5] = v5; dp[6] = v6; dp[7] = v7;
                    const uint32_t u0 = q[8], u1 = q[9], u2 = q[10], u3 = q[11], u4 = q[12], u5 = q[13],
                                   u6 = q[14], u7 = q[15];
                    dp[8] = u0; dp[9] = u1; dp[10] = u2; dp[11] = u3;
                    dp[12] = u4; dp[13] = u5; dp[14] = u6; dp[15] = u7;
                }
            }
        };
        unpack(win_tid, win_gid);
        unpack(win_keep, win_tid);
#if OL_FILL_BULK == 2
        // Debug (GSPLAT_TT_OL_FILL_BULK=2): re-read every window page one by
        // one, count and repair the pages the bulk fill got wrong.
        {
            const uint32_t scr = get_write_ptr(CB_SCRATCH + cbo);
            auto sv = reinterpret_cast<volatile uint32_t*>(scr);
            uint32_t bad = 0;
            for (uint32_t w = 0; w < nwin; w++) {
                for (uint32_t pl = 0; pl < 2u; pl++) {
                    noc_async_read(get_noc_addr(pg_lo + w, pl ? tids_acc : gids_acc), scr, PAGE_BYTES);
                    noc_async_read_barrier();
                    auto dp = reinterpret_cast<volatile uint32_t*>((pl ? win_tid : win_gid) + w * PAGE_BYTES);
                    bool diff = false;
                    for (uint32_t e = 0; e < ELEMS_PER_PAGE; e++) {
                        if (dp[e] != sv[e]) { diff = true; dp[e] = sv[e]; }
                    }
                    bad += diff ? 1u : 0u;
                }
            }
            if (bad != 0u || core_id == 0u) {
                DPRINT << "FILLCHK core " << core_id << " mover " << mover << " nb " << nb << " nwin "
                       << nwin << " pg_lo " << pg_lo << " bad " << bad << ENDL();
            }
        }
#endif
        return true;
    };
    // Gen (task #298): this mover makes its own pairs (the count-only K2 wrote
    // none) with the K2's walk: reads the pfwc counts table, builds the
    // segment table and walks its page range (pfwc_fuse.h). Args 30 lofs,
    // 31 aabb, 32 pfwc counts table, 33 nseg, 34 the K2's num_tiles.
    auto gen_window = [&]() {
        DeviceZoneScopedN("sort_ol_gen");
        const InterleavedAddrGen<true> lofs_g{get_arg_val<uint32_t>(30), PAGE_BYTES};
        const InterleavedAddrGen<true> box_g{get_arg_val<uint32_t>(31), PAGE_BYTES};
        const InterleavedAddrGen<true> ctab_g{get_arg_val<uint32_t>(32), PAGE_BYTES};
        const uint32_t nseg = get_arg_val<uint32_t>(33);
        constexpr uint32_t RS = pfwc_fuse::RA_SLOTS, OS = pfwc_fuse::OUT_SLOTS;
        static_assert((pfwc_fuse::MAX_SEG + 2u * RS + 2u * OS) * 64u <= WIN_PAGES * PAGE_BYTES,
                      "gen scratch fits the keep plane");
        const uint32_t l1_tab = win_keep;
        for (uint32_t c = 0; c < nseg; c++)
            noc_async_read(get_noc_addr(c, ctab_g), l1_tab + c * PAGE_BYTES, PAGE_BYTES);
        noc_async_read_barrier();
        invalidate_l1_cache();
        auto tab = reinterpret_cast<volatile uint32_t*>(l1_tab);
        uint32_t M = 0, Pa = 0;
        pfwc_fuse::seg_table(tab, nseg, get_arg_val<uint32_t>(34), &M, &Pa);
        const uint32_t ra = l1_tab + pfwc_fuse::MAX_SEG * PAGE_BYTES;
        GenIo<decltype(lofs_g), decltype(box_g), decltype(gids_acc), decltype(tids_acc)> io{
            lofs_g, box_g, gids_acc, tids_acc, ra, ra + RS * 64u, ra + 2u * RS * 64u,
            ra + (2u * RS + OS) * 64u, win_gid, win_tid, pg_lo, nwin, pg_lo};
        pfwc_fuse::emit_pairs_diet<false>(tab, nseg, P, tiles_x, pg_lo, npages, io,
                                          static_cast<uint32_t*>(nullptr));
        noc_async_write_barrier();  // the emit re-reads the pages past nwin
    };
    auto fill_window = [&]() {
        if (gen) {
            gen_window();
            return;
        }
        if (OL_FILL_BULK && fill_window_bulk()) return;
        fill_window_pages();
    };

    // ── 1. count this mover's kept pairs per tile ──────────────────────────
    if (!fold) {
        DeviceZoneScopedN("sort_ol_count");
        const uint32_t btid_l1 = get_write_ptr(CB_BATCH + cbo);
        const uint32_t bkeep_l1 = btid_l1 + CNT_BATCH * PAGE_BYTES;
        auto btidp = reinterpret_cast<volatile int32_t*>(btid_l1);
        auto bkeepp = reinterpret_cast<volatile int32_t*>(bkeep_l1);
        for (uint32_t t = 0; t < num_tiles; t++) hp[t] = 0;
        for (uint32_t pg0 = pg_lo; pg0 < pg_hi;) {
            const uint32_t nb = (pg_hi - pg0 < CNT_BATCH) ? (pg_hi - pg0) : CNT_BATCH;
            const bool in_win = (pg0 - pg_lo) + nb <= nwin;  // batches never straddle nwin
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t w = pg0 - pg_lo + b;
                if (in_win) {
                    noc_async_read(get_noc_addr(pg0 + b, gids_acc), win_gid + w * PAGE_BYTES, PAGE_BYTES);
                    noc_async_read(get_noc_addr(pg0 + b, tids_acc), win_tid + w * PAGE_BYTES, PAGE_BYTES);
                    noc_async_read(get_noc_addr(pg0 + b, keep_acc), win_keep + w * PAGE_BYTES, PAGE_BYTES);
                } else {
                    noc_async_read(get_noc_addr(pg0 + b, tids_acc), btid_l1 + b * PAGE_BYTES, PAGE_BYTES);
                    noc_async_read(get_noc_addr(pg0 + b, keep_acc), bkeep_l1 + b * PAGE_BYTES, PAGE_BYTES);
                }
            }
            noc_async_read_barrier();
            for (uint32_t b = 0; b < nb; b++) {
                const uint32_t w = pg0 - pg_lo + b;
                auto tp = in_win ? reinterpret_cast<volatile int32_t*>(win_tid + w * PAGE_BYTES)
                                 : btidp + b * ELEMS_PER_PAGE;
                auto kp = in_win ? reinterpret_cast<volatile int32_t*>(win_keep + w * PAGE_BYTES)
                                 : bkeepp + b * ELEMS_PER_PAGE;
                for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                    if ((pg0 + b) * ELEMS_PER_PAGE + j >= P) break;
                    if (kp[j] == 0) continue;
                    hp[static_cast<uint32_t>(tp[j])]++;
                }
            }
            pg0 += nb;
        }
    }

    auto sem_counted = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_counted_id));
    auto sem_based = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_based_id));
    if (mover == 1) {
        if (fold) {
            // BRISC's count row (the K2's row 2 * core_id) for the cursors.
            const uint32_t h_l1 = get_write_ptr(CB_H + cbo);
            for (uint32_t q = 0; q < row_pages; q++) {
                noc_async_read(get_noc_addr(core_id * 2u * row_pages + q, cnt_acc),
                               h_l1 + q * PAGE_BYTES, PAGE_BYTES);
            }
            DeviceZoneScopedN("sort_ol_fill");  // task #181: fill time and GB/s
            fill_window();
            noc_async_read_barrier();
        } else {
            // Hand the count row to mover 0.
            asm volatile("fence" ::: "memory");
            noc_semaphore_set(sem_counted, 1u);
        }
        // Wait for the bases.
        noc_semaphore_wait(sem_based, 1u);
        noc_semaphore_set(sem_based, 0u);  // re-arm for the next launch
        invalidate_l1_cache();
    } else {
        if (!fold) {
            noc_semaphore_wait(sem_counted, 1u);
            noc_semaphore_set(sem_counted, 0u);
            invalidate_l1_cache();
        }
        const uint32_t row_span = row_pages * ELEMS_PER_PAGE;
        const bool coordinator = (core_id == 0u);
        // fill: fill_window() while waiting (the coordinator after releasing).
        // Profiler builds (task #181): BRISC's fill in its own zone, read
        // barrier included (else step 4's read barrier waits for it).
        auto fill_b = [&]() {
#if defined(PROFILE_KERNEL)
            DeviceZoneScopedN("sort_ol_fillb");
            fill_window();
            noc_async_read_barrier();
#else
            fill_window();
#endif
        };
        auto barrier = [&](uint32_t arrive_id, uint32_t release_id, bool fill) {
            DeviceZoneScopedN("sort_ol_barrier");
            noc_async_write_barrier();  // this core's DRAM writes are visible first
            if (coordinator) {
                auto arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_id));
                noc_semaphore_wait(arrive, num_cores - 1u);
                noc_semaphore_set(arrive, 0u);
                for (uint32_t r = 1; r < num_cores; r++) {
                    const uint32_t xy = get_arg_val<uint32_t>(35 + r);
                    sem_inc(xy & 0xFFFFu, xy >> 16, release_id);
                }
                noc_async_atomic_barrier();
                if (fill) fill_b();
            } else {
                sem_inc(coord_x, coord_y, arrive_id);
                noc_async_atomic_barrier();
                if (fill) fill_b();
                auto release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_id));
                noc_semaphore_wait(release, 1u);
                noc_semaphore_set(release, 0u);
            }
            invalidate_l1_cache();
        };

        // ── 1b. the core's count row (both movers) to DRAM ─────────────────
        // (fold: the K2 wrote both movers' rows before this launch)
        if (!fold) {
            for (uint32_t t = 0; t < row_span; t++) rowp[t] = (t < num_tiles) ? h0p[t] + h1p[t] : 0u;
            asm volatile("fence" ::: "memory");
            for (uint32_t q = 0; q < row_pages; q++) {
                noc_async_write(row_l1 + q * PAGE_BYTES,
                                get_noc_addr(core_id * row_pages + q, cnt_acc), PAGE_BYTES);
            }
            barrier(sem_arrive1_id, sem_release1_id, false);
        }

        // ── 3. column prefix over cores for the pages this core owns ───────
        {
            DeviceZoneScopedN("sort_ol_prefix");
            const uint32_t pfx = get_write_ptr(CB_PFX);
            const uint32_t bst = pfx + num_cores * PAGE_BYTES;
            const uint32_t tst = bst + num_cores * PAGE_BYTES;
            auto cin = reinterpret_cast<volatile uint32_t*>(pfx);
            auto bout = reinterpret_cast<volatile uint32_t*>(bst);
            auto tout = reinterpret_cast<volatile uint32_t*>(tst);
            for (uint32_t p = core_id; p < row_pages; p += num_cores) {
                // Fold: BRISC's row page to cin, NCRISC's to bout (each word
                // is read once, just before its base overwrites it).
                for (uint32_t r = 0; r < num_cores; r++) {
                    if (fold) {
                        noc_async_read(get_noc_addr(2u * r * row_pages + p, cnt_acc),
                                       pfx + r * PAGE_BYTES, PAGE_BYTES);
                        noc_async_read(get_noc_addr((2u * r + 1u) * row_pages + p, cnt_acc),
                                       bst + r * PAGE_BYTES, PAGE_BYTES);
                    } else {
                        noc_async_read(get_noc_addr(r * row_pages + p, cnt_acc),
                                       pfx + r * PAGE_BYTES, PAGE_BYTES);
                    }
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                    uint32_t acc = 0, pad = 0;
                    for (uint32_t r = 0; r < num_cores; r++) {
                        const uint32_t h =
                            cin[r * ELEMS_PER_PAGE + j] + (fold ? bout[r * ELEMS_PER_PAGE + j] : 0u);
                        bout[r * ELEMS_PER_PAGE + j] = acc;
                        acc += h;
                        pad += (h + ELEMS_PER_PAGE - 1u) & ~(ELEMS_PER_PAGE - 1u);
                    }
                    tout[j] = acc;
                    tout[ELEMS_PER_PAGE + j] = pad;
                }
                asm volatile("fence" ::: "memory");
                for (uint32_t r = 0; r < num_cores; r++) {
                    noc_async_write(bst + r * PAGE_BYTES, get_noc_addr(r * row_pages + p, base_acc),
                                    PAGE_BYTES);
                }
                noc_async_write(tst, get_noc_addr(p, tot_acc), PAGE_BYTES);
                noc_async_write(tst + PAGE_BYTES, get_noc_addr(row_pages + p, tot_acc), PAGE_BYTES);
                noc_async_write_barrier();  // staging is reused by the next page
            }
        }
        barrier(sem_arrive2_id, sem_release2_id, fold);

        // ── 4. this core's base row (first slot of its records per tile) ───
        for (uint32_t q = 0; q < row_pages; q++) {
            noc_async_read(get_noc_addr(core_id * row_pages + q, base_acc), row_l1 + q * PAGE_BYTES,
                           PAGE_BYTES);
        }
        noc_async_read_barrier();
        asm volatile("fence" ::: "memory");
        noc_semaphore_set(sem_based, 1u);
    }
    // Cursors: BRISC's records first, then NCRISC's, inside the core's chunk
    // (fold: NCRISC read BRISC's count row into its own CB_H).
    {
        volatile uint32_t* h0 = fold ? hp : h0p;
        for (uint32_t t = 0; t < num_tiles; t++) curp[t] = rowp[t] + ((mover == 1) ? h0[t] : 0u);
    }

    // ── 5. emit: pack each kept pair's 32B record into its bucket slot ─────
    // Task #124 (one-launch sort v2): #100's emit ported (sort_bin.cpp
    // sub-pass 2). OL_PB pair pages per batch: batch k+1's blendrec reads (and
    // batch k+2's pair pages, past the window) are in flight while batch k
    // packs. OL_RING: records stage in per-tile runs (sort_onelaunch_algo.h)
    // and leave L1 as one write per run, not one 32 B write per record.
    // OL_PB=1 OL_RING=0 is the v1 emit's batching and write pattern. The bytes
    // written are the same for every setting.
    DeviceZoneScopedN("sort_ol_emit");
#if OL_EMIT_PROF
    // Cycles: prologue (ring starts + first reads), read barrier in the loop,
    // blendrec scan + read issue, pair read issue, process_batch (all of it),
    // its writes-flushed waits, its run write issues, the tail drain, the final
    // write barrier. Counts: records, blendrec pages, batches.
    uint32_t ep_pro = 0, ep_rdw = 0, ep_brec = 0, ep_pairs = 0, ep_proc = 0, ep_wfl = 0, ep_wiss = 0,
             ep_drain = 0, ep_wbar = 0, ep_nrec = 0, ep_npf = 0, ep_nb = 0, ep_ncold = 0, ep_nrun = 0,
             ep_nbk = 0, ep_bpg = 0;
    EP_T0(ep_t_pro);
#endif
    constexpr uint32_t PB = OL_PB;
    constexpr uint32_t BATCH_ELEMS = PB * ELEMS_PER_PAGE;
    constexpr uint32_t R = OL_RING;
    constexpr bool PUBOC = EMIT_PUBOC != 0u;
    static_assert(PB >= 1u && PB <= 16u && (PB & (PB - 1u)) == 0u && CNT_BATCH % PB == 0u,
                  "OL_PB: power of two <= 16 (batches never straddle the window)");
    static_assert(R == 0u || ((R & (R - 1u)) == 0u && R <= 16u && REC_PAGE_RECS % (R ? R : 1u) == 0u),
                  "OL_RING: power of two dividing the record page");
    const bool ring_on = (R != 0u) && num_tiles <= OL_RING_TILES && (tile_cap % REC_PAGE_RECS) == 0u;

    const bool tx_is_pow2 = (tiles_x != 0u) && ((tiles_x & (tiles_x - 1u)) == 0u);
    uint32_t tx_shift = 0u;
    if (tx_is_pow2) {
        uint32_t v = tiles_x;
        while (v > 1u) { v >>= 1; tx_shift++; }
    }
    const uint32_t tx_mask = tiles_x - 1u;

    // Pair staging for pages past the window: buffer b of 3 at plane + b*PB*64.
    const uint32_t gid_l1 = get_write_ptr(CB_GID + cbo);
    const uint32_t tid_l1 = get_write_ptr(CB_TID + cbo);
    const uint32_t keep_l1 = get_write_ptr(CB_KEEP + cbo);
    const uint32_t dep_l1 = get_write_ptr(CB_DEP + cbo);
    auto depp = reinterpret_cast<volatile uint32_t*>(dep_l1);
    constexpr uint32_t BREC_HALF = OL_BREC_HALF;
    static_assert(BREC_HALF >= BATCH_ELEMS, "OL_BREC_HALF: a ring half holds a page per pair of a batch");
    const uint32_t rec_cache_l1 = get_write_ptr(CB_REC + cbo);  // 2 x BREC_HALF blendrec pages
    const uint32_t l1_scratch = get_write_ptr(CB_SCRATCH + cbo);
    volatile uint32_t* cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1);
    int32_t blendrec_cached_g = -1;
    int32_t dep_cached_page = -1;

    // OL_RING=0: REC_BATCH records stage in l1_scratch, one 32 B write each.
    uint32_t brec_slot[REC_BATCH];
    uint32_t nbrec = 0;
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        for (uint32_t b = 0; b < nbrec; b++) {
            const uint32_t slot = brec_slot[b];
            if (slot == 0xFFFFFFFFu) continue;  // over capacity: dropped, host fails the frame
            noc_async_write(l1_scratch + b * REC_BYTES,
                            get_noc_addr(slot / REC_PAGE_RECS, bucket_acc) +
                                (slot % REC_PAGE_RECS) * REC_BYTES,
                            REC_BYTES);
        }
        noc_async_writes_flushed();
        nbrec = 0;
    };

    // OL_RING: tile t's run at ring_l1 + (t*R + c%R)*32, its first cursor of
    // this mover in startp[t].
    const uint32_t ring_l1 = (R != 0u) ? get_write_ptr(CB_RING + cbo) : 0u;
    auto startp = reinterpret_cast<volatile uint32_t*>(ring_l1 + OL_RING_TILES * R * REC_BYTES);
    // Task #160 fast emit (the default: rings, PUBOC, tiles_x a power of two):
    // one loop with register locals and the sub_int fast path inlined. Same
    // records, same slots, same writes. Its per-tile cursors are curp itself
    // (CB_CUR, L1), through a plain pointer: a 4 KB stack copy left kernel_main
    // ~0 B of stack margin (task #186), task #187 moved it here.
    constexpr bool FAST_OK = OL_EMIT_FAST && PUBOC && R != 0u;
    const bool fast = FAST_OK && ring_on && tx_is_pow2;
    uint32_t* const cur_lm = reinterpret_cast<uint32_t*>(get_write_ptr(CB_CUR + cbo));
    if (ring_on) {
        for (uint32_t t = 0; t < num_tiles; t++) startp[t] = curp[t];
    }
    const uint32_t tile_pages = tile_cap / REC_PAGE_RECS;
    auto flush_run = [&](uint32_t t, uint32_t last) {
        const uint32_t s0 = sort_ol::ring_run_start(startp[t], last, R);
        noc_async_write(ring_l1 + (t * R + (s0 & (R - 1u))) * REC_BYTES,
                        get_noc_addr(t * tile_pages + s0 / REC_PAGE_RECS, bucket_acc) +
                            (s0 % REC_PAGE_RECS) * REC_BYTES,
                        (last + 1u - s0) * REC_BYTES);
    };

    // Per-gaussian invariant words of the 32B record (== sort_bin.cpp); the
    // tile-local mean y only changes with the tile row.
    uint32_t inv_cov0 = 0, inv_cov1 = 0, inv_cov2 = 0, inv_depth = 0, inv_opr = 0, inv_cgb = 0;
    float inv_mx = 0.0f, inv_my = 0.0f;
    uint32_t inv_mx_bits = 0, inv_my_bits = 0;
    uint32_t c_ty = 0xFFFFFFFFu, c_my_bits = 0;
    auto pack_invariants = [&](uint32_t depth_key) {
        inv_cov0 = cachep[0];
        inv_cov1 = cachep[1];
        inv_cov2 = cachep[2];
        inv_depth = depth_key;
        inv_mx_bits = cachep[3];
        inv_my_bits = cachep[4];
        __builtin_memcpy(&inv_mx, &inv_mx_bits, 4);
        __builtin_memcpy(&inv_my, &inv_my_bits, 4);
        c_ty = 0xFFFFFFFFu;
        if constexpr (PUBOC) {
            inv_opr = cachep[10];
            inv_cgb = cachep[11];
        } else {
            inv_opr = sort_bin_fp32::to_unorm16(cachep[5]) | (sort_bin_fp32::to_unorm16(cachep[6]) << 16);
            inv_cgb = sort_bin_fp32::to_unorm16(cachep[7]) | (sort_bin_fp32::to_unorm16(cachep[8]) << 16);
        }
    };
    auto pack_rec = [&](volatile uint32_t* p32, uint32_t tt) {
        const uint32_t txi = tx_is_pow2 ? (tt & tx_mask) : (tt % tiles_x);
        const uint32_t tyi = tx_is_pow2 ? (tt >> tx_shift) : (tt / tiles_x);
        uint32_t mx_bits;
        if (!sort_bin_fp32::sub_int(inv_mx_bits, txi * L1_TILE_SIZE, &mx_bits)) {
            const float mx = inv_mx - static_cast<float>(txi * L1_TILE_SIZE);
            __builtin_memcpy(&mx_bits, &mx, 4);
        }
        if (tyi != c_ty) {
            c_ty = tyi;
            if (!sort_bin_fp32::sub_int(inv_my_bits, tyi * L1_TILE_SIZE, &c_my_bits)) {
                const float my = inv_my - static_cast<float>(tyi * L1_TILE_SIZE);
                __builtin_memcpy(&c_my_bits, &my, 4);
            }
        }
        p32[0] = inv_cov0;
        p32[1] = inv_cov1;
        p32[2] = inv_cov2;
        p32[3] = inv_depth;
        p32[4] = mx_bits;
        p32[5] = c_my_bits;
        p32[6] = inv_opr;
        p32[7] = inv_cgb;
    };

    // Batch k: pages [pg_lo + k*PB, +nb). Window batches read their planes in
    // place; later ones use pair buffer k % 3, issued two batches ahead. Fold:
    // keep is all ones, so every batch's keep plane is the window's first PB
    // keep pages set to 1 here (no keep page is read).
    if (fold) {
        auto ones = reinterpret_cast<volatile uint32_t*>(win_keep);
        for (uint32_t i = 0; i < PB * ELEMS_PER_PAGE; i++) ones[i] = 1u;
    }
    struct Planes { volatile int32_t* g; volatile int32_t* t; volatile int32_t* k; };
    auto planes = [&](uint32_t k) -> Planes {
        const uint32_t w = k * PB;
        if (w < nwin) {
            return {reinterpret_cast<volatile int32_t*>(win_gid + w * PAGE_BYTES),
                    reinterpret_cast<volatile int32_t*>(win_tid + w * PAGE_BYTES),
                    reinterpret_cast<volatile int32_t*>(win_keep + (fold ? 0u : w * PAGE_BYTES))};
        }
        const uint32_t o = (k % 3u) * PB * PAGE_BYTES;
        return {reinterpret_cast<volatile int32_t*>(gid_l1 + o),
                reinterpret_cast<volatile int32_t*>(tid_l1 + o),
                reinterpret_cast<volatile int32_t*>(fold ? win_keep : keep_l1 + o)};
    };
    const uint32_t nbatch = (npages + PB - 1u) / PB;
    auto batch_pages = [&](uint32_t k) { return (npages - k * PB < PB) ? (npages - k * PB) : PB; };
    auto issue_pairs = [&](uint32_t k) {
        if (k * PB < nwin) return;  // already in the window
        const uint32_t pg0 = pg_lo + k * PB;
        const uint32_t o = (k % 3u) * PB * PAGE_BYTES;
        for (uint32_t b = 0; b < batch_pages(k); b++) {
            noc_async_read(get_noc_addr(pg0 + b, gids_acc), gid_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg0 + b, tids_acc), tid_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
            if (!fold) {
                noc_async_read(get_noc_addr(pg0 + b, keep_acc), keep_l1 + o + b * PAGE_BYTES,
                               PAGE_BYTES);
            }
        }
    };
    // OL_BREC_BULK (task #196): under the fold the pairs are gaussian-major and
    // all kept, so a batch's g run from its first to its last g is near-dense
    // (pfwc segments are dense). When the run fits a ring half, its pages are
    // nb reads, one per DRAM bank (page ga + j lands at slot (j % nb) * bk_sf +
    // j / nb), instead of one 64 B read per distinct g: those were
    // request-bound, and their issue stalled most on the NoC hot spots (BRISC
    // rows y=2,3, NCRISC columns x=14,15; docs/emit-imbalance-t196).
    // bk_n[h]: pages of half h's bulk run, 0 = per-g pages.
    uint32_t bk_nb = 0;
    if (OL_BREC_BULK && fast && fold) {
        const uint64_t a0 = get_noc_addr(0u, brec_acc);
        for (uint32_t q = 1; q <= 16u; q++) {
            if (get_noc_addr(q, brec_acc) == a0 + PAGE_BYTES) {
                bk_nb = q;
                break;
            }
        }
    }
    const uint32_t bk_sf = (bk_nb != 0u) ? BREC_HALF / bk_nb : 0u;
    const uint32_t bk_cap = bk_sf * bk_nb;
    const uint32_t bk_run = bk_sf * PAGE_BYTES, bk_wrap = bk_cap * PAGE_BYTES - PAGE_BYTES;
#if OL_EMIT_TOWN
    constexpr uint32_t BK_SLOTS = sort_ol_town::SLOTS;  // TOWN: blendrec slot k % SLOTS
#else
    constexpr uint32_t BK_SLOTS = 2u;
#endif
    uint32_t bk_a[BK_SLOTS] = {}, bk_n[BK_SLOTS] = {};
    // A g outside its batch's bulk run (only if the pairs were not g-sorted):
    // one blocking read into the scratch page (unused with rings).
    auto brec_now = [&](int32_t g) -> const uint32_t* {
        noc_async_read(get_noc_addr(static_cast<uint32_t>(g), brec_acc), l1_scratch, PAGE_BYTES);
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        EP_CNT(ep_npf, 1u);
        return reinterpret_cast<const uint32_t*>(l1_scratch);
    };
    // One blendrec page per run of equal kept g (scan_g carries the last kept
    // g across batches, the same test process_batch uses) into ring half h.
    int32_t scan_g = -1;
    auto issue_brec = [&](uint32_t k, uint32_t h) {
        const Planes pl = planes(k);
        const uint32_t dst0 = rec_cache_l1 + h * BREC_HALF * PAGE_BYTES;
        const uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
        const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
        bk_n[h] = 0u;
        if (bk_nb != 0u && p0 < P) {
            const uint32_t m = (p0 + n_el > P) ? P - p0 : n_el;
            const int32_t ga = pl.g[0], gb = pl.g[m - 1u];
            const uint32_t n = static_cast<uint32_t>(gb - ga) + 1u;
            if (gb >= ga && n <= bk_cap) {
                const uint32_t q = n / bk_nb, rem = n - q * bk_nb;
                const uint32_t nr = (n < bk_nb) ? n : bk_nb;
                for (uint32_t r = 0; r < nr; r++) {
                    noc_async_read(get_noc_addr(static_cast<uint32_t>(ga) + r, brec_acc),
                                   dst0 + r * bk_sf * PAGE_BYTES, (q + (r < rem ? 1u : 0u)) * PAGE_BYTES);
                }
                bk_a[h] = static_cast<uint32_t>(ga);
                bk_n[h] = n;
                scan_g = gb;
                EP_CNT(ep_nbk, 1u);
                EP_CNT(ep_bpg, n);
                return;
            }
        }
        uint32_t n_pf = 0;
        for (uint32_t j = 0; j < n_el; j++) {
            if (p0 + j >= P) break;
            if (pl.k[j] == 0) continue;
            const int32_t gj = pl.g[j];
            if (gj != scan_g) {
                noc_async_read(get_noc_addr(static_cast<uint32_t>(gj), brec_acc), dst0 + n_pf * PAGE_BYTES,
                               PAGE_BYTES);
                n_pf++;
                scan_g = gj;
            }
        }
        EP_CNT(ep_npf, n_pf);
    };
    auto process_batch = [&](uint32_t k, uint32_t h) {
        const Planes pl = planes(k);
        const uint32_t ring0 = rec_cache_l1 + h * BREC_HALF * PAGE_BYTES;
        const uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
        const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
        uint32_t rec_slot = 0;
        for (uint32_t j = 0; j < n_el; j++) {
            if (p0 + j >= P) break;
            if (pl.k[j] == 0) continue;
            const uint32_t g = static_cast<uint32_t>(pl.g[j]);
            const uint32_t t = static_cast<uint32_t>(pl.t[j]);
            if (static_cast<int32_t>(g) != blendrec_cached_g) {
                cachep = reinterpret_cast<volatile uint32_t*>(ring0 + rec_slot * PAGE_BYTES);
                rec_slot++;
                blendrec_cached_g = static_cast<int32_t>(g);
                uint32_t key;
                if constexpr (PUBOC) {
                    key = cachep[12];
                } else {
                    const int32_t dpg = static_cast<int32_t>(g / ELEMS_PER_PAGE);
                    if (dpg != dep_cached_page) {
                        noc_async_read(get_noc_addr(static_cast<uint32_t>(dpg), depth_acc), dep_l1, PAGE_BYTES);
                        noc_async_read_barrier();
                        dep_cached_page = dpg;
                    }
                    key = depp[g % ELEMS_PER_PAGE];
                }
                pack_invariants(key);
            }
            const uint32_t c = curp[t];
            curp[t] = c + 1u;
            EP_CNT(ep_nrec, 1u);
            if (ring_on) {
                if (c < tile_cap) {  // past capacity: dropped, host fails the frame
                    const uint32_t ri = c & (R - 1u);
                    // Entry 0 starts a new run: the previous run of this tile
                    // must have left L1.
                    if (ri == 0u) {
                        EP_T0(ep_t);
                        noc_async_writes_flushed();
                        EP_ADD(ep_wfl, ep_t);
                    }
                    pack_rec(reinterpret_cast<volatile uint32_t*>(ring_l1 + (t * R + ri) * REC_BYTES), t);
                    if (ri == R - 1u) {
                        EP_T0(ep_t);
                        flush_run(t, c);
                        EP_ADD(ep_wiss, ep_t);
                    }
                }
            } else {
                const uint32_t slot = (c < tile_cap) ? t * tile_cap + c : 0xFFFFFFFFu;
                brec_slot[nbrec] = slot;
                if (slot != 0xFFFFFFFFu) {
                    pack_rec(reinterpret_cast<volatile uint32_t*>(l1_scratch + nbrec * REC_BYTES), t);
                }
                nbrec++;
                if (nbrec == REC_BATCH) flush_recs();
            }
        }
    };

    // Fast-path state, carried across batches (a run of equal g can span two).
    int32_t f_g = -1;
    uint32_t f_cov0 = 0, f_cov1 = 0, f_cov2 = 0, f_dep = 0, f_mx = 0, f_my = 0, f_opr = 0, f_cgb = 0;
    uint32_t f_ty = 0xFFFFFFFFu, f_myt = 0;
#if OL_EMIT_TOWN
    // Task #202: publish this stream to the TRISCs (nb = 0 when the TOWN path
    // does not apply: they then only acknowledge). Batch k: its owner lists and
    // blendrec pages in slot k % SLOTS, published by READY = k + 1 once its
    // reads landed; the slot is refilled once every TRISC finished batch
    // k - SLOTS. The mover writes the runs the TRISCs queue (service), then
    // marks the tile's ring free (fl). Mover parts under OL_EMIT_PROF: ep_brec
    // = list build + reads issue, ep_wfl = slot waits, ep_wiss = all queue
    // service, ep_proc = waiting for the TRISCs after the last batch, ep_pairs
    // = their final acknowledgement; ep_nrun = runs from the queues.
    namespace tw = sort_ol_town;
    static_assert(BATCH_ELEMS <= tw::LIST_MAX && OL_RING_TILES <= tw::TILES, "TOWN list / fl sizes");
    static_assert(BREC_HALF * PAGE_BYTES <= 65536u, "TOWN: blendrec page offsets in 16 bits");
    const uint32_t twn = get_write_ptr(tw::CB_TWN + cbo);
    volatile uint32_t* const twh = reinterpret_cast<volatile uint32_t*>(twn);
    volatile uint32_t* const tfl = reinterpret_cast<volatile uint32_t*>(twn + tw::FL_OFF);
    const bool town = fast && fold && num_tiles <= tw::TILES && tile_cap <= (1u << 22);
    twh[tw::H_NB] = town ? nbatch : 0u;
    twh[tw::H_READY] = 0u;
    twh[tw::H_CAP] = tile_cap;
    twh[tw::H_MSK] = tx_mask;
    twh[tw::H_SH] = tx_shift;
    twh[tw::H_RING] = ring_l1;
    twh[tw::H_CUR] = get_write_ptr(CB_CUR + cbo);  // == cur_lm
    for (uint32_t i = 0; i < tw::NT; i++) {
        twh[tw::H_FIN + i] = 0u;
        twh[tw::H_DONE + 4u * i] = 0u;
        twh[tw::H_QWP + 4u * i] = 0u;
        twh[tw::H_QRD + 4u * i] = 0u;
    }
    if (town) {
        for (uint32_t t = 0; t < num_tiles; t++) tfl[t] = startp[t];
    }
    asm volatile("fence" ::: "memory");
    twh[tw::H_GO2] = tw::MAGIC2;
    asm volatile("fence" ::: "memory");
    twh[tw::H_GO] = tw::MAGIC;
    auto tw_spin = [&]() {
        for (uint32_t i = 0; i < 8u; i++) asm volatile("nop");
    };
    if (town) {
        uint32_t q_rd[tw::NT] = {0u, 0u, 0u};
        // The TRISCs' run words: write the runs, wait until they left L1,
        // then mark the tiles' rings free and the words consumed.
        auto service = [&]() -> bool {
            uint32_t wp[tw::NT];
            bool any = false;
            invalidate_l1_cache();
            for (uint32_t i = 0; i < tw::NT; i++) {
                wp[i] = twh[tw::H_QWP + 4u * i];
                any = any || wp[i] != q_rd[i];
            }
            if (!any) return false;
            EP_T0(ep_t);
            asm volatile("fence" ::: "memory");
            for (uint32_t i = 0; i < tw::NT; i++) {
                auto q = reinterpret_cast<const volatile uint32_t*>(twn + tw::q_off(i));
                for (uint32_t j = q_rd[i]; j != wp[i]; j++) {
                    const uint32_t w = q[j & (tw::QCAP - 1u)];
                    flush_run(tw::run_tile(w), tw::run_last(w));
                }
            }
            noc_async_writes_flushed();
            for (uint32_t i = 0; i < tw::NT; i++) {
                auto q = reinterpret_cast<const volatile uint32_t*>(twn + tw::q_off(i));
                for (uint32_t j = q_rd[i]; j != wp[i]; j++) {
                    const uint32_t w = q[j & (tw::QCAP - 1u)];
                    tfl[tw::run_tile(w)] = tw::run_last(w) + 1u;
                }
                EP_CNT(ep_nrun, wp[i] - q_rd[i]);
            }
            asm volatile("fence" ::: "memory");
            for (uint32_t i = 0; i < tw::NT; i++) {
                if (wp[i] != q_rd[i]) {
                    q_rd[i] = wp[i];
                    twh[tw::H_QRD + 4u * i] = wp[i];
                }
            }
            EP_ADD(ep_wiss, ep_t);
            return true;
        };
        auto min_done = [&]() -> uint32_t {
            invalidate_l1_cache();
            uint32_t m = twh[tw::H_DONE];
            const uint32_t d1 = twh[tw::H_DONE + 4u], d2 = twh[tw::H_DONE + 8u];
            if (d1 < m) m = d1;
            if (d2 < m) m = d2;
            return m;
        };
        // Batch k's lists in slot s: per pair t << 16 | the byte offset of its
        // g's blendrec page in slot s (bulk run: the fast loop's page stepping;
        // per-g pages: one page per run of equal g from the batch start, as
        // issue_brec reads them with scan_g = -1). Fold: every pair is kept.
        // False (nothing published): a g outside the bulk run.
        auto build_lists = [&](uint32_t k, uint32_t s) -> bool {
            const Planes pl = planes(k);
            const int32_t* gp = const_cast<const int32_t*>(pl.g);
            const uint32_t* tp = reinterpret_cast<const uint32_t*>(const_cast<const int32_t*>(pl.t));
            const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
            uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
            if (p0 + n_el > P) n_el = (P > p0) ? P - p0 : 0u;
            uint32_t* const l0 = reinterpret_cast<uint32_t*>(twn + tw::list_off(s, 0u));
            uint32_t* lw[tw::NT] = {l0, l0 + tw::LIST_MAX, l0 + 2u * tw::LIST_MAX};
            const uint32_t ba = bk_a[s], bn = bk_n[s];
            uint32_t jl = 0, jr = 0, off = 0, npg = 0;
            int32_t g_c = -1;
            for (uint32_t j = 0; j < n_el; j++) {
                const int32_t g = gp[j];
                const uint32_t t = tp[j];
                if (g != g_c) {
                    g_c = g;
                    if (bn != 0u) {
                        const uint32_t jg = static_cast<uint32_t>(g) - ba;
                        if (jg >= bn) return false;
                        if (jg < jl) {
                            jl = 0;
                            jr = 0;
                            off = 0;
                        }
                        for (; jl != jg; jl++) {
                            off += bk_run;
                            if (++jr == bk_nb) {
                                jr = 0;
                                off -= bk_wrap;
                            }
                        }
                    } else {
                        off = npg * PAGE_BYTES;
                        npg++;
                    }
                }
                *lw[tw::owner(t)]++ = tw::entry(t, off);
            }
            auto desc = reinterpret_cast<volatile uint32_t*>(twn + tw::desc_off(s));
            desc[0] = static_cast<uint32_t>(lw[0] - l0);
            desc[1] = static_cast<uint32_t>(lw[1] - (l0 + tw::LIST_MAX));
            desc[2] = static_cast<uint32_t>(lw[2] - (l0 + 2u * tw::LIST_MAX));
            desc[3] = rec_cache_l1 + s * BREC_HALF * PAGE_BYTES;
            return true;
        };
        // Batch k's blendrec pages and lists into slot k % SLOTS. A g outside
        // the bulk run: wait for the run, re-read the batch one page per g.
        auto issue_town = [&](uint32_t k) {
            const uint32_t s = k % tw::SLOTS;
            scan_g = -1;
            issue_brec(k, s);
            if (build_lists(k, s)) return;
            noc_async_read_barrier();
            const Planes pl = planes(k);
            const uint32_t dst0 = rec_cache_l1 + s * BREC_HALF * PAGE_BYTES;
            const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
            const uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
            int32_t sg = -1;
            uint32_t n_pf = 0;
            for (uint32_t j = 0; j < n_el; j++) {
                if (p0 + j >= P) break;
                const int32_t gj = pl.g[j];
                if (gj != sg) {
                    noc_async_read(get_noc_addr(static_cast<uint32_t>(gj), brec_acc), dst0 + n_pf * PAGE_BYTES,
                                   PAGE_BYTES);
                    n_pf++;
                    sg = gj;
                }
            }
            EP_CNT(ep_npf, n_pf);
            bk_n[s] = 0u;
            build_lists(k, s);
        };
        if (nbatch > 0) {
            issue_pairs(0);
            noc_async_read_barrier();
            issue_town(0);
            if (nbatch > 1) issue_pairs(1);
            EP_ADD(ep_pro, ep_t_pro);
            for (uint32_t k = 0; k < nbatch; k++) {
                EP_T0(ep_t1);
                noc_async_read_barrier();  // blendrec of batch k, pairs of batch k+1
                EP_ADD(ep_rdw, ep_t1);
                asm volatile("fence" ::: "memory");
                twh[tw::H_READY] = k + 1u;
                if (k + 1u < nbatch) {
                    EP_T0(ep_t2);
                    while (min_done() + tw::SLOTS < k + 2u) {
                        if (!service()) tw_spin();
                    }
                    EP_T0(ep_t3);
                    EP_ADD(ep_wfl, ep_t2);
                    issue_town(k + 1u);
                    if (k + 2u < nbatch) issue_pairs(k + 2u);
                    EP_ADD(ep_brec, ep_t3);
                }
                service();
            }
            EP_CNT(ep_nb, nbatch);
            EP_T0(ep_t5);
            while (min_done() < nbatch) {
                if (!service()) tw_spin();
            }
            service();  // the last run words (queued before the batch counts)
            EP_ADD(ep_proc, ep_t5);
        }
        asm volatile("fence" ::: "memory");
        invalidate_l1_cache();  // the TRISCs' cursors, for the drain
    } else
#endif
    if (nbatch > 0) {
        issue_pairs(0);
        noc_async_read_barrier();
        issue_brec(0, 0);
        if (nbatch > 1) issue_pairs(1);
        uint32_t h = 0;
        EP_ADD(ep_pro, ep_t_pro);
        for (uint32_t k = 0; k < nbatch; k++) {
            EP_T0(ep_t1);
            noc_async_read_barrier();  // blendrec of batch k, pairs of batch k+1
            EP_T0(ep_t2);
            EP_ADD(ep_rdw, ep_t1);
            if (k + 1u < nbatch) {
                issue_brec(k + 1u, h ^ 1u);
                EP_T0(ep_t3);
                EP_ADD(ep_brec, ep_t2);
                if (k + 2u < nbatch) issue_pairs(k + 2u);
                EP_ADD(ep_pairs, ep_t3);
            }
            EP_T0(ep_t4);
            if (!fast) {
                process_batch(k, h);
            } else {
                // The planes and blendrec pages are complete (read barrier
                // above); plain loads from here may be scheduled freely.
                asm volatile("" ::: "memory");
                const Planes pl = planes(k);
                const int32_t* kp = const_cast<const int32_t*>(pl.k);
                const int32_t* gp = const_cast<const int32_t*>(pl.g);
                const uint32_t* tp = reinterpret_cast<const uint32_t*>(const_cast<const int32_t*>(pl.t));
                const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
                uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
                if (p0 + n_el > P) n_el = (P > p0) ? P - p0 : 0u;
                const uint8_t* bp = reinterpret_cast<const uint8_t*>(rec_cache_l1 + h * BREC_HALF * PAGE_BYTES);
                // Bulk half: g's page is at (jr, ji) = ((g - ba) % nb, (g - ba) / nb),
                // stepped from the last g (jl = its g - ba): rp = bp + jr * bk_run + ji * 64.
                const uint32_t ba = bk_a[h], bn = bk_n[h];
                const uint8_t* rp = bp;
                uint32_t jl = 0, jr = 0;
                const uint32_t cap = tile_cap, msk = tx_mask, sh = tx_shift, ring = ring_l1;
                int32_t g_c = f_g;
                uint32_t cov0 = f_cov0, cov1 = f_cov1, cov2 = f_cov2, dep = f_dep, mxb = f_mx, myb = f_my;
                uint32_t opr = f_opr, cgb = f_cgb, ty_c = f_ty, myt = f_myt;
                for (uint32_t j = 0; j < n_el; j++) {
                    if (kp[j] == 0) continue;
                    const int32_t g = gp[j];
                    const uint32_t t = tp[j];
                    if (g != g_c) {
                        g_c = g;
                        const uint32_t* cp;
                        const uint32_t jg = static_cast<uint32_t>(g) - ba;
                        if (jg < bn) {
                            if (jg < jl) {
                                jl = 0;
                                jr = 0;
                                rp = bp;
                            }
                            for (; jl != jg; jl++) {
                                rp += bk_run;
                                if (++jr == bk_nb) {
                                    jr = 0;
                                    rp -= bk_wrap;
                                }
                            }
                            cp = reinterpret_cast<const uint32_t*>(rp);
                        } else if (bn == 0u) {
                            cp = reinterpret_cast<const uint32_t*>(bp);
                            bp += PAGE_BYTES;
                        } else {
                            cp = brec_now(g);
                        }
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
                    const uint32_t c = cur_lm[t];
                    cur_lm[t] = c + 1u;
                    EP_CNT(ep_nrec, 1u);
                    if (c >= cap) continue;  // past capacity: dropped, host fails the frame
                    const uint32_t ri = c & (R - 1u);
                    if (ri == 0u) {
                        EP_T0(ep_t);
                        noc_async_writes_flushed();
                        EP_ADD(ep_wfl, ep_t);
                        EP_CNT(ep_nrun, 1u);
                    }
                    const uint32_t kx = (t & msk) * L1_TILE_SIZE;
                    uint32_t mx;
                    if (!sort_bin_fp32::sub_int32(mxb, kx, &mx)) {
                        mx = sub_int_cold(mxb, kx);
                        EP_CNT(ep_ncold, 1u);
                    }
                    const uint32_t tyi = t >> sh;
                    if (tyi != ty_c) {
                        ty_c = tyi;
                        const uint32_t ky = tyi * L1_TILE_SIZE;
                        if (!sort_bin_fp32::sub_int32(myb, ky, &myt)) {
                            myt = sub_int_cold(myb, ky);
                            EP_CNT(ep_ncold, 1u);
                        }
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
                        EP_T0(ep_t);
                        flush_run(t, c);
                        EP_ADD(ep_wiss, ep_t);
                    }
                }
                f_g = g_c;
                f_cov0 = cov0; f_cov1 = cov1; f_cov2 = cov2; f_dep = dep; f_mx = mxb; f_my = myb;
                f_opr = opr; f_cgb = cgb; f_ty = ty_c; f_myt = myt;
            }
            EP_ADD(ep_proc, ep_t4);
            h ^= 1u;
        }
        EP_CNT(ep_nb, nbatch);
    }
    EP_T0(ep_t_drain);
    flush_recs();
    if (ring_on) {
        // Each tile's partial final run (full runs were written in the loop).
        for (uint32_t t = 0; t < num_tiles; t++) {
            uint32_t last;
            const uint32_t end = curp[t];
            if (sort_ol::ring_drain(startp[t], end, tile_cap, R, &last)) flush_run(t, last);
        }
    }
    EP_ADD(ep_drain, ep_t_drain);
    EP_T0(ep_t_wbar);
    noc_async_write_barrier();
    EP_ADD(ep_wbar, ep_t_wbar);
#if OL_EMIT_TOWN
    {
        // Every TRISC is done with this stream: clear GO for the next launch.
        EP_T0(ep_t_fin);
        for (;;) {
            invalidate_l1_cache();
            if (twh[tw::H_FIN] == tw::MAGIC && twh[tw::H_FIN + 1u] == tw::MAGIC &&
                twh[tw::H_FIN + 2u] == tw::MAGIC) {
                break;
            }
            tw_spin();
        }
        twh[tw::H_GO] = 0u;
        twh[tw::H_GO2] = 0u;
        for (uint32_t i = 0; i < tw::NT; i++) twh[tw::H_FIN + i] = 0u;
        asm volatile("fence" ::: "memory");
        EP_ADD(ep_pairs, ep_t_fin);
    }
#endif
#if OL_EMIT_PROF
    DeviceTimestampedData("ep_pro", ep_pro);
    DeviceTimestampedData("ep_rdw", ep_rdw);
    DeviceTimestampedData("ep_brec", ep_brec);
    DeviceTimestampedData("ep_pairs", ep_pairs);
    DeviceTimestampedData("ep_proc", ep_proc);
    DeviceTimestampedData("ep_wfl", ep_wfl);
    DeviceTimestampedData("ep_wiss", ep_wiss);
    DeviceTimestampedData("ep_drain", ep_drain);
    DeviceTimestampedData("ep_wbar", ep_wbar);
    DeviceTimestampedData("ep_nrec", ep_nrec);
    DeviceTimestampedData("ep_npf", ep_npf);
    DeviceTimestampedData("ep_nb", ep_nb);
    DeviceTimestampedData("ep_ncold", ep_ncold);
    DeviceTimestampedData("ep_nrun", ep_nrun);
    DeviceTimestampedData("ep_nbk", ep_nbk);
    DeviceTimestampedData("ep_bpg", ep_bpg);
#endif
}
