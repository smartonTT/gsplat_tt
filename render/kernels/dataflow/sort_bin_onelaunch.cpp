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
// RUNTIME ARGS
//   0 gids  1 tids  2 keep  3 depth  4 blendrec  5 bucket  6 count rows
//   7 base rows  8 totals rows  9 pg_lo  10 pg_hi (this mover's pages)  11 P
//   12 num_tiles  13 row_pages (a row = row_pages 64 B pages)  14 core_id
//   15 num_cores  16 tile_cap  17 tiles_x  18 mover (0 BRISC, 1 NCRISC)
//   19..24 semaphores: counted, based, arrive1, release1, arrive2, release2
//   25 coordinator NoC x  26 coordinator NoC y
//   27.. (core 0 mover 0 only) NoC x | y << 16 of logical core r, r < num_cores
// COMPILE-TIME ARGS: 9 TensorAccessorArgs (gids, tids, keep, depth, blendrec,
//   bucket, count rows, base rows, totals rows).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"
#include "sort_onelaunch_algo.h"

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
#ifndef OL_WIN_PAGES
#define OL_WIN_PAGES 1536u  // == sort_device.cpp onelaunch window
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
// Task #164 (host knob GSPLAT_TT_OL_EMIT_FOLD=0 turns it off): the fast emit
// scans batch k+1 for its blendrec reads inside batch k's pack loop.
#ifndef OL_EMIT_FOLD
#define OL_EMIT_FOLD 1
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
    const uint32_t pg_lo = get_arg_val<uint32_t>(9);
    const uint32_t pg_hi = get_arg_val<uint32_t>(10);
    const uint32_t P = get_arg_val<uint32_t>(11);
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

    const uint32_t npages = pg_hi - pg_lo;
    // Window size rounded down to whole count batches (a batch is all-window
    // or all-fallback).
    const uint32_t nwin_raw = (npages < WIN_PAGES) ? npages : WIN_PAGES;
    const uint32_t nwin = (nwin_raw == npages) ? npages : (nwin_raw / CNT_BATCH) * CNT_BATCH;
    const uint32_t win_gid = get_write_ptr(CB_WIN + cbo);
    const uint32_t win_tid = win_gid + WIN_PAGES * PAGE_BYTES;
    const uint32_t win_keep = win_tid + WIN_PAGES * PAGE_BYTES;

    // ── 1. count this mover's kept pairs per tile ──────────────────────────
    {
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
        // Hand the count row to mover 0 and wait for the bases.
        asm volatile("fence" ::: "memory");
        noc_semaphore_set(sem_counted, 1u);
        noc_semaphore_wait(sem_based, 1u);
        noc_semaphore_set(sem_based, 0u);  // re-arm for the next launch
        invalidate_l1_cache();
    } else {
        noc_semaphore_wait(sem_counted, 1u);
        noc_semaphore_set(sem_counted, 0u);
        invalidate_l1_cache();
        const uint32_t row_span = row_pages * ELEMS_PER_PAGE;
        const bool coordinator = (core_id == 0u);
        auto barrier = [&](uint32_t arrive_id, uint32_t release_id) {
            DeviceZoneScopedN("sort_ol_barrier");
            noc_async_write_barrier();  // this core's DRAM writes are visible first
            if (coordinator) {
                auto arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_id));
                noc_semaphore_wait(arrive, num_cores - 1u);
                noc_semaphore_set(arrive, 0u);
                for (uint32_t r = 1; r < num_cores; r++) {
                    const uint32_t xy = get_arg_val<uint32_t>(27 + r);
                    sem_inc(xy & 0xFFFFu, xy >> 16, release_id);
                }
                noc_async_atomic_barrier();
            } else {
                sem_inc(coord_x, coord_y, arrive_id);
                noc_async_atomic_barrier();
                auto release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_id));
                noc_semaphore_wait(release, 1u);
                noc_semaphore_set(release, 0u);
            }
            invalidate_l1_cache();
        };

        // ── 1b. the core's count row (both movers) to DRAM ─────────────────
        for (uint32_t t = 0; t < row_span; t++) rowp[t] = (t < num_tiles) ? h0p[t] + h1p[t] : 0u;
        asm volatile("fence" ::: "memory");
        for (uint32_t q = 0; q < row_pages; q++) {
            noc_async_write(row_l1 + q * PAGE_BYTES, get_noc_addr(core_id * row_pages + q, cnt_acc),
                            PAGE_BYTES);
        }
        barrier(sem_arrive1_id, sem_release1_id);

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
                for (uint32_t r = 0; r < num_cores; r++) {
                    noc_async_read(get_noc_addr(r * row_pages + p, cnt_acc), pfx + r * PAGE_BYTES,
                                   PAGE_BYTES);
                }
                noc_async_read_barrier();
                for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                    uint32_t acc = 0, pad = 0;
                    for (uint32_t r = 0; r < num_cores; r++) {
                        const uint32_t h = cin[r * ELEMS_PER_PAGE + j];
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
        barrier(sem_arrive2_id, sem_release2_id);

        // ── 4. this core's base row (first slot of its records per tile) ───
        for (uint32_t q = 0; q < row_pages; q++) {
            noc_async_read(get_noc_addr(core_id * row_pages + q, base_acc), row_l1 + q * PAGE_BYTES,
                           PAGE_BYTES);
        }
        noc_async_read_barrier();
        asm volatile("fence" ::: "memory");
        noc_semaphore_set(sem_based, 1u);
    }
    // Cursors: BRISC's records first, then NCRISC's, inside the core's chunk.
    for (uint32_t t = 0; t < num_tiles; t++) curp[t] = rowp[t] + ((mover == 1) ? h0p[t] : 0u);

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
             ep_drain = 0, ep_wbar = 0, ep_nrec = 0, ep_npf = 0, ep_nb = 0;
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
    const uint32_t rec_cache_l1 = get_write_ptr(CB_REC + cbo);  // 2 x PB*16 blendrec pages
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
    // one loop with register locals, the per-tile cursors in the RISC's local
    // memory (an L1 cursor is loaded right after its store, task #26) and the
    // sub_int fast path inlined. Same records, same slots, same writes.
    constexpr bool FAST_OK = OL_EMIT_FAST && PUBOC && R != 0u;
    const bool fast = FAST_OK && ring_on && tx_is_pow2;
    uint32_t cur_lm[FAST_OK ? OL_RING_TILES : 1u];
    if (ring_on) {
        for (uint32_t t = 0; t < num_tiles; t++) startp[t] = curp[t];
        if (fast) {
            for (uint32_t t = 0; t < num_tiles; t++) cur_lm[t] = curp[t];
        }
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
    // place; later ones use pair buffer k % 3, issued two batches ahead.
    struct Planes { volatile int32_t* g; volatile int32_t* t; volatile int32_t* k; };
    auto planes = [&](uint32_t k) -> Planes {
        const uint32_t w = k * PB;
        if (w < nwin) {
            return {reinterpret_cast<volatile int32_t*>(win_gid + w * PAGE_BYTES),
                    reinterpret_cast<volatile int32_t*>(win_tid + w * PAGE_BYTES),
                    reinterpret_cast<volatile int32_t*>(win_keep + w * PAGE_BYTES)};
        }
        const uint32_t o = (k % 3u) * PB * PAGE_BYTES;
        return {reinterpret_cast<volatile int32_t*>(gid_l1 + o),
                reinterpret_cast<volatile int32_t*>(tid_l1 + o),
                reinterpret_cast<volatile int32_t*>(keep_l1 + o)};
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
            noc_async_read(get_noc_addr(pg0 + b, keep_acc), keep_l1 + o + b * PAGE_BYTES, PAGE_BYTES);
        }
    };
    // One blendrec page per run of equal kept g (scan_g carries the last kept
    // g across batches, the same test process_batch uses) into ring half h.
    int32_t scan_g = -1;
    auto issue_brec = [&](uint32_t k, uint32_t h) {
        const Planes pl = planes(k);
        const uint32_t dst0 = rec_cache_l1 + h * BATCH_ELEMS * PAGE_BYTES;
        const uint32_t n_el = batch_pages(k) * ELEMS_PER_PAGE;
        const uint32_t p0 = (pg_lo + k * PB) * ELEMS_PER_PAGE;
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
        const uint32_t ring0 = rec_cache_l1 + h * BATCH_ELEMS * PAGE_BYTES;
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
            // Task #164: a window batch k+1 is scanned in batch k's fast loop.
            const bool fold = OL_EMIT_FOLD && fast && k + 1u < nbatch && (k + 1u) * PB < nwin;
            if (k + 1u < nbatch) {
                if (!fold) issue_brec(k + 1u, h ^ 1u);
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
                const uint8_t* bp = reinterpret_cast<const uint8_t*>(rec_cache_l1 + h * BATCH_ELEMS * PAGE_BYTES);
                const uint32_t cap = tile_cap, msk = tx_mask, sh = tx_shift, ring = ring_l1;
                int32_t g_c = f_g;
                uint32_t cov0 = f_cov0, cov1 = f_cov1, cov2 = f_cov2, dep = f_dep, mxb = f_mx, myb = f_my;
                uint32_t opr = f_opr, cgb = f_cgb, ty_c = f_ty, myt = f_myt;
                // Fold (task #164): issue_brec's scan of batch k+1 (its planes
                // sit in the window), two elements per pack iteration so the
                // last read is issued halfway through this batch. Same reads,
                // same order, same ring slots as issue_brec.
                uint32_t n_sc = 0;
                const int32_t* skp = kp;
                const int32_t* sgp = gp;
                if (fold) {
                    const Planes pn = planes(k + 1u);
                    skp = const_cast<const int32_t*>(pn.k);
                    sgp = const_cast<const int32_t*>(pn.g);
                    const uint32_t p1 = p0 + PB * ELEMS_PER_PAGE;
                    n_sc = batch_pages(k + 1u) * ELEMS_PER_PAGE;
                    if (p1 + n_sc > P) n_sc = (P > p1) ? P - p1 : 0u;
                }
                int32_t s_g = scan_g;
                uint32_t s_dst = rec_cache_l1 + (h ^ 1u) * BATCH_ELEMS * PAGE_BYTES;
                uint32_t sj = 0;
#if OL_EMIT_PROF
                const uint32_t s_dst0 = s_dst;
#endif
                for (uint32_t j = 0; j < n_el; j++) {
                    if (sj < n_sc) {
                        const uint32_t se = (sj + 2u < n_sc) ? sj + 2u : n_sc;
                        for (; sj < se; sj++) {
                            if (skp[sj] == 0) continue;
                            const int32_t sg = sgp[sj];
                            if (sg != s_g) {
                                s_g = sg;
                                noc_async_read(get_noc_addr(static_cast<uint32_t>(sg), brec_acc), s_dst, PAGE_BYTES);
                                s_dst += PAGE_BYTES;
                            }
                        }
                    }
                    if (kp[j] == 0) continue;
                    const int32_t g = gp[j];
                    const uint32_t t = tp[j];
                    if (g != g_c) {
                        g_c = g;
                        const uint32_t* cp = reinterpret_cast<const uint32_t*>(bp);
                        bp += PAGE_BYTES;
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
                        EP_T0(ep_t);
                        flush_run(t, c);
                        EP_ADD(ep_wiss, ep_t);
                    }
                }
                for (; sj < n_sc; sj++) {  // only if batch k is shorter than half of k+1
                    if (skp[sj] == 0) continue;
                    const int32_t sg = sgp[sj];
                    if (sg != s_g) {
                        s_g = sg;
                        noc_async_read(get_noc_addr(static_cast<uint32_t>(sg), brec_acc), s_dst, PAGE_BYTES);
                        s_dst += PAGE_BYTES;
                    }
                }
                scan_g = s_g;
                EP_CNT(ep_npf, (s_dst - s_dst0) / PAGE_BYTES);
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
            const uint32_t end = fast ? cur_lm[t] : curp[t];
            if (sort_ol::ring_drain(startp[t], end, tile_cap, R, &last)) flush_run(t, last);
        }
    }
    EP_ADD(ep_drain, ep_t_drain);
    EP_T0(ep_t_wbar);
    noc_async_write_barrier();
    EP_ADD(ep_wbar, ep_t_wbar);
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
#endif
}
