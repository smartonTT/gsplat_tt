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

#ifndef EMIT_PUBOC
#define EMIT_PUBOC 0u
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
                   CB_BATCH = 6, CB_REC = 7, CB_SCRATCH = 8, CB_WIN = 12;
// The first WIN_PAGES pages of a mover's range stay in L1 (gid, tid, keep
// planes) from the count pass to the emit; the emit prefetches page i+1's
// blendrec and depth pages into the other half of a two-half ring while it
// packs page i. Pages past the window take the unpipelined path.
constexpr uint32_t WIN_PAGES = 1536;  // == sort_device.cpp kOneLaunchWinPages
constexpr uint32_t RING = 16;         // entries per ring half (<= 16 gaussians per page)
// Shared CBs (one copy per core): CB_ROW holds the core's count row, then its
// base row; CB_PFX stages the prefix pass (num_cores count pages, num_cores
// base pages, the totals page and the padded-totals page).
constexpr uint32_t CB_ROW = 10, CB_PFX = 11;

inline void sem_inc(uint32_t x, uint32_t y, uint32_t sem_id) {
    noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1u);
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
    DeviceZoneScopedN("sort_ol_emit");
    const bool tx_is_pow2 = (tiles_x != 0u) && ((tiles_x & (tiles_x - 1u)) == 0u);
    uint32_t tx_shift = 0u;
    if (tx_is_pow2) {
        uint32_t v = tiles_x;
        while (v > 1u) { v >>= 1; tx_shift++; }
    }
    const uint32_t tx_mask = tiles_x - 1u;

    const uint32_t gid_l1 = get_write_ptr(CB_GID + cbo);
    const uint32_t tid_l1 = get_write_ptr(CB_TID + cbo);
    const uint32_t keep_l1 = get_write_ptr(CB_KEEP + cbo);
    const uint32_t dep_l1 = get_write_ptr(CB_DEP + cbo);
    auto gidp = reinterpret_cast<volatile int32_t*>(gid_l1);
    auto tidp = reinterpret_cast<volatile int32_t*>(tid_l1);
    auto keepp = reinterpret_cast<volatile int32_t*>(keep_l1);
    const uint32_t rec_cache_l1 = get_write_ptr(CB_REC + cbo);
    const uint32_t l1_scratch = get_write_ptr(CB_SCRATCH + cbo);
    volatile uint32_t* cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1);

    int32_t blendrec_cached_g = -1;
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

    // Per-gaussian invariant words of the 32B record (== sort_bin.cpp).
    uint32_t inv_cov0 = 0, inv_cov1 = 0, inv_cov2 = 0, inv_depth = 0, inv_opr = 0, inv_cgb = 0;
    float inv_mx = 0.0f, inv_my = 0.0f;
    uint32_t inv_mx_bits = 0, inv_my_bits = 0;
    auto to_unorm = [](uint32_t bits) -> uint32_t {
        uint32_t u;
        if (sort_bin_fp32::unorm16(bits, &u)) return u;
        float v;
        __builtin_memcpy(&v, &bits, 4);
        if (v <= 0.0f) return 0u;
        if (v >= 1.0f) return 65535u;
        return static_cast<uint32_t>(v * 65535.0f + 0.5f);
    };
    auto pack_invariants = [&](uint32_t depth_key) {
        inv_cov0 = cachep[0];
        inv_cov1 = cachep[1];
        inv_cov2 = cachep[2];
        inv_depth = depth_key;
        inv_mx_bits = cachep[3];
        inv_my_bits = cachep[4];
        __builtin_memcpy(&inv_mx, &inv_mx_bits, 4);
        __builtin_memcpy(&inv_my, &inv_my_bits, 4);
        if constexpr (EMIT_PUBOC != 0u) {
            inv_opr = cachep[10];
            inv_cgb = cachep[11];
        } else {
            inv_opr = (to_unorm(cachep[5]) | (to_unorm(cachep[6]) << 16));
            inv_cgb = (to_unorm(cachep[7]) | (to_unorm(cachep[8]) << 16));
        }
    };
    auto pack_rec = [&](uint32_t b, uint32_t tt) {
        const uint32_t txi = tx_is_pow2 ? (tt & tx_mask) : (tt % tiles_x);
        const uint32_t tyi = tx_is_pow2 ? (tt >> tx_shift) : (tt / tiles_x);
        uint32_t mx_bits, my_bits;
        if (!sort_bin_fp32::sub_int(inv_mx_bits, txi * L1_TILE_SIZE, &mx_bits)) {
            const float mx = inv_mx - static_cast<float>(txi * L1_TILE_SIZE);
            __builtin_memcpy(&mx_bits, &mx, 4);
        }
        if (!sort_bin_fp32::sub_int(inv_my_bits, tyi * L1_TILE_SIZE, &my_bits)) {
            const float my = inv_my - static_cast<float>(tyi * L1_TILE_SIZE);
            __builtin_memcpy(&my_bits, &my, 4);
        }
        volatile uint32_t* p32 = reinterpret_cast<volatile uint32_t*>(l1_scratch + b * REC_BYTES);
        p32[0] = inv_cov0;
        p32[1] = inv_cov1;
        p32[2] = inv_cov2;
        p32[3] = inv_depth;
        p32[4] = mx_bits;
        p32[5] = my_bits;
        p32[6] = inv_opr;
        p32[7] = inv_cgb;
    };

    // Two-half rings: page i's distinct gaussians' blendrec pages (and the depth
    // pages they need) sit in half i & 1. Each page starts its own entries (a
    // gaussian continuing from the previous page is read again), so a half is
    // never referenced after the next prefetch into it is issued.
    const uint32_t dep_ring = dep_l1;  // RING x 2 depth pages
    uint32_t ent_dslot[2][RING];
    auto prefetch = [&](volatile int32_t* gp, volatile int32_t* kp, uint32_t pg, uint32_t h) {
        int32_t prev_g = -1;
        int32_t prev_dpg = -1;
        uint32_t nb = 0, nd = 0;
        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            if (pg * ELEMS_PER_PAGE + j >= P) break;
            if (kp[j] == 0) continue;
            const int32_t gj = gp[j];
            if (gj == prev_g) continue;
            prev_g = gj;
            noc_async_read(get_noc_addr(static_cast<uint32_t>(gj), brec_acc),
                           rec_cache_l1 + (h * RING + nb) * PAGE_BYTES, PAGE_BYTES);
            const int32_t dpg = gj / static_cast<int32_t>(ELEMS_PER_PAGE);
            if (EMIT_PUBOC == 0u && dpg != prev_dpg) {
                prev_dpg = dpg;
                noc_async_read(get_noc_addr(static_cast<uint32_t>(dpg), depth_acc),
                               dep_ring + (h * RING + nd) * PAGE_BYTES, PAGE_BYTES);
                nd++;
            }
            ent_dslot[h][nb] = nd - 1u;
            nb++;
        }
    };
    auto consume = [&](volatile int32_t* gp, volatile int32_t* tp, volatile int32_t* kp, uint32_t pg,
                       uint32_t h) {
        int32_t page_prev_g = -1;
        uint32_t k = 0;
        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            if (pg * ELEMS_PER_PAGE + j >= P) break;
            if (kp[j] == 0) continue;
            const uint32_t g = static_cast<uint32_t>(gp[j]);
            const uint32_t t = static_cast<uint32_t>(tp[j]);
            if (static_cast<int32_t>(g) != page_prev_g) {
                page_prev_g = static_cast<int32_t>(g);
                cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1 + (h * RING + k) * PAGE_BYTES);
                if (static_cast<int32_t>(g) != blendrec_cached_g) {
                    blendrec_cached_g = static_cast<int32_t>(g);
                    if constexpr (EMIT_PUBOC != 0u) {
                        pack_invariants(cachep[12]);
                    } else {
                        auto dp = reinterpret_cast<volatile uint32_t*>(
                            dep_ring + (h * RING + ent_dslot[h][k]) * PAGE_BYTES);
                        pack_invariants(dp[g % ELEMS_PER_PAGE]);
                    }
                }
                k++;
            }
            const uint32_t c = curp[t];
            curp[t] = c + 1u;
            const uint32_t slot = (c < tile_cap) ? t * tile_cap + c : 0xFFFFFFFFu;
            brec_slot[nbrec] = slot;
            if (slot != 0xFFFFFFFFu) pack_rec(nbrec, t);
            nbrec++;
            if (nbrec == REC_BATCH) flush_recs();
        }
    };
    auto wptr = [&](uint32_t plane, uint32_t i) {
        return reinterpret_cast<volatile int32_t*>(plane + i * PAGE_BYTES);
    };
    if (nwin > 0) prefetch(wptr(win_gid, 0), wptr(win_keep, 0), pg_lo, 0);
    for (uint32_t i = 0; i < npages; i++) {
        const uint32_t pg = pg_lo + i;
        const uint32_t h = i & 1u;
        if (i < nwin) {
            noc_async_read_barrier();  // page i's prefetch (issued one page ago)
            if (i + 1u < nwin) prefetch(wptr(win_gid, i + 1u), wptr(win_keep, i + 1u), pg + 1u, h ^ 1u);
            consume(wptr(win_gid, i), wptr(win_tid, i), wptr(win_keep, i), pg, h);
        } else {
            noc_async_read(get_noc_addr(pg, gids_acc), gid_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
            noc_async_read_barrier();
            prefetch(gidp, keepp, pg, h);
            noc_async_read_barrier();
            consume(gidp, tidp, keepp, pg, h);
        }
    }
    flush_recs();
    noc_async_write_barrier();
}
