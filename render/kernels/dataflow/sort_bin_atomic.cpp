// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN, atomic fixed-capacity bucket append (task #24, R8 / candidate 5.3).
//
// Replaces the count kernel + host prefix/layout + scatter kernel of sort_bin.cpp
// with ONE launch. Every tile owns a fixed-capacity bucket of `tile_cap` 32B
// records (PACK2, two per 64B page) at slot t * tile_cap in the bucket buffer.
// A core does not need to know where the other cores' records go:
//
//   1. count   each mover counts the kept pairs of its page range per tile
//              (BRISC [lo, mid), NCRISC [mid, hi) — the T-C split).
//   2. reserve NCRISC sums the two rows and, per non-empty tile, does ONE NoC
//              atomic fetch-and-add of the core's count on the tile's counter
//              (an L1 buffer, zeroed by the host before the launch). The old
//              value is the core's first slot in the tile. It publishes
//              (count << 16 | first slot) per tile as its row of the chunk
//              table, so the materialize can put the chunks back in core order.
//   3. emit    both movers scatter their records: BRISC from the first slot,
//              NCRISC from first slot + BRISC's count — the same order the
//              prefix-sum layout gave inside the core.
//
// Cores arrive at a counter in any order, so a tile's bucket holds the cores'
// chunks in arrival order. The sort key alone does not fix the order of equal
// depth keys; sort_subchunk_materialize rebuilds the canonical (core, page)
// order from the chunk table before its stable radix, which makes the output
// byte-identical to the prefix-sum layout. The per-tile totals stay in the
// counters for the host (one 4 KB read, no histogram, no layout upload). A tile
// with more than tile_cap records drops the excess here and the host fails the
// frame (the old path's MAX_TILE_ENTRIES limit is the same number).
//
// Unlike sort_bin.cpp this writes no (key, id) layout: nothing downstream of
// the radix read sort_sorted_ids except the over-cap gather, and every tile's
// full record set is now in its bucket.
//
// RUNTIME ARGS
//   0 gids  1 tids  2 keep  3 depth  4 blendrec  5 bucket  6 counters  7 table
//   8 pg_lo  9 pg_hi (this mover's pages)  10 P  11 num_tiles
//   12 row_pages (chunk-table row = row_pages 64B pages)  13 core_id
//   14 tile_cap  15 tiles_x  16 mover (0 BRISC, 1 NCRISC)
//   17 counted semaphore id  18 reserved semaphore id
// COMPILE-TIME ARGS: 8 TensorAccessorArgs (gids, tids, keep, depth, blendrec,
//   bucket, counters [L1], table).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sort_bin_fp32.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr uint32_t L1_TILE_SIZE = 32u;
constexpr uint32_t CNT_BATCH = 32u;  // count-pass pages per read barrier
constexpr uint32_t REC_BATCH = 16u;
constexpr uint32_t PACKOC_BATCH = 16u;
constexpr uint32_t PACKOC_ENT_W = 4u;
// BRISC (mover 0) uses its own copy of every private CB at id + 16.
constexpr uint32_t MOVER0_CB_OFFSET = 16;
// Private CBs (per mover).
constexpr uint32_t CB_GID = 0, CB_TID = 1, CB_KEEP = 2, CB_DEP = 3, CB_H = 4, CB_CUR = 5,
                   CB_BATCH = 6, CB_REC = 7, CB_SCRATCH = 8, CB_PACKOC = 9, CB_WIN = 12;
// The first WIN_PAGES pages of a mover's range stay in L1 (gid, tid, keep
// planes) from the count pass to the emit; the emit prefetches page i+1's
// blendrec and depth pages into the other half of a two-half ring while it
// packs page i. Pages past the window take the unpipelined path.
constexpr uint32_t WIN_PAGES = 1536;  // == sort_device.cpp kAtomicWinPages
constexpr uint32_t RING = 16;         // entries per ring half (<= 16 gaussians per page)
// Shared CBs (both movers, one copy per core).
constexpr uint32_t CB_BASE = 10, CB_TBL = 11;
// One atomic return value per 16 B word of CB_BASE: with 4 B spacing, returns
// for neighbouring tiles share a word and about one tile per frame came back
// with a stale base (overlapping chunks; the host GSPLAT_TT_SORT_ATOMIC_CHECK
// saw it, and the frame after hung).
constexpr uint32_t RET_STRIDE = 4;

}  // namespace

void kernel_main() {
    const uint32_t gids_addr = get_arg_val<uint32_t>(0);
    const uint32_t tids_addr = get_arg_val<uint32_t>(1);
    const uint32_t keep_addr = get_arg_val<uint32_t>(2);
    const uint32_t depth_addr = get_arg_val<uint32_t>(3);
    const uint32_t blendrec_addr = get_arg_val<uint32_t>(4);
    const uint32_t bucket_addr = get_arg_val<uint32_t>(5);
    const uint32_t counters_addr = get_arg_val<uint32_t>(6);
    const uint32_t table_addr = get_arg_val<uint32_t>(7);
    const uint32_t pg_lo = get_arg_val<uint32_t>(8);
    const uint32_t pg_hi = get_arg_val<uint32_t>(9);
    const uint32_t P = get_arg_val<uint32_t>(10);
    const uint32_t num_tiles = get_arg_val<uint32_t>(11);
    const uint32_t row_pages = get_arg_val<uint32_t>(12);
    const uint32_t core_id = get_arg_val<uint32_t>(13);
    const uint32_t tile_cap = get_arg_val<uint32_t>(14);
    const uint32_t tiles_x = get_arg_val<uint32_t>(15);
    const uint32_t mover = get_arg_val<uint32_t>(16);
    const uint32_t sem_counted_id = get_arg_val<uint32_t>(17);
    const uint32_t sem_reserved_id = get_arg_val<uint32_t>(18);

    constexpr auto gids_args = TensorAccessorArgs<0>();
    constexpr auto tids_args = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto keep_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto depth_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto brec_args = TensorAccessorArgs<depth_args.next_compile_time_args_offset()>();
    constexpr auto bucket_args = TensorAccessorArgs<brec_args.next_compile_time_args_offset()>();
    constexpr auto cnt_args = TensorAccessorArgs<bucket_args.next_compile_time_args_offset()>();
    constexpr auto tbl_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();
    const auto gids_acc = TensorAccessor(gids_args, gids_addr, PAGE_BYTES);
    const auto tids_acc = TensorAccessor(tids_args, tids_addr, PAGE_BYTES);
    const auto keep_acc = TensorAccessor(keep_args, keep_addr, PAGE_BYTES);
    const auto depth_acc = TensorAccessor(depth_args, depth_addr, PAGE_BYTES);
    const auto brec_acc = TensorAccessor(brec_args, blendrec_addr, PAGE_BYTES);
    const auto bucket_acc = TensorAccessor(bucket_args, bucket_addr, PAGE_BYTES);
    const auto cnt_acc = TensorAccessor(cnt_args, counters_addr, PAGE_BYTES);
    const auto tbl_acc = TensorAccessor(tbl_args, table_addr, PAGE_BYTES);

    const uint32_t cbo = (mover == 0) ? MOVER0_CB_OFFSET : 0u;
    auto hp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_H + cbo));
    auto curp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CUR + cbo));
    auto basep = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_BASE));
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
        DeviceZoneScopedN("sort_atomic_count");
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

    // ── 2. reserve: one atomic fetch-and-add per (core, non-empty tile) ────
    auto sem_counted = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_counted_id));
    auto sem_reserved = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserved_id));
    if (mover == 0) {
        asm volatile("fence" ::: "memory");
        noc_semaphore_set(sem_counted, 1u);
        noc_semaphore_wait(sem_reserved, 1u);
        noc_semaphore_set(sem_reserved, 0u);  // re-arm for the next launch
        invalidate_l1_cache();
    } else {
        noc_semaphore_wait(sem_counted, 1u);
        noc_semaphore_set(sem_counted, 0u);
        invalidate_l1_cache();
        DeviceZoneScopedN("sort_atomic_reserve");
        // The atomic returns the old counter value to a per-tile slot of the
        // shared base row. That needs the AT command buffer's return address
        // register, which firmware points at a scratch word and later kernels
        // rely on: save the buffer state and restore it after the barrier.
        NocCmdBufState saved;
        noc_cmd_buf_save_state(noc_index, write_at_cmd_buf, &saved);
        for (uint32_t t = 0; t < num_tiles; t++) {
            const uint32_t h = h0p[t] + h1p[t];
            if (h == 0u) continue;
            const uint64_t a = get_noc_addr(t / ELEMS_PER_PAGE, cnt_acc) + (t % ELEMS_PER_PAGE) * 4u;
            noc_fast_atomic_increment<noc_mode, true>(
                noc_index, write_at_cmd_buf, a, NOC_UNICAST_WRITE_VC, h, 31 /*wrap*/, false /*linked*/,
                false /*posted*/, reinterpret_cast<uint32_t>(basep + t * RET_STRIDE));
        }
        noc_async_atomic_barrier();
        noc_cmd_buf_restore_state(noc_index, write_at_cmd_buf, &saved);
        // Chunk table row: (count << 16) | first slot, 0 for an empty tile. Both
        // fit 16 bits whenever the host accepts the frame (count <= tile_cap).
        auto tblp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_TBL));
        const uint32_t row_span = row_pages * ELEMS_PER_PAGE;
        for (uint32_t t = 0; t < row_span; t++) {
            const uint32_t h = (t < num_tiles) ? h0p[t] + h1p[t] : 0u;
            tblp[t] = (h == 0u) ? 0u : ((h << 16) | (basep[t * RET_STRIDE] & 0xFFFFu));
        }
        const uint32_t tbl_l1 = get_write_ptr(CB_TBL);
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_write(tbl_l1 + pp * PAGE_BYTES, get_noc_addr(core_id * row_pages + pp, tbl_acc),
                            PAGE_BYTES);
        }
        asm volatile("fence" ::: "memory");
        noc_semaphore_set(sem_reserved, 1u);
    }
    // Cursors: BRISC's records first, then NCRISC's, inside the core's chunk.
    for (uint32_t t = 0; t < num_tiles; t++) curp[t] = basep[t * RET_STRIDE] + ((mover == 1) ? h0p[t] : 0u);

    // ── 3. emit: pack each kept pair's 32B record into its bucket slot ─────
    DeviceZoneScopedN("sort_atomic_emit");
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
    const uint32_t packoc_l1 = get_write_ptr(CB_PACKOC + cbo);
    auto packocp = reinterpret_cast<volatile uint32_t*>(packoc_l1);
    volatile uint32_t* cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1);

    int32_t blendrec_cached_g = -1;
    uint32_t brec_slot[REC_BATCH];
    uint32_t nbrec = 0;
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        for (uint32_t b = 0; b < nbrec; b++) {
            const uint32_t slot = brec_slot[b];
            if (slot == 0xFFFFFFFFu) continue;  // over capacity: dropped, host fails the frame
            noc_async_write(l1_scratch + b * 32u,
                            get_noc_addr(slot >> 1, bucket_acc) + (slot & 1u) * 32u, 32u);
        }
        noc_async_writes_flushed();
        nbrec = 0;
    };
    // iter 132: publish the packed op/color words into blendrec[10],[11] (16B
    // chunk of words 8..11, words 8,9 unchanged) — kept for any reader of them.
    uint32_t packoc_g[PACKOC_BATCH];
    uint32_t n_packoc = 0;
    auto flush_packoc = [&]() {
        if (n_packoc == 0) return;
        for (uint32_t b = 0; b < n_packoc; b++) {
            noc_async_write(packoc_l1 + b * (PACKOC_ENT_W * 4u),
                            get_noc_addr(packoc_g[b], brec_acc) + 32u, PACKOC_ENT_W * 4u);
        }
        noc_async_writes_flushed();
        n_packoc = 0;
    };

    // Per-gaussian invariant words of the 32B record (see sort_bin.cpp).
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
        inv_opr = (to_unorm(cachep[5]) | (to_unorm(cachep[6]) << 16));
        inv_cgb = (to_unorm(cachep[7]) | (to_unorm(cachep[8]) << 16));
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
        volatile uint32_t* p32 = reinterpret_cast<volatile uint32_t*>(l1_scratch + b * 32u);
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
            if (dpg != prev_dpg) {
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
                    auto dp = reinterpret_cast<volatile uint32_t*>(
                        dep_ring + (h * RING + ent_dslot[h][k]) * PAGE_BYTES);
                    pack_invariants(dp[g % ELEMS_PER_PAGE]);
                    volatile uint32_t* ent = packocp + n_packoc * PACKOC_ENT_W;
                    ent[0] = cachep[8];
                    ent[1] = cachep[9];
                    ent[2] = inv_opr;
                    ent[3] = inv_cgb;
                    packoc_g[n_packoc] = g;
                    n_packoc++;
                    if (n_packoc == PACKOC_BATCH) flush_packoc();
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
    flush_packoc();
    noc_async_write_barrier();
}
