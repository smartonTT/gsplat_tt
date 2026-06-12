// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sort BIN kernel — R4/R5 resident-pairs handoff (GSPLAT_TT_RESIDENT_PAIRS).
//
// Replaces the host binning (Pass1 counts + Pass2 stable scatter) of
// gsplat_cpu::sort.cpp with a multi-core device pass that reads the RESIDENT
// tile_assign outputs (full-P gaussian-major (gid,tid) pairs + keep mask) and
// the resident proj_m_depth, and writes the PAGE-ALIGNED per-tile (key,id)
// layout the device radix kernel (sort_radix_tile.cpp) consumes — entirely on
// device. No D2H of the pairs, no H2D of keys/ids.
//
// The full-P pairs are split across cores in contiguous 16-pair page ranges
// (identical split for count + scatter). Two modes (host launches both):
//
//   mode 0 (count): each core builds a per-tile histogram of its KEPT pairs in
//     L1 and writes that row into bin2d[core_id*stride .. +num_tiles).
//
//   mode 1 (scatter): the host has, from the per-core histograms, computed the
//     page-aligned per-tile starts and each core's per-tile global base offset
//     (an exclusive prefix over cores within each tile), written back into
//     bin2d. The core does a LOCAL counting-sort of its kept pairs into L1
//     (grouped by tile, gaussian-major within each tile), then writes each
//     tile's contiguous block to DRAM at base_row[t] using batched, page-
//     congruent sub-page NoC writes (the gather_visible_scatter pattern). The
//     within-tile order across cores is gaussian-major (core c's block before
//     core c+1's) — exactly the CPU's stable pre-sort order.
//
// RUNTIME ARGS (all uint32):
//   0: gids_addr   1: tids_addr   2: keep_addr   3: depth_addr  (resident in)
//   4: bin2d_addr  (per-core 2D: hist out / base in, stride-padded rows)
//   5: keys_out_addr  6: ids_out_addr  (page-aligned layout, scatter out)
//   7: page_start  8: page_count  9: P  10: num_tiles  11: stride
//   12: core_id    13: mode (0=count, 1=scatter)
//
// COMPILE-TIME ARGS: 7 TensorAccessorArgs
//   gids, tids, keep, depth, bin2d, keys_out, ids_out.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

// Bit-exact IEEE 754 fp32→fp16 (round-to-nearest-even, no flush-to-zero).
// Used for packing the 32B L1 record (M0, GSPLAT_TT_L1_RECORD).
inline uint16_t fp32_to_fp16(float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    const uint32_t sign  = (u >> 31) & 1u;
    const uint32_t exp32 = (u >> 23) & 0xffu;
    const uint32_t mant  = u & 0x7fffffu;
    // NaN / Inf
    if (exp32 == 0xff) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    }
    int32_t e = static_cast<int32_t>(exp32) - 127 + 15;
    if (e >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7c00u);  // overflow → inf
    }
    if (e <= 0) {
        // Denormal or underflow: encode as denormal fp16 (or 0).
        if (e < -10) return static_cast<uint16_t>(sign << 15);
        const uint32_t m = (mant | 0x800000u) >> (1 - e + 13);
        const uint32_t round = (mant | 0x800000u) >> (-e + 12) & 1u;
        return static_cast<uint16_t>((sign << 15) | m + round);
    }
    // Normal.
    const uint32_t m16 = mant >> 13;
    const uint32_t round = (mant >> 12) & 1u;
    return static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(e) << 10) | m16 + round);
}

}  // namespace

void kernel_main() {
    const uint32_t gids_addr   = get_arg_val<uint32_t>(0);
    const uint32_t tids_addr   = get_arg_val<uint32_t>(1);
    const uint32_t keep_addr   = get_arg_val<uint32_t>(2);
    const uint32_t depth_addr  = get_arg_val<uint32_t>(3);
    const uint32_t bin2d_addr  = get_arg_val<uint32_t>(4);
    const uint32_t keys_addr   = get_arg_val<uint32_t>(5);
    const uint32_t ids_addr    = get_arg_val<uint32_t>(6);
    const uint32_t page_start  = get_arg_val<uint32_t>(7);
    const uint32_t page_count  = get_arg_val<uint32_t>(8);
    const uint32_t P           = get_arg_val<uint32_t>(9);
    const uint32_t num_tiles   = get_arg_val<uint32_t>(10);
    const uint32_t stride      = get_arg_val<uint32_t>(11);
    const uint32_t core_id     = get_arg_val<uint32_t>(12);
    const uint32_t mode        = get_arg_val<uint32_t>(13);
    // The Tracy device zone is opened inside each pass below (histogram vs emit)
    // with a COMPILE-TIME-LITERAL name. DeviceZoneScopedN hashes its argument via
    // Hash16_CT(const char (&)[N]); a runtime ternary decays to const char* and
    // fails that template's N deduction (kernel_profiler.hpp:110) under
    // TT_METAL_DEVICE_PROFILER=1. Two static-named zones keep the per-pass labels.
    // TILE_BUCKET: scatter the FULL projected record into a per-tile contiguous
    // bucket in arbitrary (bin/gaussian) order, so the depth sort can move into
    // the per-tile L1 pass and NO random DRAM read is needed to build the bucket.
    // proj_m_blendrec is 1 record (64B) per page (page==g); tile_recs mirrors the
    // keys/ids layout (record page e ↔ keys/ids element e).
    const uint32_t blendrec_addr = get_arg_val<uint32_t>(15);
    const uint32_t tile_recs_addr= get_arg_val<uint32_t>(16);
    const uint32_t recbase_addr  = get_arg_val<uint32_t>(17);  // dense per-(core,tile) base
    // L1_RECORD (PACK2): scatter 32B records into pre-sized per-tile buckets.
    // buf_l1_recs is BUCKET_FIT*num_tiles logical slots, two per 64B page;
    // buf_l1_rec_base
    // provides per-(core,tile) start slot = t*BUCKET_FIT + prefix.
    const uint32_t l1_recs_addr  = get_arg_val<uint32_t>(18);
    const uint32_t l1_base_addr  = get_arg_val<uint32_t>(19);
    const uint32_t l1_bucket_fit = get_arg_val<uint32_t>(20);  // per-tile bucket slot count
    const uint32_t l1_tiles_x    = get_arg_val<uint32_t>(21);  // tiles per row (tile-local mean)
    constexpr uint32_t L1_TILE_SIZE = 32u;  // microblock tile = 32x32 px
    // iter 135 (bit-identical strength reduction): the per-pair tile-local-mean
    // recompute in pack_rec needs tt/l1_tiles_x and tt%l1_tiles_x. l1_tiles_x is
    // a RUNTIME arg, so the compiler cannot strength-reduce it and emits a soft
    // __udivmodsi4 PER KEPT PAIR on the divider-less NCRISC (this is the dominant
    // sort_bucket_emit zone, every pair is compute-exposed once blendrec is
    // cached). When the tile grid is a power of two (e.g. 1024px/32 => 32 tiles
    // per row) the divide is EXACTLY a right shift and the modulo EXACTLY a mask
    // — BIT-IDENTICAL for unsigned (same quotient/remainder, no rounding). Detect
    // the power-of-two case ONCE here; non-power-of-two grids fall back to the
    // exact divmod so output is unchanged for any tiles_x.
    const bool tx_is_pow2 = (l1_tiles_x != 0u) && ((l1_tiles_x & (l1_tiles_x - 1u)) == 0u);
    uint32_t tx_shift = 0u;
    if (tx_is_pow2) { uint32_t v = l1_tiles_x; while (v > 1u) { v >>= 1; tx_shift++; } }
    const uint32_t tx_mask = l1_tiles_x - 1u;

    constexpr auto gids_args  = TensorAccessorArgs<0>();
    constexpr auto tids_args  = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto keep_args  = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto depth_args = TensorAccessorArgs<keep_args.next_compile_time_args_offset()>();
    constexpr auto bin2d_args = TensorAccessorArgs<depth_args.next_compile_time_args_offset()>();
    constexpr auto keys_args  = TensorAccessorArgs<bin2d_args.next_compile_time_args_offset()>();
    constexpr auto ids_args   = TensorAccessorArgs<keys_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto tile_recs_args= TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto recbase_args  = TensorAccessorArgs<tile_recs_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args  = TensorAccessorArgs<recbase_args.next_compile_time_args_offset()>();
    constexpr auto l1_base_args  = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();

    const auto gids_acc  = TensorAccessor(gids_args,  gids_addr,  PAGE_BYTES);
    const auto tids_acc  = TensorAccessor(tids_args,  tids_addr,  PAGE_BYTES);
    const auto keep_acc  = TensorAccessor(keep_args,  keep_addr,  PAGE_BYTES);
    const auto depth_acc = TensorAccessor(depth_args, depth_addr, PAGE_BYTES);
    const auto bin2d_acc = TensorAccessor(bin2d_args, bin2d_addr, PAGE_BYTES);
    const auto keys_acc  = TensorAccessor(keys_args,  keys_addr,  PAGE_BYTES);
    const auto ids_acc   = TensorAccessor(ids_args,   ids_addr,   PAGE_BYTES);
    const auto blendrec_acc = TensorAccessor(blendrec_args, blendrec_addr, PAGE_BYTES);
    // tile_recs (dense 64B record buffer) is parsed to keep the accessor offsets
    // aligned with the host's CT-arg order, but the blend reader serves every
    // tile from the 32B buf_l1_recs (L1_RECORD) and never reads tile_recs, so it
    // is intentionally unwritten here.
    const auto tile_recs_acc= TensorAccessor(tile_recs_args, tile_recs_addr, PAGE_BYTES);
    const auto recbase_acc  = TensorAccessor(recbase_args,  recbase_addr,  PAGE_BYTES);
    (void)tile_recs_acc;
    (void)recbase_acc;  // dense per-(core,tile) base — only used by the retired tile_recs scatter
    // PACK2: two 32B splats per 64B page (slot s => page s/2, half s&1 at +32*half).
    // Accessor page = 64B; sub-64B page size is unreliable on BH.
    const auto l1_recs_acc  = TensorAccessor(l1_recs_args,  l1_recs_addr,  64u);
    const auto l1_base_acc  = TensorAccessor(l1_base_args,  l1_base_addr,  PAGE_BYTES);

    // CB layout (declared in sort_device.cpp binning program):
    //   0 gid_in (64B)  1 tid_in (64B)  2 keep_in (64B)  3 depth (64B)
    //   4 row    (MAX_BIN_TILES*4: hist out / base in)
    //   5 cur    (MAX_BIN_TILES*4: local per-tile cursor)
    //   6 off    (MAX_BIN_TILES*4: local per-tile L1 offset)
    //   7 ksort  (BIN_LOCAL_MAX*4: L1 counting-sort keys)
    //   8 isort  (BIN_LOCAL_MAX*4: L1 counting-sort ids)
    constexpr uint32_t CB_GID = 0, CB_TID = 1, CB_KEEP = 2, CB_DEP = 3,
                       CB_ROW = 4, CB_CUR = 5, CB_OFF = 6, CB_KS = 7, CB_IS = 8;
    constexpr uint32_t CB_REC = 9;     // 64B blendrec staging (read page g, +depth, write bucket)
    constexpr uint32_t CB_L1BASE   = 11; // per-(core,tile) L1 slot base row (= t*BUCKET_FIT + prefix)
    constexpr uint32_t CB_L1SCRATCH = 12; // 32B staging buffer for pack → noc write
    constexpr uint32_t CB_PACKOC   = 13; // iter 132: 64B-per-gaussian blendrec page write-back ring (publishes packed op/color)

    const uint32_t gid_l1  = get_write_ptr(CB_GID);
    const uint32_t tid_l1  = get_write_ptr(CB_TID);
    const uint32_t keep_l1 = get_write_ptr(CB_KEEP);
    const uint32_t dep_l1  = get_write_ptr(CB_DEP);
    const uint32_t row_l1  = get_write_ptr(CB_ROW);

    auto gidp  = reinterpret_cast<volatile int32_t*>(gid_l1);
    auto tidp  = reinterpret_cast<volatile int32_t*>(tid_l1);
    auto keepp = reinterpret_cast<volatile int32_t*>(keep_l1);
    auto depp  = reinterpret_cast<volatile uint32_t*>(dep_l1);
    auto rowp  = reinterpret_cast<volatile uint32_t*>(row_l1);

    const uint32_t row_pages = (num_tiles + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
    const uint32_t base_page = core_id * (stride / ELEMS_PER_PAGE);
    const uint32_t pg_lo = page_start;
    const uint32_t pg_hi = page_start + page_count;

    if (mode == 0) {
        DeviceZoneScopedN("sort_bin_hist");
        // ── count: per-tile histogram of kept pairs ─────────────────────
        for (uint32_t t = 0; t < num_tiles; t++) rowp[t] = 0;
        for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
            noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
            noc_async_read_barrier();
            for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
                const uint32_t p = pg * ELEMS_PER_PAGE + j;
                if (p >= P) break;
                if (keepp[j] == 0) continue;
                rowp[static_cast<uint32_t>(tidp[j])]++;
            }
        }
        for (uint32_t pp = 0; pp < row_pages; pp++) {
            noc_async_write(row_l1 + pp * PAGE_BYTES,
                            get_noc_addr(base_page + pp, bin2d_acc), PAGE_BYTES);
        }
        noc_async_write_barrier();
        return;
    }

    DeviceZoneScopedN("sort_bucket_emit");
    // ── scatter ─────────────────────────────────────────────────────────
    // Load this core's global base row.
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, bin2d_acc),
                       row_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
    // Load this core's L1 slot base row (per-tile slot index = t*BUCKET_FIT + prefix).
    const uint32_t l1base_l1 = get_write_ptr(CB_L1BASE);
    auto l1basep = reinterpret_cast<volatile uint32_t*>(l1base_l1);
    for (uint32_t pp = 0; pp < row_pages; pp++) {
        noc_async_read(get_noc_addr(base_page + pp, l1_base_acc),
                       l1base_l1 + pp * PAGE_BYTES, PAGE_BYTES);
    }
    noc_async_read_barrier();
    // The local slot cursor reuses curp[t]: it gets reset to 0 in the prefix loop
    // below, then incremented in the scatter loop alongside the (key,id) write.
    const uint32_t l1_scratch = get_write_ptr(CB_L1SCRATCH);

    const uint32_t cur_l1 = get_write_ptr(CB_CUR);
    const uint32_t off_l1 = get_write_ptr(CB_OFF);
    auto curp = reinterpret_cast<volatile uint32_t*>(cur_l1);
    auto offp = reinterpret_cast<volatile uint32_t*>(off_l1);

    // Sub-pass 1: local per-tile kept count -> curp (reused as scratch count).
    for (uint32_t t = 0; t < num_tiles; t++) curp[t] = 0;
    for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
        noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
        noc_async_read_barrier();
        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            const uint32_t p = pg * ELEMS_PER_PAGE + j;
            if (p >= P) break;
            if (keepp[j] == 0) continue;
            curp[static_cast<uint32_t>(tidp[j])]++;
        }
    }
    const uint32_t ks_l1 = get_write_ptr(CB_KS);
    const uint32_t is_l1 = get_write_ptr(CB_IS);
    auto ksp = reinterpret_cast<volatile uint32_t*>(ks_l1);
    auto isp = reinterpret_cast<volatile uint32_t*>(is_l1);

    // PAGE-ALIGNED exclusive prefix over tiles -> local L1 block offsets; reset
    // curp to 0. Each tile's L1 block is rounded up to a whole page so the
    // write-out can emit only whole-page, page-aligned DRAM transfers (no
    // sub-page writes -> no multi-writer/same-chunk DRAM write race).
    uint32_t run = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        offp[t] = run;
        const uint32_t c = curp[t];
        run += ((c + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE) * ELEMS_PER_PAGE;
        curp[t] = 0;
    }
    // Pre-fill the L1 counting-sort region with the max key (0xffffffff) / 0 id
    // so unfilled tail slots of each tile's page-block become padding the stable
    // radix pushes to the end of the tile.
    for (uint32_t i = 0; i < run; i++) { ksp[i] = 0xffffffffu; isp[i] = 0u; }

    int32_t dep_cached_page = -1;

    // iter 114 (sort Stage 2): cache blendrec[g] ONCE per gaussian. Pairs are
    // gaussian-major, so consecutive kept pairs share the same g; the old fill
    // re-read the same 64B blendrec page once PER PAIR (the "random per-pair
    // blendrec gather" the Stage-2 plan calls out — a K-tile gaussian re-read
    // its record K times). We now read blendrec[g] into ONE L1 cache page when g
    // changes and pack every pair's 32B record straight from that cache. The
    // bytes written to buf_l1_recs are bit-identical; only the redundant DRAM
    // reads are removed (blendrec read once per gaussian, not once per pair).
    // The packed 32B records still stage into a per-batch region of l1_scratch
    // and flush to buf_l1_recs under ONE write barrier per REC_BATCH.
    constexpr uint32_t REC_BATCH = 16u;
    const uint32_t rec_cache_l1 = get_write_ptr(CB_REC);  // 64B cache for current g's blendrec
    volatile uint32_t* cachep = reinterpret_cast<volatile uint32_t*>(rec_cache_l1);
    int32_t blendrec_cached_g = -1;
    uint32_t brec_l1_slot[REC_BATCH];  // abs slot in buf_l1_recs (overflow → 0xFFFFFFFF)
    uint32_t nbrec = 0;
    // The 32B record is packed at enqueue time (blendrec already cached), so the
    // flush only issues the pre-packed writes under one barrier — no per-record
    // blendrec read barrier here.
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        for (uint32_t b = 0; b < nbrec; b++) {
            // Skip the 32B scatter for overflow records (heavy tile past its bucket).
            if (brec_l1_slot[b] == 0xFFFFFFFFu) continue;
            const uint32_t slot = brec_l1_slot[b];
            const uint32_t page = slot >> 1;
            const uint32_t half_off = (slot & 1u) * 32u;
            noc_async_write(l1_scratch + b * 32u,
                            get_noc_addr(page, l1_recs_acc) + half_off,
                            32u);
        }
        noc_async_write_barrier();
        nbrec = 0;
    };

    // iter 132: publish the per-gaussian packed op/color words (inv_opr, inv_cgb)
    // into blendrec[10],[11] so the depth-sorted materialize overflow gather COPIES
    // them (it already reads the 64B blendrec page) instead of re-deriving 4
    // fp32->UNORM16 conversions per overflow record.
    //
    // WRITE GRANULARITY (measured): sub-page 8B/4B splats to byte-offset 40 of a
    // blendrec page did NOT land on this BH (materialize read 0 -> 25.5 dB) — 8B at
    // a non-16B-aligned offset is below the DRAM write granule. A full-64B page
    // write-back DOES land (mirrors gather), but costs +15.4 ms on NCRISC-KERNEL
    // (Tracy) which couples ~1:1 into BRISC-FW via handoff stalls and exactly
    // cancels the proj_scatter revert. So publish the MINIMAL 16B-aligned chunk that
    // covers words 10,11: a 16B write at byte-offset 32 = words [8,9,10,11]. 16B is
    // the natural DRAM granule (16-aligned offset + 16B size) so it lands. Words 8,9
    // are re-written with their EXACT original gather values (cb, depth/0), so a
    // gaussian processed by two cores at a pair-page boundary writes byte-identical
    // 16B — no clobber, fully idempotent. ~1/4 the bytes + copy of the 64B version.
    // Staged in a ring (one write barrier per batch), mirroring flush_recs.
    constexpr uint32_t PACKOC_BATCH = 16u;
    constexpr uint32_t PACKOC_ENT_W = 4u;   // 16B chunk = 4 u32 (words 8,9,10,11)
    const uint32_t packoc_l1 = get_write_ptr(CB_PACKOC);
    auto packocp = reinterpret_cast<volatile uint32_t*>(packoc_l1);
    uint32_t packoc_g[PACKOC_BATCH];
    uint32_t n_packoc = 0;
    auto flush_packoc = [&]() {
        if (n_packoc == 0) return;
        for (uint32_t b = 0; b < n_packoc; b++) {
            noc_async_write(packoc_l1 + b * (PACKOC_ENT_W * 4u),
                            get_noc_addr(packoc_g[b], blendrec_acc) + 32u,
                            PACKOC_ENT_W * 4u);
        }
        noc_async_write_barrier();
        n_packoc = 0;
    };

    // Pack the 32B PACK2 record. Covariance FULL fp32 — precision-critical (the
    // blend recomputes the conic via det = a*c - b*b, which loses too much to
    // fp16 when a,c are large, ~10000s px^2 => only ~47 dB). Mean is TILE-LOCAL
    // fp32 (sub-px center); opacity/color are [0,1] => UNORM16 (~30x tighter than
    // fp16, the op/color precision wall).
    // 64B blendrec (fp32 words): 0=cov_a 1=cov_b 2=cov_c 3=mx 4=my 5=op
    //                            6=cr    7=cg    8=cb    9=depth_key(u32)
    // 32B layout (M0): [0]fp32 cov_a [1]fp32 cov_b [2]fp32 cov_c [3]u32 depth_key
    //   [4]fp32 mx_local [5]fp32 my_local [6]unorm16 op,r [7]unorm16 g,b
    //
    // iter-128: hoist the per-gaussian-INVARIANT packing out of the per-pair
    // loop. cov(0,1,2), depth_key(3), and the op/color UNORM16 words(6,7) are
    // functions of the GAUSSIAN only — identical for every one of the K tiles a
    // gaussian touches — yet the old pack recomputed the four UNORM conversions
    // (float multiplies) and re-read the cached blendrec for EVERY pair. Measured
    // (iter-128 ablation, 30-view makespan): pack_rec was ~27 ms/view = 71 % of
    // sort_bucket_emit (the 32B scatter writes were only ~1.6 ms — the scattered-
    // DRAM-write premise was refuted). Only the tile-local mean (words 4,5) varies
    // per pair, so we compute the invariant words ONCE per gaussian (when blendrec
    // is read) into registers, and per pair only subtract the tile origin from the
    // cached fp32 mean. The bytes written to buf_l1_recs are BIT-IDENTICAL.
    uint32_t inv_cov0 = 0, inv_cov1 = 0, inv_cov2 = 0, inv_depth = 0,
             inv_opr = 0, inv_cgb = 0;
    float inv_mx = 0.0f, inv_my = 0.0f;
    auto pack_invariants = [&](uint32_t depth_key) {
        inv_cov0 = cachep[0];  // fp32 cov_a (exact: no fp16 det cancellation)
        inv_cov1 = cachep[1];  // fp32 cov_b
        inv_cov2 = cachep[2];  // fp32 cov_c
        inv_depth = depth_key;
        inv_mx = *reinterpret_cast<const volatile float*>(&cachep[3]);
        inv_my = *reinterpret_cast<const volatile float*>(&cachep[4]);
        // iter 132: compute the op/color UNORM16 pack ONCE per gaussian, here on
        // the NCRISC side (gather now writes raw fp32 op/cr/cg/cb at words 5..8,
        // keeping the pack off the BRISC proj_scatter long pole). The two packed
        // words feed this core's in-budget bucket record AND are published into
        // blendrec[10],[11] (publish_packoc) so the depth-sorted materialize
        // overflow gather COPIES them instead of re-packing. Byte-identical to
        // the iter-131 birth pack (same fp32 inputs, same rounding formula).
        float op = *reinterpret_cast<const volatile float*>(&cachep[5]);
        float cr = *reinterpret_cast<const volatile float*>(&cachep[6]);
        float cg = *reinterpret_cast<const volatile float*>(&cachep[7]);
        float cb_v = *reinterpret_cast<const volatile float*>(&cachep[8]);
        auto to_unorm = [](float v) -> uint32_t {
            if (v <= 0.0f) return 0u;
            if (v >= 1.0f) return 65535u;
            return static_cast<uint32_t>(v * 65535.0f + 0.5f);
        };
        inv_opr = (to_unorm(op) | (to_unorm(cr) << 16));
        inv_cgb = (to_unorm(cg) | (to_unorm(cb_v) << 16));
    };
    auto pack_rec = [&](uint32_t b, uint32_t tt) {
        // Tile-local mean: the blend reader reconstructs absolute via
        // mean = local + tile_origin. Only this is per-pair (per-tile) work.
        // tx = tt % l1_tiles_x, ty = tt / l1_tiles_x — soft-divmod replaced by a
        // shift/mask on the power-of-two grid (bit-identical; see hoist above).
        const uint32_t txi = tx_is_pow2 ? (tt & tx_mask) : (tt % l1_tiles_x);
        const uint32_t tyi = tx_is_pow2 ? (tt >> tx_shift) : (tt / l1_tiles_x);
        float mx = inv_mx - static_cast<float>(txi * L1_TILE_SIZE);
        float my = inv_my - static_cast<float>(tyi * L1_TILE_SIZE);
        volatile uint32_t* p32 =
            reinterpret_cast<volatile uint32_t*>(l1_scratch + b * 32u);
        uint32_t mx_bits, my_bits;
        __builtin_memcpy(&mx_bits, &mx, 4);
        __builtin_memcpy(&my_bits, &my, 4);
        p32[0] = inv_cov0;
        p32[1] = inv_cov1;
        p32[2] = inv_cov2;
        p32[3] = inv_depth;
        p32[4] = mx_bits;  // fp32 tile-local mean x (sub-px center precision)
        p32[5] = my_bits;  // fp32 tile-local mean y
        p32[6] = inv_opr;
        p32[7] = inv_cgb;
    };

    // Sub-pass 2: counting-sort kept pairs into L1, grouped by tile (gaussian-
    // major within each tile). key = depth_bits[g], id = g.
    for (uint32_t pg = pg_lo; pg < pg_hi; pg++) {
        noc_async_read(get_noc_addr(pg, gids_acc), gid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, keep_acc), keep_l1, PAGE_BYTES);
        noc_async_read_barrier();
        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            const uint32_t p = pg * ELEMS_PER_PAGE + j;
            if (p >= P) break;
            if (keepp[j] == 0) continue;
            const uint32_t g = static_cast<uint32_t>(gidp[j]);
            const uint32_t t = static_cast<uint32_t>(tidp[j]);
            const int32_t dpg = static_cast<int32_t>(g / ELEMS_PER_PAGE);
            if (dpg != dep_cached_page) {
                noc_async_read(get_noc_addr(static_cast<uint32_t>(dpg), depth_acc),
                               dep_l1, PAGE_BYTES);
                noc_async_read_barrier();
                dep_cached_page = dpg;
            }
            const uint32_t key = depp[g % ELEMS_PER_PAGE];
            const uint32_t li = offp[t] + curp[t];
            // Scatter the full record to its per-tile bucket slot. DENSE layout:
            // tile t's records occupy slots [t*FIT, ...); this core's k-th kept
            // pair for tile t goes to l1basep[t] + curp[t] (l1basep[t] =
            // t*FIT + prefix of cores < this core). Each (core,tile) region is
            // disjoint (the "[tile][core][slot]" address partitioning), so cores
            // never share a slot — no race, no atomics.
            //
            // Stage 2: blendrec[g] is read ONCE per gaussian into rec_cache_l1
            // (consecutive pairs share g), then every pair's 32B record is packed
            // straight from that cache — no random per-pair re-read.
            if (static_cast<int32_t>(g) != blendrec_cached_g) {
                noc_async_read(get_noc_addr(g, blendrec_acc), rec_cache_l1, PAGE_BYTES);
                noc_async_read_barrier();
                blendrec_cached_g = static_cast<int32_t>(g);
                // key (= depp[g % 16]) is the GAUSSIAN's depth — invariant across
                // its pairs — so the full invariant prefix is computed once here.
                pack_invariants(key);
                // iter 132: stage the 16B blendrec chunk [words 8,9,10,11] — words
                // 8,9 keep their original gather bytes (cb, depth/0), words 10,11 get
                // the packed op/color — written back 16B-aligned (offset 32) in
                // flush_packoc for materialize to copy.
                {
                    volatile uint32_t* ent =
                        packocp + n_packoc * PACKOC_ENT_W;
                    ent[0] = cachep[8];   // original cb (fp32) — preserved
                    ent[1] = cachep[9];   // original depth/0 — preserved
                    ent[2] = inv_opr;     // -> blendrec[10]
                    ent[3] = inv_cgb;     // -> blendrec[11]
                    packoc_g[n_packoc] = g;
                    n_packoc++;
                    if (n_packoc == PACKOC_BATCH) flush_packoc();
                }
            }
            {
                // Absolute slot in buf_l1_recs = per-core base + local cursor.
                // Clamp to this tile's pre-sized bucket [t*FIT, (t+1)*FIT): tiles
                // whose count exceeds FIT (e.g. max_tile_n > BUCKET_FIT) would
                // otherwise scatter past their bucket — corrupting neighbor tiles'
                // buckets and (for the last tiles) writing past buf_l1_recs into
                // adjacent DRAM. The blend reader serves such heavy tiles from the
                // dense gather fallback (Lb > MB_BUCKET_FIT), never from this bucket,
                // so dropping the overflow records here is correct. Sentinel
                // 0xFFFFFFFF marks "skip the 32B scatter" for this batched entry.
                const uint32_t l1_slot = l1basep[t] + curp[t];
                const uint32_t out_slot =
                    (l1_slot < (t + 1u) * l1_bucket_fit) ? l1_slot : 0xFFFFFFFFu;
                brec_l1_slot[nbrec] = out_slot;
                if (out_slot != 0xFFFFFFFFu) {
                    pack_rec(nbrec, t);  // pack into l1_scratch + nbrec*32
                }
                nbrec++;
                if (nbrec == REC_BATCH) flush_recs();
            }
            curp[t] = curp[t] + 1;
            ksp[li] = key;
            isp[li] = g;
        }
    }
    flush_recs();    // drain the partial final batch
    flush_packoc();  // iter 132: drain the partial final packed-op/color batch

    // Write each tile's page-aligned L1 block to DRAM as WHOLE PAGES. The block
    // base (rowp[t]) and L1 source (offp[t]) are both page-aligned, and the size
    // is a page multiple, so every transfer is a clean whole-page write to pages
    // this core owns EXCLUSIVELY. No sub-page writes and no shared pages -> no
    // multi-writer DRAM write race. Tail slots already hold max-key/0 padding
    // (pre-filled above), which the stable radix sorts to the end of the tile;
    // host compaction keeps only the real count.
    for (uint32_t t = 0; t < num_tiles; t++) {
        const uint32_t n = curp[t];
        if (n == 0) continue;
        const uint32_t base_pg = rowp[t] / ELEMS_PER_PAGE;   // page-aligned base
        const uint32_t src = offp[t];                        // page-aligned L1 src
        const uint32_t pages = (n + ELEMS_PER_PAGE - 1) / ELEMS_PER_PAGE;
        for (uint32_t pp = 0; pp < pages; pp++) {
            const uint32_t soff = (src + pp * ELEMS_PER_PAGE) * 4;
            noc_async_write(ks_l1 + soff, get_noc_addr(base_pg + pp, keys_acc), PAGE_BYTES);
            noc_async_write(is_l1 + soff, get_noc_addr(base_pg + pp, ids_acc),  PAGE_BYTES);
        }
        noc_async_write_barrier();
    }
}
