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
    // L1_RECORD: scatter 32B fp16-packed records into pre-sized per-tile buckets.
    // buf_l1_recs is BUCKET_FIT*num_tiles slots × 32B each; buf_l1_rec_base
    // provides per-(core,tile) start slot = t*BUCKET_FIT + prefix.
    const uint32_t l1_recs_addr  = get_arg_val<uint32_t>(18);
    const uint32_t l1_base_addr  = get_arg_val<uint32_t>(19);
    const uint32_t l1_bucket_fit = get_arg_val<uint32_t>(20);  // per-tile bucket slot count
    const uint32_t l1_tiles_x    = get_arg_val<uint32_t>(21);  // tiles per row (tile-local mean)
    constexpr uint32_t L1_TILE_SIZE = 32u;  // microblock tile = 32x32 px

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
    // M0: 32B fp16 record packed into the LOW 32B of a 64B DRAM page (sub-64B
    // paging is unreliable here — see sort_device l1_rec_bytes). Accessor page =
    // 64B (one record per page); only 32B are written/read.
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

    // T1b: ring-buffered record scatter. Instead of a synchronous read+write
    // barrier per record (which serialized the bin stage at ~+20 ms), stage up
    // to REC_BATCH records in CB_REC, issue all reads under ONE barrier, inject
    // depth, then issue all writes under ONE barrier. The bucket scatter is off
    // the blend critical path, so this batching just hides the NoC latency.
    constexpr uint32_t REC_BATCH = 16u;
    const uint32_t rec_l1_base = get_write_ptr(CB_REC);
    uint32_t brec_key[REC_BATCH];
    uint32_t brec_l1_slot[REC_BATCH];  // abs slot index in buf_l1_recs for each batched entry (overflow → 0xFFFFFFFF)
    uint32_t brec_tile[REC_BATCH];     // tile id for each batched entry (tile-local mean)
    uint32_t nbrec = 0;
    // L1 32B record staging (reuse a 32B aligned region immediately after rec_l1_base ring;
    // we write it with a single noc_async_write per record during flush).
    auto flush_recs = [&]() {
        if (nbrec == 0) return;
        noc_async_read_barrier();
        for (uint32_t b = 0; b < nbrec; b++) {
            const uint32_t slot = rec_l1_base + b * PAGE_BYTES;
            volatile uint32_t* sp = reinterpret_cast<volatile uint32_t*>(slot);
          // Skip the 32B scatter for overflow records (heavy tile past its bucket).
          if (brec_l1_slot[b] != 0xFFFFFFFFu) {
            // Pack the 32B record from the 64B blendrec in L1 and write to
            // buf_l1_recs (low 32B of a 64B page). Covariance stays FULL fp32 —
            // it is precision-critical (the blend recomputes the conic via
            // det = a*c - b*b, which loses too much to fp16 when a,c are large,
            // ~10000s px^2 => only ~47 dB). Mean is TILE-LOCAL fp16 (small => sub-
            // 0.1px ULP); opacity/color fp16.
            // 64B blendrec (fp32 words): 0=cov_a 1=cov_b 2=cov_c 3=mx 4=my 5=op
            //                            6=cr    7=cg    8=cb    9=depth_key(u32)
            // 32B layout (M0): cov (a,b,c) + tile-local mean fp32 (precision-
            // critical); opacity/color are [0,1] => UNORM16 (~30x tighter than fp16
            // in the same bytes; this is the op/color precision wall).
            //   [0]=fp32 cov_a [1]=fp32 cov_b [2]=fp32 cov_c [3]=u32 depth_key
            //   [4]=fp32 mx_local [5]=fp32 my_local
            //   [6]=unorm16 opacity,r   [7]=unorm16 g,b
            float mx = *reinterpret_cast<const volatile float*>(&sp[3]);
            float my = *reinterpret_cast<const volatile float*>(&sp[4]);
            // Tile-local mean: the blend reader reconstructs absolute via
            // mean = local + tile_origin.
            {
                const uint32_t tt = brec_tile[b];
                mx -= static_cast<float>((tt % l1_tiles_x) * L1_TILE_SIZE);
                my -= static_cast<float>((tt / l1_tiles_x) * L1_TILE_SIZE);
            }
            float op = *reinterpret_cast<const volatile float*>(&sp[5]);
            float cr = *reinterpret_cast<const volatile float*>(&sp[6]);
            float cg = *reinterpret_cast<const volatile float*>(&sp[7]);
            float cb_v = *reinterpret_cast<const volatile float*>(&sp[8]);
            const uint32_t depth_key = brec_key[b];
            // Pack into per-batch slot of l1_scratch (16 × 32B = 512B), then issue
            // async write. Each batch entry has its own 32B region so they can all
            // be in-flight simultaneously without aliasing.
            volatile uint32_t* p32 =
                reinterpret_cast<volatile uint32_t*>(l1_scratch + b * 32u);
            uint32_t mx_bits, my_bits;
            __builtin_memcpy(&mx_bits, &mx, 4);
            __builtin_memcpy(&my_bits, &my, 4);
            auto to_unorm = [](float v) -> uint32_t {
                if (v <= 0.0f) return 0u;
                if (v >= 1.0f) return 65535u;
                return static_cast<uint32_t>(v * 65535.0f + 0.5f);
            };
            p32[0] = sp[0];  // fp32 cov_a (exact: no fp16 det cancellation)
            p32[1] = sp[1];  // fp32 cov_b
            p32[2] = sp[2];  // fp32 cov_c
            p32[3] = depth_key;
            p32[4] = mx_bits;  // fp32 tile-local mean x (sub-px center precision)
            p32[5] = my_bits;  // fp32 tile-local mean y
            p32[6] = (to_unorm(op) | (to_unorm(cr) << 16));
            p32[7] = (to_unorm(cg) | (to_unorm(cb_v) << 16));
            noc_async_write(l1_scratch + b * 32u,
                            get_noc_addr(brec_l1_slot[b], l1_recs_acc),
                            32u);
          }
        }
        noc_async_write_barrier();
        nbrec = 0;
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
            // Scatter the full record to its per-tile bucket page. DENSE layout:
            // tile t's records occupy pages [starts[t], starts[t]+counts[t]); this
            // core's k-th kept pair for tile t goes to recrowp[t] + curp[t]
            // (recrowp[t] = starts[t] + prefix of cores < this core). Each record
            // is its own 64B page, so even dense, cores never share a page (no
            // race, no atomics). Read blendrec[g]
            // (page g), inject depth=key as the 10th field, write 64B. T1 keeps
            // it correctness-simple: one staging buffer, read+write barriered per
            // record (the scatter is off the blend critical path; T1b can ring-
            // buffer to pipeline if the bin stage cost matters).
            {
                // Stage into the ring; flush in batches (issue-ahead, 1 barrier
                // per REC_BATCH instead of 2 per record). rec_page captured with
                // the CURRENT curp[t] (matches the pre-increment dense layout).
                const uint32_t slot = rec_l1_base + nbrec * PAGE_BYTES;
                noc_async_read(get_noc_addr(g, blendrec_acc), slot, PAGE_BYTES);
                brec_key[nbrec] = key;
                // Absolute slot in buf_l1_recs = per-core base + local cursor.
                // Clamp to this tile's pre-sized bucket [t*FIT, (t+1)*FIT): tiles
                // whose count exceeds FIT (e.g. max_tile_n > BUCKET_FIT) would
                // otherwise scatter past their bucket — corrupting neighbor tiles'
                // buckets and (for the last tiles) writing past buf_l1_recs into
                // adjacent DRAM. The blend reader serves such heavy tiles from the
                // dense gather fallback (Lb > MB_BUCKET_FIT), never from this bucket,
                // so dropping the overflow records here is correct. Sentinel
                // 0xFFFFFFFF marks "skip the 32B scatter" for this batched entry.
                {
                    const uint32_t l1_slot = l1basep[t] + curp[t];
                    brec_l1_slot[nbrec] =
                        (l1_slot < (t + 1u) * l1_bucket_fit) ? l1_slot : 0xFFFFFFFFu;
                }
                brec_tile[nbrec] = t;  // tile-local mean reconstruction in flush
                nbrec++;
                if (nbrec == REC_BATCH) flush_recs();
            }
            curp[t] = curp[t] + 1;
            ksp[li] = key;
            isp[li] = g;
        }
    }
    flush_recs();  // drain the partial final batch

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
