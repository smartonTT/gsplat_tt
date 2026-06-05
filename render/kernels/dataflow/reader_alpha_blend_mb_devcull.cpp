// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-cull microblock-major alpha-blend READER (GSPLAT_TT_MB_DEVCULL).
//
// Unlike reader_alpha_blend_mb.cpp (which streams pre-culled, pre-conic 16-word
// rows the host built), this reader is fed ONLY compact per-gaussian attributes
// + per-tile depth-sorted candidate id lists. For each candidate it:
//   1. gathers attr[g] (raw cov a,b,c, image-space center, opacity, color),
//   2. computes the conic ci_a/ci_b/ci_c (det = max(a*c-b*b,1e-6)),
//   3. reproduces build_gaussian_major_tile's microblock cull on-core (opacity
//      floor -> bbox -> per-microblock constrained-min Mahalanobis vs the
//      contrib_floor threshold) to build the 32-bit microblock-coverage mask,
//   4. emits the SAME 16-word row the standard mb compute kernel consumes:
//        [cov_a, cov_b, cov_c, mx_local, my_local, 0, opacity, cr, cg, cb,
//         mask, 0,0,0,0,0]
// The compute kernel (MB_DEVCONIC) recomputes A,B,C from the raw cov on the
// SFPU and dispatches the blend to the masked microblocks exactly as today.
//
// This moves the host conic + microblock cull onto the device and removes the
// ~200MB/frame coeff-row upload (we ship M*64B attrs + P*4B ids instead).
//
// RUNTIME ARGS
//   0: attrs_addr        DRAM base of per-gaussian attr pages (64B each)
//   1: ids_addr          DRAM base of concatenated per-tile id lists (uint32)
//   2: ids_off_addr      DRAM base of per-tile prefix-sum offsets (uint32)
//   3: xramp_addr        DRAM base of the shared permuted xramp tile
//   4: yramp_addr        DRAM base of the shared permuted yramp tile
//   5: tile_ids_addr     DRAM base of this frame's LPT tile-id list (uint32)
//   6: tile_ids_start    this core's element offset into that list
//   7: tile_ids_count    number of tile IDs this core handles
//   8: tiles_x           tiles per image row (for tile origin)
//   9: contrib_floor     fp32 (bit-reinterpreted) microblock contrib floor
//  10: cull_disabled     1 => skip the constrained-min (m2_min := 0 in bbox)
//
// COMPILE-TIME ARGS: 6 TensorAccessorArgs: attrs, ids, ids_off, xramp, yramp,
// tile_ids. All DRAM-interleaved.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t ATTR_PAGE_BYTES = 64;   // 16 fp32, 9 used
constexpr uint32_t IDS_PAGE_BYTES = 64;    // 16 uint32 ids per page
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t TILE_SIZE = 32;

constexpr uint32_t CB_XRAMP     = 0;
constexpr uint32_t CB_YRAMP     = 1;
constexpr uint32_t CB_MB_COUNTS = 3;
constexpr uint32_t CB_SCR_IDS   = 4;   // reader-private: ids page scratch
constexpr uint32_t CB_SCR_ATTR  = 5;   // reader-private: attr page scratch
constexpr uint32_t CB_SCR_MASK  = 6;   // reader-private: 2x64B cull_masks page scratch (MB_SFPU_CULL)
constexpr uint32_t CB_CORE_TILES = 7;  // MB_RESIDENT: hand tile_ids_count to compute (no host arg)
constexpr uint32_t CB_BUCKET_BULK = 12; // subchunk: bulk L1 slab records (mask in word3)

// MB_COUNTS flags (slot 1): bit0=emit_tile, bit1=continue_blend, bit2=l1_bulk.
constexpr uint32_t MB_FLAG_EMIT = 1u;
constexpr uint32_t MB_FLAG_CONTINUE = 2u;
constexpr uint32_t MB_FLAG_L1_BULK = 4u;

// PACK2 (iter 50): two 32B splats per 64B DRAM page; splat g => page g/2, +32*(g&1).
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;

// M1b: FIXED-SIZE bulk CB slots (mirror alpha_blend_compute_mb.cpp). The bulk
// CBs are circular + accessed linearly over a whole tile span; a wrapping
// reservation corrupts the tail. Reserve/push a fixed slot per tile so every
// tile is slot-aligned (CBs sized as 2 slots) — no ring straddle.
constexpr uint32_t BULK_REC_SLOT = (MB_BUCKET_FIT + 1u) >> 1;          // 4096

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

// L1 store/visibility ordering fence. On Blackhole L1 is a small write-THROUGH
// cache, so a producer's stores reach L1, but a `fence` is needed so a freshly
// written CB row is coherent before cb_push_back signals the consumer (and so
// the compiler does not reorder the row stores past the push). == the
// invalidate_l1_cache() the runtime uses for cross-proc L1 handoff.
inline void mb_cb_commit_fence() {
    asm volatile("fence" ::: "memory");
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_blend_load");
    // RESIDENT blend (GSPLAT_TT_RESIDENT_BLEND): attrs gathered straight from
    // the device-resident per-component SoA proj_m_* buffers by id, candidate
    // ids consumed from resident sort_sorted_ids, per-tile [start,end) from
    // resident sort_tile_ranges. No host-built/uploaded attr table or id list.
    const uint32_t a_addr         = get_arg_val<uint32_t>(0);   // proj_m_a
    const uint32_t b_addr         = get_arg_val<uint32_t>(1);   // proj_m_b
    const uint32_t c_addr         = get_arg_val<uint32_t>(2);   // proj_m_c
    const uint32_t px_addr        = get_arg_val<uint32_t>(3);   // proj_m_px
    const uint32_t py_addr        = get_arg_val<uint32_t>(4);   // proj_m_py
    const uint32_t op_addr        = get_arg_val<uint32_t>(5);   // proj_m_opacity
    const uint32_t col_addr       = get_arg_val<uint32_t>(6);   // proj_m_colors (AoS M*3)
    const uint32_t ids_addr       = get_arg_val<uint32_t>(7);   // sort_sorted_ids
    const uint32_t ranges_addr    = get_arg_val<uint32_t>(8);   // sort_tile_ranges
    const uint32_t xramp_addr     = get_arg_val<uint32_t>(9);
    const uint32_t yramp_addr     = get_arg_val<uint32_t>(10);
    const uint32_t tile_ids_addr  = get_arg_val<uint32_t>(11);
    const uint32_t lpt_meta_addr  = get_arg_val<uint32_t>(12);
    const uint32_t core_index     = get_arg_val<uint32_t>(13);
    const uint32_t tiles_x        = get_arg_val<uint32_t>(14);
    const float contrib_floor     = bits_to_f(get_arg_val<uint32_t>(15));
    const bool cull_disabled      = get_arg_val<uint32_t>(16) != 0;
    // M3: cull_masks / cull_mask_base are gone — the mask travels in slab word3.
    const uint32_t blendrec_addr   = get_arg_val<uint32_t>(17);  // resident proj_m_blendrec (AoS)
    const uint32_t tile_recs_addr  = get_arg_val<uint32_t>(18);  // resident sort_tile_recs (dense)
    const uint32_t bucket_meta_addr= get_arg_val<uint32_t>(19);  // resident sort_bucket_meta (start,count)
    const uint32_t l1_recs_addr    = get_arg_val<uint32_t>(20);  // M0: pre-sized 32B record bucket
    const uint32_t subchunk_meta_addr = get_arg_val<uint32_t>(21);  // blend_subchunk_meta [dir_base,num_sc]
    const uint32_t subchunk_payload_addr = get_arg_val<uint32_t>(22);
    const uint32_t subchunk_dir_addr = get_arg_val<uint32_t>(23);

    constexpr auto a_args        = TensorAccessorArgs<0>();
    constexpr auto b_args        = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args        = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto px_args       = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto py_args       = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto op_args       = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto col_args      = TensorAccessorArgs<op_args.next_compile_time_args_offset()>();
    constexpr auto ids_args      = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
    constexpr auto ranges_args   = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto xramp_args    = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto yramp_args    = TensorAccessorArgs<xramp_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args = TensorAccessorArgs<yramp_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args  = TensorAccessorArgs<tile_ids_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args   = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto tile_recs_args  = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto bucket_meta_args= TensorAccessorArgs<tile_recs_args.next_compile_time_args_offset()>();
    constexpr auto l1_recs_args    = TensorAccessorArgs<bucket_meta_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_meta_args = TensorAccessorArgs<l1_recs_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_payload_args =
        TensorAccessorArgs<subchunk_meta_args.next_compile_time_args_offset()>();
    constexpr auto subchunk_dir_args =
        TensorAccessorArgs<subchunk_payload_args.next_compile_time_args_offset()>();

    // proj_m_* / sort_* are 64B (16-elem) DRAM-interleaved SoA pages.
    constexpr uint32_t SOA_PAGE_BYTES = 64;
    const auto a_acc        = TensorAccessor(a_args,        a_addr,        SOA_PAGE_BYTES);
    const auto b_acc        = TensorAccessor(b_args,        b_addr,        SOA_PAGE_BYTES);
    const auto c_acc        = TensorAccessor(c_args,        c_addr,        SOA_PAGE_BYTES);
    const auto px_acc       = TensorAccessor(px_args,       px_addr,       SOA_PAGE_BYTES);
    const auto py_acc       = TensorAccessor(py_args,       py_addr,       SOA_PAGE_BYTES);
    const auto op_acc       = TensorAccessor(op_args,       op_addr,       SOA_PAGE_BYTES);
    const auto col_acc      = TensorAccessor(col_args,      col_addr,      SOA_PAGE_BYTES);
    const auto ids_acc      = TensorAccessor(ids_args,      ids_addr,      IDS_PAGE_BYTES);
    const auto ranges_acc   = TensorAccessor(ranges_args,   ranges_addr,   SOA_PAGE_BYTES);
    const auto xramp_acc    = TensorAccessor(xramp_args,    xramp_addr,    RAMP_TILE_BYTES);
    const auto yramp_acc    = TensorAccessor(yramp_args,    yramp_addr,    RAMP_TILE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, 64);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, 64);
    // Under AoS the per-component SoA gather is replaced by proj_m_blendrec; the
    // SoA accessors stay bound (ABI parity) but are unused on this path.
    (void)a_acc; (void)b_acc; (void)c_acc; (void)px_acc; (void)py_acc;
    (void)op_acc; (void)col_acc;
    // PACK2: 64B DRAM pages hold two 32B splats (see sort_bin scatter).
    const auto l1_recs_acc = TensorAccessor(l1_recs_args, l1_recs_addr, L1_PACK_PAGE_BYTES);
    // Host-free LPT: read this core's (start,count) from resident sort_lpt_meta.
    constexpr uint32_t META_ELEMS_PER_PAGE = 16u;
    const uint32_t meta_elem0 = core_index * 2u;
    const uint32_t meta_page0 = meta_elem0 / META_ELEMS_PER_PAGE;
    const uint32_t meta_ip0   = meta_elem0 % META_ELEMS_PER_PAGE;
    const uint32_t meta_scratch = get_write_ptr(CB_SCR_IDS);
    auto meta_ptr = reinterpret_cast<volatile uint32_t*>(meta_scratch);
    noc_async_read(get_noc_addr(meta_page0, lpt_meta_acc), meta_scratch, 64);
    noc_async_read_barrier();
    uint32_t tile_ids_start = meta_ptr[meta_ip0];
    uint32_t tile_ids_count = 0;
    if (meta_ip0 + 1u < META_ELEMS_PER_PAGE) {
        tile_ids_count = meta_ptr[meta_ip0 + 1u];
    } else {
        noc_async_read(get_noc_addr(meta_page0 + 1u, lpt_meta_acc), meta_scratch, 64);
        noc_async_read_barrier();
        tile_ids_count = meta_ptr[0];
    }
    // Host-free: compute reads tile count from this CB instead of a runtime arg.
    cb_reserve_back(CB_CORE_TILES, 1);
    reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CORE_TILES))[0] = tile_ids_count;
    cb_push_back(CB_CORE_TILES, 1);
    // proj_m_blendrec: one 64B AoS record page per gaussian (page index == g).
    const auto blendrec_acc   = TensorAccessor(blendrec_args,   blendrec_addr,   SOA_PAGE_BYTES);
    // sort_tile_recs: DENSE per-tile bucket, one 64B record per kept candidate
    // {a,b,c,px,py,op,r,g,b,depth} (page index == bucket slot). sort_bucket_meta:
    // per-tile (start,count) u32 pair.
    const auto tile_recs_acc   = TensorAccessor(tile_recs_args,   tile_recs_addr,   SOA_PAGE_BYTES);
    const auto bucket_meta_acc = TensorAccessor(bucket_meta_args, bucket_meta_addr, 64);
    const auto subchunk_meta_acc = TensorAccessor(subchunk_meta_args, subchunk_meta_addr, 64);
    const auto subchunk_payload_acc =
        TensorAccessor(subchunk_payload_args, subchunk_payload_addr, L1_PACK_PAGE_BYTES);
    const auto subchunk_dir_acc = TensorAccessor(subchunk_dir_args, subchunk_dir_addr, 64);
    // Cull math moved to SFPU; mask is precomputed. contrib_floor still gates
    // reader-side row suppress (thr<0 sentinel == op<=floor).
    (void)cull_disabled;

    if (tile_ids_count == 0) {
        return;
    }

    // Cache this core's tile-ID slice in L1 (private ids scratch CB).
    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t scratch_addr = get_write_ptr(CB_SCR_IDS);
        auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
        const uint32_t ids_per_page = 64 / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            uint64_t page_noc = get_noc_addr(page_idx, tile_ids_acc);
            noc_async_read(page_noc, scratch_addr, 64);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) {
                tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            }
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    // Constant permuted coordinate ramps: identical for every tile. Stream
    // once per core (not once per tile) so compute can reuse the same CB pages
    // across its whole tile loop without redundant 8KB/tile NoC reads.
    cb_reserve_back(CB_XRAMP, 1);
    noc_async_read_tile(0, xramp_acc, get_write_ptr(CB_XRAMP));
    cb_reserve_back(CB_YRAMP, 1);
    noc_async_read_tile(0, yramp_acc, get_write_ptr(CB_YRAMP));
    noc_async_read_barrier();
    cb_push_back(CB_XRAMP, 1);
    cb_push_back(CB_YRAMP, 1);


    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;
        const float tx_tile = static_cast<float>(tx * TILE_SIZE);
        const float ty_tile = static_cast<float>(ty * TILE_SIZE);

        // (1) Per-tile candidate id range [id_start, id_end).
        uint32_t id_start, id_end;
        // Resident sort_tile_ranges: (start,end) uint32 pair per tile at
        // elements [tile_id*2, tile_id*2+1]. Equivalent to the host ids_off
        // prefix because sort_sorted_ids IS the per-tile depth-sorted concat
        // (start/end into it == ids_off[t]/ids_off[t+1]); empty tiles read
        // (0,0) -> L==0, matching the uploaded path.
        {
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            const uint32_t elem0 = tile_id * 2u;
            const uint32_t page = elem0 >> 4;
            const uint32_t off = elem0 & 0xF;
            auto rng_ptr = reinterpret_cast<volatile uint32_t*>(scr);
            noc_async_read_tile(page, ranges_acc, scr);
            noc_async_read_barrier();
            id_start = rng_ptr[off];
            if (off + 1u < 16u) {
                id_end = rng_ptr[off + 1u];
            } else {
                noc_async_read_tile(page + 1u, ranges_acc, scr);
                noc_async_read_barrier();
                id_end = rng_ptr[0];
            }
        }
        const uint32_t L = id_end - id_start;
        // Post-sort subchunk dispatch (iter 48): fat tiles (count > MB_BUCKET_FIT)
        // are processed as a depth-ordered sequence of subchunks; blend state
        // carries across subchunks via MB_COUNTS flags (bit0=emit, bit1=continue).
        uint32_t dir_base = 0;
        uint32_t num_subchunks = 1;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            noc_async_read_tile(pg, subchunk_meta_acc, scr);
            noc_async_read_barrier();
            auto smp = reinterpret_cast<volatile uint32_t*>(scr);
            dir_base = smp[off];
            num_subchunks = smp[off + 1u];
            if (num_subchunks == 0u) {
                num_subchunks = 1u;
            }
        }

        for (uint32_t sc = 0; sc < num_subchunks; ++sc) {
            const uint32_t sc_off = sc * MB_BUCKET_FIT;
            uint32_t L_sub = (sc_off >= L) ? 0u
                : ((L - sc_off > MB_BUCKET_FIT) ? MB_BUCKET_FIT : (L - sc_off));
            uint32_t flags = ((sc > 0u) ? MB_FLAG_CONTINUE : 0u)
                | ((sc + 1u == num_subchunks) ? MB_FLAG_EMIT : 0u);
            // M1b: read payload_page for ALL subchunks (single-subchunk too) so
            // every tile can consume the materialized slab via process_tile_l1_blend.
            uint32_t payload_page = 0;
            {
                const uint32_t de = (dir_base + sc) * 4u;
                const uint32_t dpg = de >> 4;
                const uint32_t dof = de & 0xF;
                const uint32_t scr = get_write_ptr(CB_SCR_IDS);
                noc_async_read_tile(dpg, subchunk_dir_acc, scr);
                noc_async_read_barrier();
                payload_page = reinterpret_cast<volatile uint32_t*>(scr)[dof];
            }

        // M1: ALL tiles (single + fat) consume the materialized depth-sorted
        // PACK2 slab from L1. M3: the slab record's word3 now carries the cull
        // mask (written by the cull writer), so there is NO separate cull_masks
        // bulk load — the blend compute reads the mask from rec[3].
        if (L_sub > 0) {
            const uint32_t rec_pages = (L_sub + 1u) >> 1;
            const uint32_t sc_flags = flags | MB_FLAG_L1_BULK;
            // Safe now that the bulk compute path has the MATH->UNPACK back-pressure
            // ack (the fast payload DMA otherwise raced slot-recycle vs MATH reads)
            // and the bulk CB is slot-aligned (no ring straddle on variable tiles).
            DeviceZoneScopedN("rd_l1_bulk");
            cb_reserve_back(CB_BUCKET_BULK, BULK_REC_SLOT);
            const uint32_t buck = get_write_ptr(CB_BUCKET_BULK);
            {
                const uint32_t page0 = payload_page;
                uint32_t pp = 0;
                while (pp < rec_pages) {
                    const uint32_t end = (pp + 64u < rec_pages) ? pp + 64u : rec_pages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(
                            page0 + q, subchunk_payload_acc,
                            buck + q * L1_PACK_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
            }
            mb_cb_commit_fence();
            cb_push_back(CB_BUCKET_BULK, BULK_REC_SLOT);

            cb_reserve_back(CB_MB_COUNTS, 1);
            {
                auto cnt_ptr = reinterpret_cast<volatile uint32_t*>(
                    get_write_ptr(CB_MB_COUNTS));
                cnt_ptr[0] = L_sub;
                cnt_ptr[1] = sc_flags;
            }
            cb_push_back(CB_MB_COUNTS, 1);
            continue;
        }
        cb_reserve_back(CB_MB_COUNTS, 1);
        {
            auto cnt_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
            cnt_ptr[0] = L_sub;
            cnt_ptr[1] = flags;
        }
        cb_push_back(CB_MB_COUNTS, 1);
        }  // end subchunk loop
    }
}
