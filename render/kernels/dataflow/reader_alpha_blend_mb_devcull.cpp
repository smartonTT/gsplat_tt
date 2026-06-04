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

#if defined(MB_RD_ROW_SUPPRESS_DPRINT) || defined(MB_C1_PAYLOAD_DEBUG)
#include "api/debug/dprint.h"
#endif

namespace {

// Iter 67: skip in-budget coeff push when SFPU cull already zeroed coverage or
// opacity is at/below contrib_floor (thr<0 sentinel on the cull path).
inline bool rd_row_suppress(uint32_t mask, float op_f, float contrib_floor) {
    if (mask == 0u) {
        return true;
    }
    return op_f <= contrib_floor;
}

constexpr uint32_t ATTR_PAGE_BYTES = 64;   // 16 fp32, 9 used
constexpr uint32_t IDS_PAGE_BYTES = 64;    // 16 uint32 ids per page
constexpr uint32_t RAMP_TILE_BYTES = 32 * 32 * 4;
constexpr uint32_t TILE_SIZE = 32;

constexpr uint32_t CB_XRAMP     = 0;
constexpr uint32_t CB_YRAMP     = 1;
constexpr uint32_t CB_MB_COEFF  = 2;
constexpr uint32_t CB_MB_COUNTS = 3;
constexpr uint32_t CB_SCR_IDS   = 4;   // reader-private: ids page scratch
constexpr uint32_t CB_SCR_ATTR  = 5;   // reader-private: attr page scratch
constexpr uint32_t CB_SCR_MASK  = 6;   // reader-private: 2x64B cull_masks page scratch (MB_SFPU_CULL)
constexpr uint32_t CB_CORE_TILES = 7;  // MB_RESIDENT: hand tile_ids_count to compute (no host arg)
constexpr uint32_t CB_BUCKET = 9;      // in-budget: L1 bucket + radix sort scratch (coeff stream)
constexpr uint32_t CB_BSORT  = 10;     // L1 sort scratch: in_idx[FIT] + out_idx[FIT] + counts[256]
constexpr uint32_t CB_BMASK  = 11;     // in-budget: whole-tile cull_masks (paired with CB_BUCKET)
constexpr uint32_t CB_BUCKET_BULK = 12; // overflow subchunk: bulk L1 records (no coeff stream)
constexpr uint32_t CB_BMASK_BULK  = 13; // overflow subchunk: bulk L1 cull_masks

// MB_COUNTS flags (slot 1): bit0=emit_tile, bit1=continue_blend, bit2=l1_bulk.
constexpr uint32_t MB_FLAG_EMIT = 1u;
constexpr uint32_t MB_FLAG_CONTINUE = 2u;
constexpr uint32_t MB_FLAG_L1_BULK = 4u;

// PACK2 (iter 50): two 32B splats per 64B DRAM page; splat g => page g/2, +32*(g&1).
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;

inline volatile uint32_t* l1_splat_words(uint32_t buck_base, uint32_t g) {
    return reinterpret_cast<volatile uint32_t*>(
        buck_base + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
}

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline uint32_t f_to_bits(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

// L1 store/visibility ordering fence. On Blackhole L1 is a small write-THROUGH
// cache, so a producer's stores reach L1, but a `fence` is needed so a freshly
// written CB row is coherent before cb_push_back signals the consumer (and so
// the compiler does not reorder the row stores past the push). == the
// invalidate_l1_cache() the runtime uses for cross-proc L1 handoff.
inline void mb_cb_commit_fence() {
    asm volatile("fence" ::: "memory");
}

// RESIDENT gather helper: read one fp32/uint32 element `elem` from a 64B-page
// (16-elem) DRAM-interleaved SoA buffer via `acc`, returning the raw 32-bit
// word. Used to gather each visible gaussian's attributes straight out of the
// resident proj_m_* / sort_* buffers (no host-built+uploaded attr table). The
// whole 64B page is fetched and the requested lane extracted.
template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    const uint32_t page = elem >> 4;     // / 16
    const uint32_t off = elem & 0xF;     // % 16
    noc_async_read_tile(page, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[off];
}

// Pipelined gather: number of 64B page slots per gaussian (a,b,c,px,py,op +
// 3 AoS color pages). Same nine reads as the per-gaussian read_soa_u32 path,
// but issued back-to-back into distinct L1 slots so ONE barrier covers the
// whole chunk instead of paying full NoC latency per field.
//
// S1 (GSPLAT_TT_BLEND_AOS / MB_BLEND_AOS): the projection/gather stage emits a
// single contiguous Array-of-Structs record per gaussian (proj_m_blendrec,
// {a,b,c,px,py,op,cr,cg,cb} padded to one 64B page). Each candidate then needs
// exactly ONE contiguous 64B NoC read instead of the 7-9 random SoA pages —
// txns 7-9 -> 1 and over-fetch 16x -> ~1.8x, with byte-identical fields.
constexpr uint32_t GATHER_FIELDS = 1;        // one packed AoS record page / gaussian
constexpr uint32_t GATHER_SLOT_BYTES = GATHER_FIELDS * 64u;   // 64B (AoS) or 576B (SoA) / gaussian

// AoS gather: issue ONE contiguous 64B record read per gaussian (page == g) into
// the chunk buffer. Reads left in flight; caller barriers once before consuming.
template <typename REC>
inline void issue_chunk_reads_aos(
    const uint32_t* gids, uint32_t take, uint32_t buf_addr, const REC& rec_acc) {
    for (uint32_t j = 0; j < take; ++j) {
        noc_async_read_tile(gids[j], rec_acc, buf_addr + j * GATHER_SLOT_BYTES);
    }
}


// Load one ids page worth of candidate ids ([<=16]) for the current tile into
// `out`, returning the count. Self-contained read+barrier (cheap: one page).
template <typename IDS>
inline uint32_t load_ids_chunk(
    const IDS& ids_acc, uint32_t id_start, uint32_t processed, uint32_t L,
    uint32_t scratch_addr, uint32_t* out) {
    constexpr uint32_t ids_per_page = IDS_PAGE_BYTES / 4;  // 16
    const uint32_t global_idx = id_start + processed;
    const uint32_t page_idx = global_idx / ids_per_page;
    const uint32_t in_page  = global_idx % ids_per_page;
    auto ids_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    noc_async_read_tile(page_idx, ids_acc, scratch_addr);
    noc_async_read_barrier();
    uint32_t take = ids_per_page - in_page;
    if (take > L - processed) take = L - processed;
    for (uint32_t i = 0; i < take; ++i) out[i] = ids_ptr[in_page + i];
    return take;
}

// SFPU-cull path: the 32-bit microblock mask is precomputed on the SFPU and
// kept resident in cull_masks, indexed identically to sort_sorted_ids (global
// candidate index == id_start + position). Each ids chunk (<=16, page-aligned
// by load_ids_chunk) maps to exactly one 64B/16-elem cull_masks page, so we
// prefetch that page alongside the chunk's attr reads (covered by the same
// barrier) and read the masks back with a pure integer load -> NO float and NO
// constrained-min on this data mover. Returns the in-page offset of the chunk.
template <typename Acc>
inline uint32_t load_mask_page(const Acc& acc, uint32_t global_idx, uint32_t take, uint32_t scratch_addr) {
    // global_idx == cull_base (16-aligned) + chunk-start position. A chunk holds
    // `take` (<=16) candidates starting at in-page offset off=(global_idx&0xF).
    // The consumer reads mask_ptr[off + j] for j in [0,take), so it only touches
    // a SECOND 64B/16-elem page when off+take > 16 (the chunk straddles a page
    // boundary). Issue page pg unconditionally and pg+1 ONLY on a straddle: the
    // first chunk of every tile (off==0) and all chunks of 16-aligned tiles never
    // straddle, so this drops one redundant random DRAM read on those chunks.
    // Byte-identical to the prior 2-page load for the bytes the consumer reads.
    const uint32_t off = global_idx & 0xF;
    const uint32_t pg = global_idx >> 4;
    noc_async_read_tile(pg, acc, scratch_addr);
    if (off + take > 16u) {
        noc_async_read_tile(pg + 1u, acc, scratch_addr + IDS_PAGE_BYTES);
    }
    return off;
}

inline uint32_t pack_fp32_unorm16(float v) {
    if (v <= 0.0f) return 0u;
    if (v >= 1.0f) return 65535u;
    return static_cast<uint32_t>(v * 65535.0f + 0.5f);
}

// PACK2 pack from 64B proj_m_blendrec (sort_bin layout) into one 32B L1 splat.
inline void pack_blendrec_to_l1(
    volatile uint32_t* aos, volatile uint32_t* splat,
    float tx_tile, float ty_tile) {
    float mx = bits_to_f(aos[3]);
    float my = bits_to_f(aos[4]);
    mx -= tx_tile;
    my -= ty_tile;
    splat[0] = aos[0];
    splat[1] = aos[1];
    splat[2] = aos[2];
    splat[3] = aos[9];
    splat[4] = f_to_bits(mx);
    splat[5] = f_to_bits(my);
    const float op = bits_to_f(aos[5]);
    const float cr = bits_to_f(aos[6]);
    const float cg = bits_to_f(aos[7]);
    const float cb = bits_to_f(aos[8]);
    splat[6] = pack_fp32_unorm16(op) | (pack_fp32_unorm16(cr) << 16);
    splat[7] = pack_fp32_unorm16(cg) | (pack_fp32_unorm16(cb) << 16);
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
    const uint32_t cull_masks_addr = get_arg_val<uint32_t>(17);  // resident cull_masks
    const uint32_t cull_base_addr  = get_arg_val<uint32_t>(18);  // per-tile page-aligned mask base
    const uint32_t blendrec_addr   = get_arg_val<uint32_t>(19);  // resident proj_m_blendrec (AoS)
    const uint32_t tile_recs_addr  = get_arg_val<uint32_t>(20);  // resident sort_tile_recs (dense)
    const uint32_t bucket_meta_addr= get_arg_val<uint32_t>(21);  // resident sort_bucket_meta (start,count)
    const uint32_t l1_recs_addr    = get_arg_val<uint32_t>(22);  // M0: pre-sized 32B record bucket
    const uint32_t subchunk_meta_addr = get_arg_val<uint32_t>(23);  // blend_subchunk_meta [dir_base,num_sc]
    const uint32_t subchunk_payload_addr = get_arg_val<uint32_t>(24);
    const uint32_t subchunk_dir_addr = get_arg_val<uint32_t>(25);

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
    constexpr auto cull_masks_args = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto cull_base_args  = TensorAccessorArgs<cull_masks_args.next_compile_time_args_offset()>();
    constexpr auto blendrec_args   = TensorAccessorArgs<cull_base_args.next_compile_time_args_offset()>();
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
    const auto cull_masks_acc = TensorAccessor(cull_masks_args, cull_masks_addr, IDS_PAGE_BYTES);
    const auto cull_base_acc  = TensorAccessor(cull_base_args,  cull_base_addr,  64);
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
        // Per-tile PAGE-ALIGNED base of this tile's masks in cull_masks (the cull
        // writer uses the same base). cull_masks is NOT indexed by the dense sort
        // range (those id_start values are not 16-aligned, and unaligned NoC->DRAM
        // writes get shifted), so the mask for tile-local candidate p is at
        // cull_masks[cull_base + p].
        const uint32_t cull_base = read_soa_u32(cull_base_acc, tile_id, get_write_ptr(CB_SCR_IDS));
        // T2/T3: serve this tile from its DENSE L1-resident record bucket — NO
        // per-candidate attr gather. Load the whole bucket once (bulk, batched
        // barriers), STABLE depth-sort the candidates IN L1 (index permutation,
        // records stay put), and emit coeff rows reading records from L1. The
        // stable LSD radix reproduces the DRAM radix order, so cull_masks (still
        // depth-sorted in DRAM, indexed cull_base+k) stays aligned. Tiles whose
        // record set exceeds MB_BUCKET_FIT*64B of L1 fall through to the gather.
        uint32_t rec_start = 0;
        uint32_t Lb = 0;
        {
            const uint32_t e0 = tile_id * 2u;
            const uint32_t pg = e0 >> 4;
            const uint32_t off = e0 & 0xF;
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            noc_async_read_tile(pg, bucket_meta_acc, scr);
            noc_async_read_barrier();
            auto bmp = reinterpret_cast<volatile uint32_t*>(scr);
            rec_start = bmp[off];
            Lb = bmp[off + 1u];  // dense bucket count (off even -> off+1 same page)
        }
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
            uint32_t payload_page = 0;
            if (num_subchunks > 1u) {
                const uint32_t de = (dir_base + sc) * 4u;
                const uint32_t dpg = de >> 4;
                const uint32_t dof = de & 0xF;
                const uint32_t scr = get_write_ptr(CB_SCR_IDS);
                noc_async_read_tile(dpg, subchunk_dir_acc, scr);
                noc_async_read_barrier();
                payload_page = reinterpret_cast<volatile uint32_t*>(scr)[dof];
            }
            const uint32_t id_start_sc = id_start + sc_off;
            const uint32_t cull_base_sc = cull_base + sc_off;

        if (num_subchunks == 1u && Lb > 0 && Lb <= MB_BUCKET_FIT) {
            const uint32_t L = Lb;
            const uint32_t npages = (L + 1u) >> 1;
            cb_reserve_back(CB_BUCKET, npages);
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                const uint32_t page0 = tile_id * (MB_BUCKET_FIT >> 1);
                uint32_t pp = 0;
                while (pp < npages) {
                    const uint32_t end = (pp + 64u < npages) ? pp + 64u : npages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(page0 + q, l1_recs_acc,
                                            buck + q * L1_PACK_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
            }
            const uint32_t bs = get_write_ptr(CB_BSORT);
            uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
            uint32_t* idxB = idxA + MB_BUCKET_FIT;
            uint32_t* cnt  = idxB + MB_BUCKET_FIT;
            auto key_of = [&](uint32_t idx) -> uint32_t {
                return l1_splat_words(buck, idx)[3];
            };
            uint32_t* sorted;
            if (L <= 16u) {
                for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
                for (uint32_t i = 1; i < L; ++i) {
                    const uint32_t tmp = idxA[i];
                    const uint32_t ki = key_of(tmp);
                    uint32_t j = i;
                    while (j > 0 && key_of(idxA[j - 1]) > ki) { idxA[j] = idxA[j - 1]; --j; }
                    idxA[j] = tmp;
                }
                sorted = idxA;
            } else {
                for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
                uint32_t* cur = idxA;
                uint32_t* nxt = idxB;
                for (uint32_t byte = 0; byte < 4u; ++byte) {
                    const uint32_t shift = byte * 8u;
                    for (uint32_t c = 0; c < 256u; ++c) cnt[c] = 0;
                    for (uint32_t i = 0; i < L; ++i) cnt[(key_of(cur[i]) >> shift) & 0xFFu]++;
                    uint32_t sum = 0;
                    for (uint32_t c = 0; c < 256u; ++c) { const uint32_t t = cnt[c]; cnt[c] = sum; sum += t; }
                    for (uint32_t i = 0; i < L; ++i) {
                        const uint32_t b = (key_of(cur[i]) >> shift) & 0xFFu;
                        nxt[cnt[b]++] = cur[i];
                    }
                    uint32_t* t = cur; cur = nxt; nxt = t;
                }
                sorted = cur;  // even pass count -> back in idxA
            }
            // Bulk-load this tile's WHOLE cull_masks region into L1 ONCE (cull_base
            // is 16-aligned -> mask[k] == L1[k]) with batched barriers. Tile-local
            // L1 cull (step D) fills the L1-interleaved buffer — no read spin.
            const uint32_t mpages = (L + 15u) >> 4;
            cb_reserve_back(CB_BMASK, mpages);
            const uint32_t bmask = get_write_ptr(CB_BMASK);
            auto bmptr = reinterpret_cast<volatile uint32_t*>(bmask);
            {
                const uint32_t mpg0 = cull_base >> 4;
                uint32_t pp = 0;
                while (pp < mpages) {
                    const uint32_t end = (pp + 64u < mpages) ? pp + 64u : mpages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(mpg0 + q, cull_masks_acc, bmask + q * IDS_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
#ifndef MB_TILE_L1_MASKS
                for (volatile int _s = 0; _s < (MB_CULL_SPIN); ++_s) { }
#endif
            }
            // CB_BUCKET/CB_BMASK are reader-private L1 scratch only (reserve +
            // get_write_ptr, never push/pop). Pushing them wedged the ring when
            // in-budget tiles shared CB_BUCKET with overflow bulk (db9dcd0) or
            // when back-to-back in-budget tiles ran ahead of compute drain.
            uint32_t emit_n = 0;
#if defined(MB_RD_ROW_SUPPRESS_DPRINT)
            uint32_t suppress_mask0 = 0;
            uint32_t suppress_op = 0;
#endif
            for (uint32_t k = 0; k < L; ++k) {
                const uint32_t mask = bmptr[k];
                auto recp32 = l1_splat_words(buck, sorted[k]);
                constexpr float kUnormInv = 1.0f / 65535.0f;
                const float op_f =
                    static_cast<float>(recp32[6] & 0xffffu) * kUnormInv;
                if (rd_row_suppress(mask, op_f, contrib_floor)) {
#if defined(MB_RD_ROW_SUPPRESS_DPRINT)
                    if (mask == 0u) {
                        ++suppress_mask0;
                    } else {
                        ++suppress_op;
                    }
#endif
                    continue;
                }
                ++emit_n;
            }
            // Count before coeff stream: CB_MB_COEFF depth is 8; compute must
            // drain while the reader emits (see blend_device.cpp cb_cfg).
            cb_reserve_back(CB_MB_COUNTS, 1);
            {
                auto cnt_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
                cnt_ptr[0] = emit_n;
                cnt_ptr[1] = MB_FLAG_EMIT;  // emit tile (single in-budget subchunk)
            }
            cb_push_back(CB_MB_COUNTS, 1);
            {
            // MEASUREMENT zone: the per-candidate emit loop (repack record ->
            // 64B coeff row -> CB_MB_COEFF push -> fence) for ONE in-budget tile.
            // Axis (B): reader-emit cost. One zone per in-budget tile (exclusive
            // with rd_overflow), so per-core marker count stays bounded; durations
            // are summed in post-processing across the 30-view Tracy CSV.
            DeviceZoneScopedN("rd_bk_emit");
            for (uint32_t k = 0; k < L; ++k) {
                const uint32_t idx = sorted[k];
                auto recp32 = l1_splat_words(buck, idx);
                const uint32_t cov_a_bits = recp32[0];
                const uint32_t cov_b_bits = recp32[1];
                const uint32_t cov_c_bits = recp32[2];
                const float mx_f = bits_to_f(recp32[4]);
                const float my_f = bits_to_f(recp32[5]);
                const uint32_t w6 = recp32[6], w7 = recp32[7];
                constexpr float kUnormInv = 1.0f / 65535.0f;
                const float op_f = static_cast<float>(w6 & 0xffffu) * kUnormInv;
                const float cr_f = static_cast<float>(w6 >> 16)     * kUnormInv;
                const float cg_f = static_cast<float>(w7 & 0xffffu) * kUnormInv;
                const float cb_f = static_cast<float>(w7 >> 16)     * kUnormInv;
                const uint32_t op_bits = f_to_bits(op_f);
                const uint32_t mask = bmptr[k];  // 16-aligned cull_base -> mask[k]==L1[k]
                if (rd_row_suppress(mask, op_f, contrib_floor)) {
                    continue;
                }
                // Reconstruct the absolute mean so the inline-mask path and the
                // emitted row's mxl = (mean - tx_tile) both work unchanged.
                const float mean_x  = mx_f + tx_tile;
                const float mean_y  = my_f + ty_tile;
                const uint32_t cr = f_to_bits(cr_f);
                const uint32_t cg = f_to_bits(cg_f);
                const uint32_t cb = f_to_bits(cb_f);
                cb_reserve_back(CB_MB_COEFF, 1);
                auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COEFF));
                const uint32_t mxl_bits = f_to_bits(mean_x - tx_tile);
                const uint32_t myl_bits = f_to_bits(mean_y - ty_tile);
                row[0] = cov_a_bits;
                row[1] = cov_b_bits;
                row[2] = cov_c_bits;
                row[3] = mxl_bits;
                row[4] = myl_bits;
                row[5] = 0u;
                row[6] = op_bits;
                row[7] = cr;
                row[8] = cg;
                row[9] = cb;
                row[10] = mask;
                row[11] = 0u;
                row[12] = 0u;
                row[13] = 0u;
                row[14] = 0u;
                row[15] = 0u;
                // Order all row payload stores before cb_push_back's stream-reg
                // increment (write-through L1) so a consumer that observes the push
                // also observes the full row. The actual fast-producer race (UNPACK
                // recycling the slot before the slow MATH read) is fixed by the
                // MATH->UNPACK back-pressure ack in the compute kernel.
                mb_cb_commit_fence();
                cb_push_back(CB_MB_COEFF, 1);
            }
#if defined(MB_RD_ROW_SUPPRESS_DPRINT)
            DPRINT << "RDSUP t=" << tile_id << " L=" << L << " emit=" << emit_n
                   << " m0=" << suppress_mask0 << " op=" << suppress_op << ENDL();
#endif
            }  // end rd_bk_emit zone
            continue;
        }

        // Overflow: PACK2 bulk L1 + cp_l1_blend (iter 51). Step C1 payload path
        // gated behind num_subchunks>1 once dir/mat ordering verified on device.
        if (L_sub > 0) {
            const uint32_t rec_pages = (L_sub + 1u) >> 1;
            const uint32_t mpages_mask = (L_sub + 15u) >> 4;
            const uint32_t sc_flags = flags | MB_FLAG_L1_BULK;
            // C1: overflow subchunks DMA prebuilt PACK2 (iter 85: mat pack overlap fix).
            const bool use_payload = (num_subchunks > 1u);

            DeviceZoneScopedN("rd_l1_bulk");
            cb_reserve_back(CB_BMASK_BULK, mpages_mask);
            const uint32_t bmask = get_write_ptr(CB_BMASK_BULK);
            {
                const uint32_t mpg0 = cull_base_sc >> 4;
                uint32_t pp = 0;
                while (pp < mpages_mask) {
                    const uint32_t end = (pp + 64u < mpages_mask) ? pp + 64u : mpages_mask;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(mpg0 + q, cull_masks_acc,
                                            bmask + q * IDS_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
#ifndef MB_TILE_L1_MASKS
                for (volatile int _s = 0; _s < (MB_CULL_SPIN); ++_s) { }
#endif
            }

#if defined(MB_C1_PAYLOAD_DEBUG)
            // iter 84: compare DRAM payload vs blendrec gather (first 16 PACK2 words).
            {
                static uint32_t c1dbg_tiles = 0;
                if (num_subchunks > 1u && L_sub > 0u && c1dbg_tiles < 3u) {
                    uint32_t dir_pg = 0;
                    uint32_t dir_l = 0;
                    uint32_t dir_fl = 0;
                    {
                        const uint32_t de = (dir_base + sc) * 4u;
                        const uint32_t dpg = de >> 4;
                        const uint32_t dof = de & 0xF;
                        const uint32_t dscr = get_write_ptr(CB_SCR_IDS);
                        noc_async_read_tile(dpg, subchunk_dir_acc, dscr);
                        noc_async_read_barrier();
                        auto dp = reinterpret_cast<volatile uint32_t*>(dscr);
                        dir_pg = dp[dof];
                        dir_l = dp[dof + 1u];
                        dir_fl = dp[dof + 2u];
                    }
                    const uint32_t cmp_splats =
                        (L_sub < 2u) ? L_sub : 2u;  // 16 words = 2 splats
                    const uint32_t cmp_words = cmp_splats * 8u;
                    const uint32_t pay_scr = get_write_ptr(CB_SCR_ATTR);
                    const uint32_t aos_scr = pay_scr + L1_PACK_PAGE_BYTES;
                    uint32_t pay_w[16];
                    uint32_t gat_w[16];
                    for (uint32_t w = 0; w < cmp_words; ++w) {
                        pay_w[w] = 0u;
                        gat_w[w] = 0u;
                    }
                    {
                        const uint32_t pp0 = payload_page;
                        noc_async_read_tile(pp0, subchunk_payload_acc, pay_scr);
                        noc_async_read_barrier();
                        auto pp = reinterpret_cast<volatile uint32_t*>(pay_scr);
                        for (uint32_t w = 0; w < cmp_words; ++w) {
                            pay_w[w] = pp[w];
                        }
                    }
                    uint32_t gids[2];
                    const uint32_t ids_scr = get_write_ptr(CB_SCR_IDS);
                    const uint32_t take0 = load_ids_chunk(
                        ids_acc, id_start_sc, 0, cmp_splats, ids_scr, gids);
                    issue_chunk_reads_aos(gids, take0, aos_scr, blendrec_acc);
                    noc_async_read_barrier();
                    for (uint32_t j = 0; j < take0; ++j) {
                        volatile uint32_t splat[8];
                        pack_blendrec_to_l1(
                            reinterpret_cast<volatile uint32_t*>(
                                aos_scr + j * GATHER_SLOT_BYTES),
                            splat, tx_tile, ty_tile);
                        for (uint32_t w = 0; w < 8u; ++w) {
                            gat_w[j * 8u + w] = splat[w];
                        }
                    }
                    uint32_t mism = 0;
                    uint32_t fk = 0xFFFFFFFFu;
                    uint32_t fw = 0xFFFFFFFFu;
                    for (uint32_t w = 0; w < cmp_words; ++w) {
                        if (pay_w[w] != gat_w[w]) {
                            ++mism;
                            if (fk == 0xFFFFFFFFu) {
                                fk = w >> 3;
                                fw = w & 7u;
                            }
                        }
                    }
                    DPRINT << "C1DBG t=" << tile_id << " sc=" << sc << " L=" << L_sub
                           << " pay_pg=" << payload_page << " dir_pg=" << dir_pg
                           << " dir_L=" << dir_l << " dir_fl=" << dir_fl
                           << " pg_eq=" << (payload_page == dir_pg)
                           << " mism=" << mism << " fk=" << fk << " fw=" << fw
                           << ENDL();
                    if (mism > 0u) {
                        DPRINT << "C1DBG pay0=" << pay_w[0] << " gat0=" << gat_w[0]
                               << " pay7=" << pay_w[7] << " gat7=" << gat_w[7]
                               << ENDL();
                    }
                    ++c1dbg_tiles;
                }
            }
#endif

            cb_reserve_back(CB_BUCKET_BULK, rec_pages);
            const uint32_t buck = get_write_ptr(CB_BUCKET_BULK);
            if (use_payload) {
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
            } else {
                const uint32_t ids_scr = get_write_ptr(CB_SCR_IDS);
                const uint32_t aos_base = get_write_ptr(CB_SCR_ATTR);
                uint32_t gids[16];
                uint32_t processed = 0;
                while (processed < L_sub) {
                    const uint32_t take = load_ids_chunk(
                        ids_acc, id_start_sc, processed, L_sub, ids_scr, gids);
                    issue_chunk_reads_aos(gids, take, aos_base, blendrec_acc);
                    noc_async_read_barrier();
                    for (uint32_t j = 0; j < take; ++j) {
                        pack_blendrec_to_l1(
                            reinterpret_cast<volatile uint32_t*>(
                                aos_base + j * GATHER_SLOT_BYTES),
                            l1_splat_words(buck, processed + j),
                            tx_tile, ty_tile);
                    }
                    processed += take;
                }
            }
            mb_cb_commit_fence();
            cb_push_back(CB_BMASK_BULK, mpages_mask);
            cb_push_back(CB_BUCKET_BULK, rec_pages);

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
