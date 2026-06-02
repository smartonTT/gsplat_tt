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
#if defined(MB_SFPU_CULL_DEBUG) || defined(FUSE_AB_ROW) || defined(MB_BUCKET_MASK_DEBUG)
#include "api/debug/dprint.h"
#endif

namespace {

constexpr uint32_t NUM_MB = 32;
constexpr uint32_t ATTR_PAGE_BYTES = 64;   // 16 fp32, 9 used
constexpr uint32_t COEFF_ROW_BYTES = 64;   // 16 fp32 row the compute kernel reads
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
constexpr uint32_t CB_TILE_MASKS = 8;  // MB_TILE_MASK_L1: whole-tile cull_masks region, bulk-loaded once/tile
#ifdef MB_TILE_BUCKET
constexpr uint32_t CB_BUCKET = 9;      // L1-resident dense record bucket for this tile
constexpr uint32_t CB_BSORT  = 10;     // L1 sort scratch: in_idx[FIT] + out_idx[FIT] + counts[256]
constexpr uint32_t CB_BMASK  = 11;     // L1-resident whole-tile cull_masks (bulk-loaded once/tile)
#ifndef MB_BUCKET_FIT
#define MB_BUCKET_FIT 8192u            // max candidates/tile served from L1 (else gather fallback)
#endif
#endif

constexpr float kInf = 1e30f;

inline int ifloor(float v) {
    int i = static_cast<int>(v);
    if (static_cast<float>(i) > v) {
        i -= 1;
    }
    return i;
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
[[maybe_unused]] inline void mb_cb_commit_fence() {
    asm volatile("fence" ::: "memory");
}

#ifdef MB_RESIDENT
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
#ifdef MB_BLEND_AOS
constexpr uint32_t GATHER_FIELDS = 1;        // one packed AoS record page / gaussian
#else
constexpr uint32_t GATHER_FIELDS = 9;
#endif
constexpr uint32_t GATHER_SLOT_BYTES = GATHER_FIELDS * 64u;   // 64B (AoS) or 576B (SoA) / gaussian

#ifdef MB_BLEND_AOS
// AoS gather: issue ONE contiguous 64B record read per gaussian (page == g) into
// the chunk buffer. Reads left in flight; caller barriers once before consuming.
template <typename REC>
inline void issue_chunk_reads_aos(
    const uint32_t* gids, uint32_t take, uint32_t buf_addr, const REC& rec_acc) {
    for (uint32_t j = 0; j < take; ++j) {
        noc_async_read_tile(gids[j], rec_acc, buf_addr + j * GATHER_SLOT_BYTES);
    }
}
#endif

// Issue (no barrier) all SoA page reads for `take` gaussians into `buf_addr`.
// Reads are left in flight; caller barriers once before consuming `buf_addr`.
template <typename A, typename B, typename C, typename PX, typename PY,
          typename OP, typename COL>
inline void issue_chunk_reads(
    const uint32_t* gids, uint32_t take, uint32_t buf_addr,
    const A& a_acc, const B& b_acc, const C& c_acc, const PX& px_acc,
    const PY& py_acc, const OP& op_acc, const COL& col_acc) {
    for (uint32_t j = 0; j < take; ++j) {
        const uint32_t g = gids[j];
        const uint32_t s = buf_addr + j * GATHER_SLOT_BYTES;
        const uint32_t pg = g >> 4;            // scalar SoA page (elem == g)
        noc_async_read_tile(pg, a_acc,  s + 0u * 64u);
        noc_async_read_tile(pg, b_acc,  s + 1u * 64u);
        noc_async_read_tile(pg, c_acc,  s + 2u * 64u);
        noc_async_read_tile(pg, px_acc, s + 3u * 64u);
        noc_async_read_tile(pg, py_acc, s + 4u * 64u);
        noc_async_read_tile(pg, op_acc, s + 5u * 64u);
        // AoS M*3 colors: r/g/b are 3 CONSECUTIVE elems (e0,e0+1,e0+2), so they
        // span at most TWO 16-elem pages and usually ONE (e0%16 <= 13). Read page
        // p0 into slot 6 and, only when r/g/b straddle a page boundary, the next
        // page into slot 7 (contiguous 32-elem window). Collapses the old 3 reads
        // of the same page to 1 (common) or 2 (boundary) -- byte-identical data,
        // ~2 fewer NoC reads/candidate. Consumer indexes the window by (elem-base).
        const uint32_t e0 = g * 3u;
        const uint32_t cpg0 = e0 >> 4;
        const uint32_t cpg1 = (e0 + 2u) >> 4;
        noc_async_read_tile(cpg0, col_acc, s + 6u * 64u);
        if (cpg1 != cpg0) {
            noc_async_read_tile(cpg1, col_acc, s + 7u * 64u);
        }
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

#ifdef MB_SFPU_CULL
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
#endif
#endif

// Reproduce build_gaussian_major_tile's per-gaussian microblock cull EXACTLY.
// Returns the 32-bit microblock-coverage mask. tx_tile/ty_tile are the float
// tile-origin pixel coords (tx*32, ty*32).
// [[maybe_unused]]: under GSPLAT_TT_SFPU_CULL the mask is precomputed on the
// SFPU (resident cull_masks) and this soft-float cull is bypassed.
[[maybe_unused]] inline uint32_t compute_microblock_mask(
    float a, float b, float c, float mean_x, float mean_y, float opacity,
    float tx_tile, float ty_tile, float contrib_floor, bool cull_disabled) {
    float det = a * c - b * b;
    if (det < 1e-6f) {
        det = 1e-6f;
    }
    const float ci_a = c / det;
    const float ci_b = -b / det;
    const float ci_c = a / det;

    if (opacity <= contrib_floor) {
        return 0u;  // peak alpha below floor everywhere.
    }

    const float log_thresh = __builtin_logf(contrib_floor / opacity);  // <= 0
    const float rd = __builtin_sqrtf(-2.0f * log_thresh);
    const float a_pos = a > 0.0f ? a : 0.0f;
    const float c_pos = c > 0.0f ? c : 0.0f;
    const float x_half = rd * __builtin_sqrtf(a_pos);
    const float y_half = rd * __builtin_sqrtf(c_pos);

    const float bb_x_min = mean_x - x_half;
    const float bb_x_max = mean_x + x_half;
    const float bb_y_min = mean_y - y_half;
    const float bb_y_max = mean_y + y_half;
    int mx_lo = ifloor((bb_x_min - tx_tile) * 0.125f);
    int mx_hi = ifloor((bb_x_max - tx_tile) * 0.125f);
    int my_lo = ifloor((bb_y_min - ty_tile) * 0.25f);
    int my_hi = ifloor((bb_y_max - ty_tile) * 0.25f);
    if (mx_lo < 0) mx_lo = 0;
    if (mx_hi > 3) mx_hi = 3;
    if (my_lo < 0) my_lo = 0;
    if (my_hi > 7) my_hi = 7;
    if (mx_lo > mx_hi || my_lo > my_hi) {
        return 0u;
    }

    const float thresh_m2 = -2.0f * log_thresh;
    const float ci_a_safe = ci_a > 1e-12f ? ci_a : 1e-12f;
    const float ci_c_safe = ci_c > 1e-12f ? ci_c : 1e-12f;
    // Soft-float divides on the RISC data-mover dominate the cull. The inner
    // constrained-min recomputed two divides PER microblock, but each divide is
    // loop-invariant: v_star = -ci_b*u_fix/ci_c_safe depends only on the column
    // (mx, via u_fix) and u_star = -ci_b*v_fix/ci_a_safe only on the row (my,
    // via v_fix). Hoisting them — plus the u_fix/v_fix-only polynomial terms —
    // out of the inner loop is BIT-IDENTICAL (same operands, same op order, so
    // same rounding => same mask) and collapses up-to-(2*Wmb*Hmb) divides per
    // gaussian to (Wmb + Hmb).
    const float two_ci_b = 2.0f * ci_b;

    // Per-column (mx-invariant across rows) precompute.
    bool  col_x_inside[4] = {false, false, false, false};
    float col_u_lo[4]     = {0.0f, 0.0f, 0.0f, 0.0f};
    float col_u_hi[4]     = {0.0f, 0.0f, 0.0f, 0.0f};
    float col_v_star[4]   = {0.0f, 0.0f, 0.0f, 0.0f};  // unclamped -ci_b*u_fix/ci_c_safe
    float col_t1v[4]      = {0.0f, 0.0f, 0.0f, 0.0f};  // ci_a*u_fix*u_fix
    float col_t2v[4]      = {0.0f, 0.0f, 0.0f, 0.0f};  // (2*ci_b)*u_fix
    for (int mx = mx_lo; mx <= mx_hi; ++mx) {
        const float mb_ox = tx_tile + static_cast<float>(mx * 8);
        const float u_lo = mb_ox - mean_x;
        const float u_hi = u_lo + 8.0f;
        const bool x_inside = (u_lo <= 0.0f) && (0.0f <= u_hi);
        col_x_inside[mx] = x_inside;
        col_u_lo[mx] = u_lo;
        col_u_hi[mx] = u_hi;
        if (!cull_disabled && !x_inside) {
            const float u_fix = (u_lo > 0.0f) ? u_lo : u_hi;
            col_v_star[mx] = -ci_b * u_fix / ci_c_safe;
            col_t1v[mx] = ci_a * u_fix * u_fix;
            col_t2v[mx] = two_ci_b * u_fix;
        }
    }

    uint32_t mask = 0u;
    for (int my = my_lo; my <= my_hi; ++my) {
        const float mb_oy = ty_tile + static_cast<float>(my * 4);
        const float v_lo = mb_oy - mean_y;
        const float v_hi = v_lo + 4.0f;
        const bool y_inside = (v_lo <= 0.0f) && (0.0f <= v_hi);
        const float v_fix = (v_lo > 0.0f) ? v_lo : v_hi;
        // Per-row (my-invariant across columns) precompute.
        float row_u_star = 0.0f;  // unclamped -ci_b*v_fix/ci_a_safe
        float row_t3h = 0.0f;     // ci_c*v_fix*v_fix
        if (!cull_disabled && !y_inside) {
            row_u_star = -ci_b * v_fix / ci_a_safe;
            row_t3h = ci_c * v_fix * v_fix;
        }
        for (int mx = mx_lo; mx <= mx_hi; ++mx) {
            const bool x_inside = col_x_inside[mx];

            float m2_min;
            if (cull_disabled || (x_inside && y_inside)) {
                m2_min = 0.0f;
            } else {
                float m2_v = kInf;
                if (!x_inside) {
                    float v_star = col_v_star[mx];
                    if (v_star < v_lo) v_star = v_lo;
                    if (v_star > v_hi) v_star = v_hi;
                    m2_v = col_t1v[mx] + col_t2v[mx] * v_star +
                           ci_c * v_star * v_star;
                }
                float m2_h = kInf;
                if (!y_inside) {
                    float u_star = row_u_star;
                    if (u_star < col_u_lo[mx]) u_star = col_u_lo[mx];
                    if (u_star > col_u_hi[mx]) u_star = col_u_hi[mx];
                    m2_h = ci_a * u_star * u_star + two_ci_b * u_star * v_fix +
                           row_t3h;
                }
                m2_min = (m2_v < m2_h) ? m2_v : m2_h;
            }
            if (m2_min <= thresh_m2) {
                mask |= (1u << ((my << 2) | mx));
            }
        }
    }
    return mask;
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("blend_rd");  // Tracy device-timeline stage label (blend reader/devcull)
#ifdef MB_RESIDENT
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
#ifdef MB_SFPU_CULL
    const uint32_t cull_masks_addr = get_arg_val<uint32_t>(17);  // resident cull_masks
    const uint32_t cull_base_addr  = get_arg_val<uint32_t>(18);  // per-tile page-aligned mask base
#ifdef MB_BLEND_AOS
    const uint32_t blendrec_addr   = get_arg_val<uint32_t>(19);  // resident proj_m_blendrec (AoS)
#ifdef MB_TILE_BUCKET
    const uint32_t tile_recs_addr  = get_arg_val<uint32_t>(20);  // resident sort_tile_recs (dense)
    const uint32_t bucket_meta_addr= get_arg_val<uint32_t>(21);  // resident sort_bucket_meta (start,count)
#endif
#endif
#endif

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
#ifdef MB_SFPU_CULL
    constexpr auto cull_masks_args = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();
    constexpr auto cull_base_args  = TensorAccessorArgs<cull_masks_args.next_compile_time_args_offset()>();
#ifdef MB_BLEND_AOS
    constexpr auto blendrec_args   = TensorAccessorArgs<cull_base_args.next_compile_time_args_offset()>();
#ifdef MB_TILE_BUCKET
    constexpr auto tile_recs_args  = TensorAccessorArgs<blendrec_args.next_compile_time_args_offset()>();
    constexpr auto bucket_meta_args= TensorAccessorArgs<tile_recs_args.next_compile_time_args_offset()>();
#endif
#endif
#endif

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
#ifdef MB_BLEND_AOS
    // Under AoS the per-component SoA gather is replaced by proj_m_blendrec; the
    // SoA accessors stay bound (ABI parity) but are unused on this path.
    (void)a_acc; (void)b_acc; (void)c_acc; (void)px_acc; (void)py_acc;
    (void)op_acc; (void)col_acc;
#endif
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
#ifdef MB_SFPU_CULL
    const auto cull_masks_acc = TensorAccessor(cull_masks_args, cull_masks_addr, IDS_PAGE_BYTES);
    const auto cull_base_acc  = TensorAccessor(cull_base_args,  cull_base_addr,  64);
#ifdef MB_BLEND_AOS
    // proj_m_blendrec: one 64B AoS record page per gaussian (page index == g).
    const auto blendrec_acc   = TensorAccessor(blendrec_args,   blendrec_addr,   SOA_PAGE_BYTES);
#ifdef MB_TILE_BUCKET
    // sort_tile_recs: DENSE per-tile bucket, one 64B record per kept candidate
    // {a,b,c,px,py,op,r,g,b,depth} (page index == bucket slot). sort_bucket_meta:
    // per-tile (start,count) u32 pair.
    const auto tile_recs_acc   = TensorAccessor(tile_recs_args,   tile_recs_addr,   SOA_PAGE_BYTES);
    const auto bucket_meta_acc = TensorAccessor(bucket_meta_args, bucket_meta_addr, 64);
#endif
#endif
    // Cull math (and thus these scalars) moved to the SFPU cull pass; the mask
    // is now read precomputed. Keep the args for ABI/signature parity.
#if !defined(MB_SFPU_CULL_DEBUG)
    (void)contrib_floor;
    (void)cull_disabled;
#endif
#endif
#else
    const uint32_t attrs_addr     = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr       = get_arg_val<uint32_t>(1);
    const uint32_t ids_off_addr   = get_arg_val<uint32_t>(2);
    const uint32_t xramp_addr     = get_arg_val<uint32_t>(3);
    const uint32_t yramp_addr     = get_arg_val<uint32_t>(4);
    const uint32_t tile_ids_addr  = get_arg_val<uint32_t>(5);
    const uint32_t tile_ids_start = get_arg_val<uint32_t>(6);
    const uint32_t tile_ids_count = get_arg_val<uint32_t>(7);
    const uint32_t tiles_x        = get_arg_val<uint32_t>(8);
    const float contrib_floor     = bits_to_f(get_arg_val<uint32_t>(9));
    const bool cull_disabled      = get_arg_val<uint32_t>(10) != 0;

    constexpr auto attrs_args    = TensorAccessorArgs<0>();
    constexpr auto ids_args      = TensorAccessorArgs<attrs_args.next_compile_time_args_offset()>();
    constexpr auto ids_off_args  = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto xramp_args    = TensorAccessorArgs<ids_off_args.next_compile_time_args_offset()>();
    constexpr auto yramp_args    = TensorAccessorArgs<xramp_args.next_compile_time_args_offset()>();
    constexpr auto tile_ids_args = TensorAccessorArgs<yramp_args.next_compile_time_args_offset()>();

    const auto attrs_acc    = TensorAccessor(attrs_args,    attrs_addr,    ATTR_PAGE_BYTES);
    const auto ids_acc      = TensorAccessor(ids_args,      ids_addr,      IDS_PAGE_BYTES);
    const auto ids_off_acc  = TensorAccessor(ids_off_args,  ids_off_addr,  /*page=*/4);
    const auto xramp_acc    = TensorAccessor(xramp_args,    xramp_addr,    RAMP_TILE_BYTES);
    const auto yramp_acc    = TensorAccessor(yramp_args,    yramp_addr,    RAMP_TILE_BYTES);
    const auto tile_ids_acc = TensorAccessor(tile_ids_args, tile_ids_addr, 64);
#endif

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
#ifdef MB_RESIDENT
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
#else
        {
            const uint32_t off_scratch = get_write_ptr(CB_SCR_IDS);
            auto off_ptr = reinterpret_cast<volatile uint32_t*>(off_scratch);
            uint64_t off_noc = get_noc_addr(tile_id, ids_off_acc);
            noc_async_read(off_noc, off_scratch, 4);
            noc_async_read_barrier();
            id_start = off_ptr[0];
            off_noc = get_noc_addr(tile_id + 1, ids_off_acc);
            noc_async_read(off_noc, off_scratch, 4);
            noc_async_read_barrier();
            id_end = off_ptr[0];
        }
#endif
        const uint32_t L = id_end - id_start;
#ifdef MB_SFPU_CULL
        // Per-tile PAGE-ALIGNED base of this tile's masks in cull_masks (the cull
        // writer uses the same base). cull_masks is NOT indexed by the dense sort
        // range (those id_start values are not 16-aligned, and unaligned NoC->DRAM
        // writes get shifted), so the mask for tile-local candidate p is at
        // cull_masks[cull_base + p].
        const uint32_t cull_base = read_soa_u32(cull_base_acc, tile_id, get_write_ptr(CB_SCR_IDS));
#endif
#ifdef MB_TILE_BUCKET
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
        if (Lb > 0 && Lb <= MB_BUCKET_FIT) {
            const uint32_t L = Lb;
            const uint32_t buck = get_write_ptr(CB_BUCKET);
            {
                uint32_t i = 0;
                while (i < L) {
                    const uint32_t end = (i + 64u < L) ? i + 64u : L;
                    for (uint32_t q = i; q < end; ++q) {
                        noc_async_read_tile(rec_start + q, tile_recs_acc, buck + q * SOA_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    i = end;
                }
            }
#if defined(MB_BUCKET_PREFETCH_MASK) && !defined(MB_BUCKET_DBG_INLINE) && !defined(MB_BUCKET_MASK)
            // T3: ISSUE this tile's whole cull_masks bulk read NOW (before the L1
            // depth sort), so the sort's real compute hides the NoC read latency.
            // The matching barrier lands just before the emit loop — NO settle spin
            // (the elapsed sort work is the settle window the spin used to provide).
            const uint32_t bmask = get_write_ptr(CB_BMASK);
            auto bmptr = reinterpret_cast<volatile uint32_t*>(bmask);
            {
                const uint32_t mpg0 = cull_base >> 4;
                const uint32_t mpages = (L + 15u) >> 4;
                for (uint32_t q = 0; q < mpages; ++q) {
                    noc_async_read_tile(mpg0 + q, cull_masks_acc, bmask + q * IDS_PAGE_BYTES);
                }
            }
#endif
            const uint32_t bs = get_write_ptr(CB_BSORT);
            uint32_t* idxA = reinterpret_cast<uint32_t*>(bs);
            uint32_t* idxB = idxA + MB_BUCKET_FIT;
            uint32_t* cnt  = idxB + MB_BUCKET_FIT;
            auto key_of = [&](uint32_t idx) -> uint32_t {
                return reinterpret_cast<volatile uint32_t*>(buck + idx * SOA_PAGE_BYTES)[9];
            };
            uint32_t* sorted;
#ifdef MB_BUCKET_DBG_NOSORT
            for (uint32_t i = 0; i < L; ++i) idxA[i] = i;
            sorted = idxA;
#else
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
#endif
            cb_reserve_back(CB_MB_COUNTS, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS))[0] = L;
            cb_push_back(CB_MB_COUNTS, 1);
#if defined(MB_BUCKET_MASK)
            // ROUTE C: the per-microblock keep mask was BAKED into record word 10
            // by the sort-stage cull (writer_bucket_cull.cpp). It is read back from
            // the L1-resident record below — NO cull_masks DRAM round-trip, NO
            // settle/read-completion spin (sort-stage writes read back spin-free).
#elif defined(MB_BUCKET_PREFETCH_MASK) && !defined(MB_BUCKET_DBG_INLINE)
            // Masks were ISSUED before the sort; the sort compute hid the latency.
            // One barrier lands them now — settle spin (if any) is reduced vs the
            // non-prefetch path since the sort already provided a settle window.
            noc_async_read_barrier();
#if defined(MB_CULL_SPIN)
            for (volatile int _s = 0; _s < (MB_CULL_SPIN); ++_s) { }
#endif
#elif !defined(MB_BUCKET_DBG_INLINE)
            // Bulk-load this tile's WHOLE cull_masks region into L1 ONCE (cull_base
            // is 16-aligned -> mask[k] == L1[k]) with batched barriers + a single
            // per-tile settle, instead of a per-candidate NoC read + spin. The
            // records are already L1-resident, so the candidate loop is pure L1.
            const uint32_t bmask = get_write_ptr(CB_BMASK);
            auto bmptr = reinterpret_cast<volatile uint32_t*>(bmask);
            {
                const uint32_t mpg0 = cull_base >> 4;
                const uint32_t mpages = (L + 15u) >> 4;
                uint32_t pp = 0;
                while (pp < mpages) {
                    const uint32_t end = (pp + 64u < mpages) ? pp + 64u : mpages;
                    for (uint32_t q = pp; q < end; ++q) {
                        noc_async_read_tile(mpg0 + q, cull_masks_acc, bmask + q * IDS_PAGE_BYTES);
                    }
                    noc_async_read_barrier();
                    pp = end;
                }
#if defined(MB_CULL_SPIN)
                for (volatile int _s = 0; _s < (MB_CULL_SPIN); ++_s) { }
#endif
            }
#endif
            for (uint32_t k = 0; k < L; ++k) {
                const uint32_t idx = sorted[k];
                auto recp = reinterpret_cast<volatile uint32_t*>(buck + idx * SOA_PAGE_BYTES);
#if defined(MB_BUCKET_EMIT_SPIN)
                // Diagnostic: per-record busy-wait in the bucket emit loop. If this
                // recovers the gate with mask=recp[10], the Lb>64 bug is a timing/
                // settle race the fast L1 emit exposes (the slow debug/inline paths
                // mask it incidentally).
                for (volatile int _es = 0; _es < (MB_BUCKET_EMIT_SPIN); ++_es) { }
#endif
                const uint32_t cov_a_bits = recp[0];
                const uint32_t cov_b_bits = recp[1];
                const uint32_t cov_c_bits = recp[2];
                const float mean_x = bits_to_f(recp[3]);
                const float mean_y = bits_to_f(recp[4]);
                const uint32_t op_bits = recp[5];
                const uint32_t cr = recp[6];
                const uint32_t cg = recp[7];
                const uint32_t cb = recp[8];
#if defined(MB_BUCKET_MASK)
                // ROUTE C: keep mask baked into record word 10 by the sort-stage
                // SFPU cull. Pure L1 load — spin-free, no random gather, no
                // cull_masks DRAM buffer on this path.
#if defined(MB_BUCKET_FORCE_INLINE)
                // Diagnostic: bucket-cull RMW still runs (BUCKET_MASK), but the
                // blend recomputes the mask inline from the L1 record instead of
                // reading recp[10]. Isolates "recp[10] read is wrong" from "RMW
                // corrupts records".
                const uint32_t mask = compute_microblock_mask(
                    bits_to_f(cov_a_bits), bits_to_f(cov_b_bits), bits_to_f(cov_c_bits),
                    mean_x, mean_y, bits_to_f(op_bits), tx_tile, ty_tile,
                    contrib_floor, cull_disabled);
#else
                const uint32_t mask = recp[10];
#endif
#if defined(MB_BUCKET_MASK_DEBUG)
                // Compare the BAKED mask to the inline soft-float reference for
                // THIS record. Accumulate per-tile counters and emit ONE summary
                // line per dense tile (L>64): how many records have a wrong baked
                // mask, and whether the divergence is a big mismatch vs SFPU noise.
                // Also track the largest-idx mismatch to see if corruption starts
                // beyond a record-index boundary.
                if (L > 64u) {
                    const uint32_t ref_mask = compute_microblock_mask(
                        bits_to_f(cov_a_bits), bits_to_f(cov_b_bits), bits_to_f(cov_c_bits),
                        mean_x, mean_y, bits_to_f(op_bits), tx_tile, ty_tile,
                        contrib_floor, cull_disabled);
                    static uint32_t _bm_mism = 0;   // records where baked != ref
                    static uint32_t _bm_big = 0;    // mismatch in >1 microblock bit
                    static uint32_t _bm_min_idx = 0xffffffffu;
                    static uint32_t _bm_max_idx = 0;
                    if (ref_mask != mask) {
                        _bm_mism++;
                        uint32_t x = ref_mask ^ mask;
                        uint32_t pc = 0; while (x) { pc += (x & 1u); x >>= 1u; }
                        if (pc > 1u) _bm_big++;
                        if (idx < _bm_min_idx) _bm_min_idx = idx;
                        if (idx > _bm_max_idx) _bm_max_idx = idx;
                    }
                    if (k + 1u == L) {  // last record of the tile -> emit summary
                        static uint32_t _bm_tiles = 0;
                        if (_bm_tiles < 40u) {
                            _bm_tiles++;
                            DPRINT << "BSUM t=" << tile_id << " L=" << L
                                   << " mism=" << _bm_mism << " big=" << _bm_big
                                   << " minidx=" << _bm_min_idx
                                   << " maxidx=" << _bm_max_idx << ENDL();
                        }
                        _bm_mism = 0; _bm_big = 0;
                        _bm_min_idx = 0xffffffffu; _bm_max_idx = 0;
                    }
                }
#endif
#elif defined(MB_BUCKET_DBG_INLINE)
                // Correctness probe: compute the mask inline from the L1 record
                // (correct for THIS record regardless of sort order).
                const uint32_t mask = compute_microblock_mask(
                    bits_to_f(cov_a_bits), bits_to_f(cov_b_bits), bits_to_f(cov_c_bits),
                    mean_x, mean_y, bits_to_f(op_bits), tx_tile, ty_tile,
                    contrib_floor, cull_disabled);
#else
                const uint32_t mask = bmptr[k];  // 16-aligned cull_base -> mask[k]==L1[k]
#endif
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
#if defined(MB_BUCKET_AB_PROBE)
                // A/B PROBE (producer side): stamp a per-tile sequence index in
                // row[11] and an XOR checksum of payload words 0..10 in row[12].
                // The compute verifies both. A wrong seq => the consumer read a
                // STALE whole row (the slot's previous occupant); a wrong checksum
                // => a TORN row (some words fresh, some stale). Either => the bug
                // is a producer->consumer L1 visibility race (HYPOTHESIS A). If
                // every row verifies but PSNR stays ~42, the rows are delivered
                // perfectly and the fault is the SFPU pipeline at full feed
                // (HYPOTHESIS B).
                row[11] = k;
                row[12] = cov_a_bits ^ cov_b_bits ^ cov_c_bits ^ mxl_bits ^ myl_bits ^
                          0u ^ op_bits ^ cr ^ cg ^ cb ^ mask;
#endif
#if defined(MB_BUCKET_CB_FENCE)
                // Order all row payload stores before cb_push_back's stream-reg
                // increment (write-through L1) so a consumer that observes the push
                // also observes the full row. The actual fast-producer race (UNPACK
                // recycling the slot before the slow MATH read) is fixed by the
                // MATH->UNPACK back-pressure ack in the compute kernel.
                mb_cb_commit_fence();
#endif
                cb_push_back(CB_MB_COEFF, 1);
            }
            continue;
        }
#endif
#ifdef MB_TILE_MASK_L1
        // Bulk-load this tile's ENTIRE mask region [cull_base, cull_base+L) into
        // L1 ONCE (cull_base is 16-aligned, so whole-page reads; ONE barrier).
        // The candidate loop then reads masks from L1 (pure load) — there is NO
        // per-candidate cull_masks NoC read, hence no read-completion window and
        // no need for the per-candidate MB_CULL_SPIN (the ~89 ms blend cost). The
        // diagnostic showed spin=0 with the per-chunk mask read drops to 37.7 dB;
        // bulk-loading the whole region with a single barrier lands all masks
        // before any are consumed.
        const uint32_t tile_mask_l1 = get_write_ptr(CB_TILE_MASKS);
        auto tile_mask_ptr = reinterpret_cast<volatile uint32_t*>(tile_mask_l1);
        // Do NOT assume cull_base is 16-aligned: the per-chunk path indexes with
        // off=(cull_base+p)&0xF and pg=(cull_base+p)>>4, so it is robust to any
        // alignment. The bulk path must match: page-0 is (cull_base>>4) and the
        // mask for tile-local candidate p lives at L1 element (mask_off + p),
        // where mask_off = cull_base & 0xF. Reading only ceil(L/16) pages (the
        // old assumption) drops the last partial page when mask_off>0 AND shifts
        // EVERY mask by mask_off -> garbage masks (the ~38 dB plateau).
        const uint32_t mask_off0 = cull_base & 0xFu;
        if (L > 0) {
            const uint32_t mpg0 = cull_base >> 4;                 // page of cull_base
            const uint32_t mpages = (mask_off0 + L + 15u) >> 4;   // pages covering [off,off+L)
#if defined(MB_TILE_MASK_SETTLE)
            // The cull pass writes cull_masks to DRAM EARLIER in the same Finish-
            // less dispatch; freshly written DRAM pages need a settle window to
            // DRAIN before a NoC read returns correct bits (a barrier alone does
            // not suffice — the per-chunk path spun for the SAME reason). The
            // settle must precede the read ISSUE: a post-read spin cannot fix bits
            // already latched stale into L1. ONE settle per tile, not per
            // candidate (~30x cheaper than the old per-candidate spin).
            for (volatile int _s = 0; _s < (MB_TILE_MASK_SETTLE); ++_s) { }
#endif
            // Drain in batches so the NoC outstanding-read queue never overflows
            // (issuing thousands of reads before one barrier can silently drop
            // transactions on a too-deep queue). MB_TILE_MASK_BATCH pages/barrier.
#ifndef MB_TILE_MASK_BATCH
#define MB_TILE_MASK_BATCH 64u
#endif
            uint32_t pp = 0;
            while (pp < mpages) {
                const uint32_t end = (pp + (MB_TILE_MASK_BATCH) < mpages)
                                         ? pp + (MB_TILE_MASK_BATCH) : mpages;
                for (uint32_t q = pp; q < end; ++q) {
                    noc_async_read_tile(mpg0 + q, cull_masks_acc,
                                        tile_mask_l1 + q * IDS_PAGE_BYTES);
                }
                noc_async_read_barrier();
                pp = end;
            }
        }
#endif

        // (3) Per-tile gaussian-row count (compute reads slot 0). One row is
        // emitted per candidate (mask==0 candidates dispatch nothing).
        cb_reserve_back(CB_MB_COUNTS, 1);
        {
            auto cnt_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
            cnt_ptr[0] = L;
        }
        cb_push_back(CB_MB_COUNTS, 1);

        // (4) Stream the candidate ids: gather attr, cull, emit coeff row.
#ifdef MB_RESIDENT
        // RESIDENT pipelined gather. The old path issued 9 blocking
        // read+barrier pairs per gaussian (~9 full NoC latencies each) — the
        // dominant blend cost. Here we batch one ids-page worth of gaussians
        // (<=16) per chunk: issue all 9*take SoA page reads ahead of a SINGLE
        // barrier, and double-buffer so the cull/emit of chunk K overlaps the
        // in-flight reads of chunk K+1 (separate L1 buffer by parity). Same
        // reads / same bytes / byte-identical emitted rows as before.
        if (L > 0) {
            const uint32_t attr_base = get_write_ptr(CB_SCR_ATTR);
            constexpr uint32_t CHUNK_MAX = IDS_PAGE_BYTES / 4;          // 16
            constexpr uint32_t BUF_BYTES = CHUNK_MAX * GATHER_SLOT_BYTES;
            const uint32_t ids_scr = get_write_ptr(CB_SCR_IDS);
            uint32_t gids[2][CHUNK_MAX];
            uint32_t take_buf[2];
#ifdef MB_SFPU_CULL
            // Per-buffer cull_masks scratch (2 buffers x 2 pages x 64B). The
            // chunk's 2-page mask window is read with a dedicated barrier in the
            // consume loop below (NOT folded into the shared attr prefetch
            // barrier, which did not reliably land the small mask reads under
            // fast timing). gstart_buf records each chunk's within-tile start so
            // the consume can recompute cull_base + gstart for the mask page.
            // Under MB_TILE_MASK_L1 the whole tile's masks are already resident in
            // CB_TILE_MASKS (bulk-loaded above), so the per-chunk scratch/read is
            // skipped; gstart_buf is still used to index the resident L1 masks.
            uint32_t gstart_buf[2];
#if !defined(MB_TILE_MASK_L1)
            constexpr uint32_t MASK_BUF_BYTES = 2u * IDS_PAGE_BYTES;  // 128B
            const uint32_t mask_scr = get_write_ptr(CB_SCR_MASK);
#endif
#endif

            uint32_t processed = 0;
            // Prologue: load + issue chunk 0 into buffer 0.
            take_buf[0] = load_ids_chunk(ids_acc, id_start, processed, L, ids_scr, gids[0]);
#ifdef MB_SFPU_CULL
            gstart_buf[0] = processed;
#endif
#ifdef MB_BLEND_AOS
            issue_chunk_reads_aos(gids[0], take_buf[0], attr_base + 0u * BUF_BYTES, blendrec_acc);
#else
            issue_chunk_reads(gids[0], take_buf[0], attr_base + 0u * BUF_BYTES,
                              a_acc, b_acc, c_acc, px_acc, py_acc, op_acc, col_acc);
#endif
            processed += take_buf[0];

            uint32_t cur = 0;
            while (take_buf[cur] > 0) {
                noc_async_read_barrier();   // chunk `cur` attrs have landed
                const uint32_t take = take_buf[cur];
                const uint32_t nxt = cur ^ 1u;

#if defined(MB_SFPU_CULL) && !defined(MB_TILE_MASK_L1)
                // Dedicated per-chunk mask-page read with its OWN barrier, issued
                // here (after the attr barrier, BEFORE the next chunk's prefetch)
                // so the barrier waits ONLY for these 2 small page reads. Folding
                // the mask reads into the shared prefetch barrier (which also
                // covers 144 attr page reads) did NOT reliably land them under
                // fast timing -> the prefetched mask read returned stale/partial
                // bits and dropped microblocks (PSNR 30 < keep-all 43.76). This
                // private barrier is correct-by-construction and costs one barrier
                // per ~16 candidates without disturbing the attr prefetch overlap.
                const uint32_t mask_off = load_mask_page(
                    cull_masks_acc, cull_base + gstart_buf[cur], take, mask_scr + cur * MASK_BUF_BYTES);
                noc_async_read_barrier();
                auto mask_ptr = reinterpret_cast<volatile uint32_t*>(mask_scr + cur * MASK_BUF_BYTES);
#endif

                // Prefetch chunk `nxt` into the other buffer (reads stay in
                // flight while we cull/emit chunk `cur` below).
                if (processed < L) {
                    take_buf[nxt] = load_ids_chunk(ids_acc, id_start, processed, L, ids_scr, gids[nxt]);
#ifdef MB_SFPU_CULL
                    gstart_buf[nxt] = processed;
#endif
#ifdef MB_BLEND_AOS
                    issue_chunk_reads_aos(gids[nxt], take_buf[nxt], attr_base + nxt * BUF_BYTES, blendrec_acc);
#else
                    issue_chunk_reads(gids[nxt], take_buf[nxt], attr_base + nxt * BUF_BYTES,
                                      a_acc, b_acc, c_acc, px_acc, py_acc, op_acc, col_acc);
#endif
                    processed += take_buf[nxt];
                } else {
                    take_buf[nxt] = 0;
                }

                const uint32_t buf = attr_base + cur * BUF_BYTES;
                for (uint32_t j = 0; j < take; ++j) {
                    const uint32_t g = gids[cur][j];
                    const uint32_t s = buf + j * GATHER_SLOT_BYTES;
#ifdef MB_BLEND_AOS
                    // S1: one contiguous AoS record page. Fields are dense at
                    // offsets [0..8] of the 64B record (see proj_m_blendrec emit
                    // in gather_visible_scatter.cpp): {a,b,c,px,py,op,cr,cg,cb}.
                    auto recp = reinterpret_cast<volatile uint32_t*>(s);
                    const uint32_t cov_a_bits = recp[0];
                    const uint32_t cov_b_bits = recp[1];
                    const uint32_t cov_c_bits = recp[2];
                    const uint32_t mx_bits    = recp[3];
                    const uint32_t my_bits    = recp[4];
                    const uint32_t op_bits    = recp[5];
                    const uint32_t cr = recp[6];
                    const uint32_t cg = recp[7];
                    const uint32_t cb = recp[8];
                    (void)g;  // only referenced by the SoA lane math / debug path
#else
                    const uint32_t lane = g & 0xF;
                    const uint32_t cov_a_bits = reinterpret_cast<volatile uint32_t*>(s + 0u * 64u)[lane];
                    const uint32_t cov_b_bits = reinterpret_cast<volatile uint32_t*>(s + 1u * 64u)[lane];
                    const uint32_t cov_c_bits = reinterpret_cast<volatile uint32_t*>(s + 2u * 64u)[lane];
                    const uint32_t mx_bits    = reinterpret_cast<volatile uint32_t*>(s + 3u * 64u)[lane];
                    const uint32_t my_bits    = reinterpret_cast<volatile uint32_t*>(s + 4u * 64u)[lane];
                    const uint32_t op_bits    = reinterpret_cast<volatile uint32_t*>(s + 5u * 64u)[lane];
                    // Colours live in a contiguous 2-page window at slot 6 (see
                    // issue_chunk_reads): index by (elem - page0_base), page0_base
                    // = (e0>>4)*16. e0+2-base <= 17 < 32 so always in-window.
                    const uint32_t e0 = g * 3u;
                    const uint32_t cbase = (e0 >> 4) * 16u;
                    auto cwin = reinterpret_cast<volatile uint32_t*>(s + 6u * 64u);
                    const uint32_t cr = cwin[(e0 + 0u) - cbase];
                    const uint32_t cg = cwin[(e0 + 1u) - cbase];
                    const uint32_t cb = cwin[(e0 + 2u) - cbase];
#endif

                    const float mean_x = bits_to_f(mx_bits);
                    const float mean_y = bits_to_f(my_bits);

#if defined(MB_TILE_MASK_L1)
                    // Whole-tile masks are resident in L1 (bulk-loaded once per
                    // tile). Index by within-tile depth position PLUS the page
                    // offset of cull_base (mask_off0), matching the per-chunk path
                    // (cull_masks[cull_base + p]); pure L1 load, no NoC read.
                    const uint32_t mask = tile_mask_ptr[mask_off0 + gstart_buf[cur] + j];
#elif defined(MB_SFPU_CULL)
                    const uint32_t mask = mask_ptr[mask_off + j];
#if defined(MB_CULL_SPIN) && !defined(MB_BUCKET_NO_FALLBACK_SPIN)
                    // DRAM pages freshly written by the cull pass need a brief
                    // settling window before NoC reads are consumed; barriers
                    // alone do not suffice. Per-candidate volatile spin (count
                    // from GSPLAT_TT_CULL_SPIN, default in blend_device.cpp).
                    for (volatile int _s = 0; _s < (MB_CULL_SPIN); ++_s) { }
#endif
#if defined(MB_SFPU_CULL_DEBUG)
                    {
                        const uint32_t ref_mask = compute_microblock_mask(
                            bits_to_f(cov_a_bits), bits_to_f(cov_b_bits),
                            bits_to_f(cov_c_bits), mean_x, mean_y,
                            bits_to_f(op_bits), tx_tile, ty_tile,
                            contrib_floor, cull_disabled);
                        static uint32_t dbg_n = 0;
                        static uint32_t dbg_mm = 0;
                        const uint32_t k_dbg = cull_base + gstart_buf[cur] + j;
                        dbg_n++;
                        if (ref_mask != mask && dbg_mm < 60u) {
                            dbg_mm++;
                            DPRINT << "CULLMM t=" << tile_id << " g=" << g << " k=" << k_dbg
                                   << " base=" << cull_base << " gs=" << gstart_buf[cur]
                                   << " j=" << j << " L=" << L
                                   << " ref=" << ref_mask << " sfpu=" << mask << ENDL();
                            DPRINT << "CULLMC k=" << k_dbg
                                   << " a=" << F32(bits_to_f(cov_a_bits))
                                   << " b=" << F32(bits_to_f(cov_b_bits))
                                   << " c=" << F32(bits_to_f(cov_c_bits))
                                   << " mx=" << F32(mean_x) << " my=" << F32(mean_y)
                                   << " op=" << F32(bits_to_f(op_bits)) << ENDL();
                        }
                    }
#endif
#else
                    const float cov_a  = bits_to_f(cov_a_bits);
                    const float cov_b  = bits_to_f(cov_b_bits);
                    const float cov_c  = bits_to_f(cov_c_bits);
                    const float opac   = bits_to_f(op_bits);
                    const uint32_t mask = compute_microblock_mask(
                        cov_a, cov_b, cov_c, mean_x, mean_y, opac, tx_tile, ty_tile,
                        contrib_floor, cull_disabled);
#endif

                    cb_reserve_back(CB_MB_COEFF, 1);
                    auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COEFF));
                    row[0] = cov_a_bits;                         // raw cov_a
                    row[1] = cov_b_bits;                         // raw cov_b
                    row[2] = cov_c_bits;                         // raw cov_c
                    row[3] = f_to_bits(mean_x - tx_tile);        // mx_local
                    row[4] = f_to_bits(mean_y - ty_tile);        // my_local
                    row[5] = 0u;
                    row[6] = op_bits;                            // opacity
                    row[7] = cr;
                    row[8] = cg;
                    row[9] = cb;
                    row[10] = mask;
                    row[11] = 0u;
                    row[12] = 0u;
                    row[13] = 0u;
                    row[14] = 0u;
                    row[15] = 0u;
#ifdef FUSE_AB_ROW
                    if ((tile_id % 1500u) == 7u) {
                        static uint32_t _ab_n = 0;
                        const uint32_t _ab_p = gstart_buf[cur] + j;
                        if (_ab_n < 320u) { _ab_n++;
                            DPRINT << "ABROW t=" << tile_id << " p=" << _ab_p << " g=" << g
                                   << " a=" << F32(bits_to_f(row[0]))
                                   << " mxl=" << F32(bits_to_f(row[3]))
                                   << " op=" << F32(bits_to_f(row[6]))
                                   << " m=" << row[10] << ENDL();
                        }
                    }
#endif
#if defined(MB_BUCKET_CB_FENCE)
                    // Order payload stores before push (same rationale as the
                    // bucket emit above); back-pressure is enforced consumer-side.
                    mb_cb_commit_fence();
#endif
                    cb_push_back(CB_MB_COEFF, 1);
                }
                cur = nxt;
            }
        }
#else
        const uint32_t ids_per_page = IDS_PAGE_BYTES / 4;  // 16
        uint32_t processed = 0;
        while (processed < L) {
            const uint32_t global_idx = id_start + processed;
            const uint32_t page_idx = global_idx / ids_per_page;
            const uint32_t in_page  = global_idx % ids_per_page;
            const uint32_t ids_scratch = get_write_ptr(CB_SCR_IDS);
            auto ids_ptr = reinterpret_cast<volatile uint32_t*>(ids_scratch);
            noc_async_read(get_noc_addr(page_idx, ids_acc), ids_scratch, IDS_PAGE_BYTES);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > L - processed) take = L - processed;

            for (uint32_t j = 0; j < take; j++) {
                const uint32_t g = ids_ptr[in_page + j];

                // Gather attr[g] (64B) into the private attr scratch.
                const uint32_t attr_scratch = get_write_ptr(CB_SCR_ATTR);
                auto attr_ptr = reinterpret_cast<volatile uint32_t*>(attr_scratch);
                noc_async_read_tile(g, attrs_acc, attr_scratch);
                noc_async_read_barrier();
                const uint32_t cov_a_bits = attr_ptr[0];
                const uint32_t cov_b_bits = attr_ptr[1];
                const uint32_t cov_c_bits = attr_ptr[2];
                const uint32_t mx_bits    = attr_ptr[3];
                const uint32_t my_bits    = attr_ptr[4];
                const uint32_t op_bits    = attr_ptr[5];
                const uint32_t cr         = attr_ptr[6];
                const uint32_t cg         = attr_ptr[7];
                const uint32_t cb         = attr_ptr[8];

                const float cov_a  = bits_to_f(cov_a_bits);
                const float cov_b  = bits_to_f(cov_b_bits);
                const float cov_c  = bits_to_f(cov_c_bits);
                const float mean_x = bits_to_f(mx_bits);
                const float mean_y = bits_to_f(my_bits);
                const float opac   = bits_to_f(op_bits);

                const uint32_t mask = compute_microblock_mask(
                    cov_a, cov_b, cov_c, mean_x, mean_y, opac, tx_tile, ty_tile,
                    contrib_floor, cull_disabled);

                cb_reserve_back(CB_MB_COEFF, 1);
                auto row = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COEFF));
                row[0] = cov_a_bits;                         // raw cov_a
                row[1] = cov_b_bits;                         // raw cov_b
                row[2] = cov_c_bits;                         // raw cov_c
                row[3] = f_to_bits(mean_x - tx_tile);        // mx_local
                row[4] = f_to_bits(mean_y - ty_tile);        // my_local
                row[5] = 0u;
                row[6] = op_bits;                            // opacity
                row[7] = cr;
                row[8] = cg;
                row[9] = cb;
                row[10] = mask;
                row[11] = 0u;
                row[12] = 0u;
                row[13] = 0u;
                row[14] = 0u;
                row[15] = 0u;
                cb_push_back(CB_MB_COEFF, 1);
            }
            processed += take;
        }
#endif
    }
}
