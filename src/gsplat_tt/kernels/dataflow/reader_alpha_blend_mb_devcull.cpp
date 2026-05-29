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
#endif

// Reproduce build_gaussian_major_tile's per-gaussian microblock cull EXACTLY.
// Returns the 32-bit microblock-coverage mask. tx_tile/ty_tile are the float
// tile-origin pixel coords (tx*32, ty*32).
inline uint32_t compute_microblock_mask(
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
    uint32_t mask = 0u;
    for (int my = my_lo; my <= my_hi; ++my) {
        const float mb_oy = ty_tile + static_cast<float>(my * 4);
        const float v_lo = mb_oy - mean_y;
        const float v_hi = v_lo + 4.0f;
        const bool y_inside = (v_lo <= 0.0f) && (0.0f <= v_hi);
        const float v_fix = (v_lo > 0.0f) ? v_lo : v_hi;
        for (int mx = mx_lo; mx <= mx_hi; ++mx) {
            const float mb_ox = tx_tile + static_cast<float>(mx * 8);
            const float u_lo = mb_ox - mean_x;
            const float u_hi = u_lo + 8.0f;
            const bool x_inside = (u_lo <= 0.0f) && (0.0f <= u_hi);

            float m2_min;
            if (cull_disabled || (x_inside && y_inside)) {
                m2_min = 0.0f;
            } else {
                float m2_v = kInf;
                if (!x_inside) {
                    const float u_fix = (u_lo > 0.0f) ? u_lo : u_hi;
                    float v_star = -ci_b * u_fix / ci_c_safe;
                    if (v_star < v_lo) v_star = v_lo;
                    if (v_star > v_hi) v_star = v_hi;
                    m2_v = ci_a * u_fix * u_fix + 2.0f * ci_b * u_fix * v_star +
                           ci_c * v_star * v_star;
                }
                float m2_h = kInf;
                if (!y_inside) {
                    float u_star = -ci_b * v_fix / ci_a_safe;
                    if (u_star < u_lo) u_star = u_lo;
                    if (u_star > u_hi) u_star = u_hi;
                    m2_h = ci_a * u_star * u_star + 2.0f * ci_b * u_star * v_fix +
                           ci_c * v_fix * v_fix;
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
    const uint32_t tile_ids_start = get_arg_val<uint32_t>(12);
    const uint32_t tile_ids_count = get_arg_val<uint32_t>(13);
    const uint32_t tiles_x        = get_arg_val<uint32_t>(14);
    const float contrib_floor     = bits_to_f(get_arg_val<uint32_t>(15));
    const bool cull_disabled      = get_arg_val<uint32_t>(16) != 0;

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

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        const uint32_t tx = tile_id % tiles_x;
        const uint32_t ty = tile_id / tiles_x;
        const float tx_tile = static_cast<float>(tx * TILE_SIZE);
        const float ty_tile = static_cast<float>(ty * TILE_SIZE);

        // (1) Shared permuted coordinate ramps (page 0 of each ramp buffer).
        cb_reserve_back(CB_XRAMP, 1);
        noc_async_read_tile(0, xramp_acc, get_write_ptr(CB_XRAMP));
        cb_reserve_back(CB_YRAMP, 1);
        noc_async_read_tile(0, yramp_acc, get_write_ptr(CB_YRAMP));
        noc_async_read_barrier();
        cb_push_back(CB_XRAMP, 1);
        cb_push_back(CB_YRAMP, 1);

        // (2) Per-tile candidate id range [id_start, id_end).
        uint32_t id_start, id_end;
#ifdef MB_RESIDENT
        // Resident sort_tile_ranges: (start,end) uint32 pair per tile at
        // elements [tile_id*2, tile_id*2+1]. Equivalent to the host ids_off
        // prefix because sort_sorted_ids IS the per-tile depth-sorted concat
        // (start/end into it == ids_off[t]/ids_off[t+1]); empty tiles read
        // (0,0) -> L==0, matching the uploaded path.
        {
            const uint32_t scr = get_write_ptr(CB_SCR_IDS);
            id_start = read_soa_u32(ranges_acc, tile_id * 2u + 0u, scr);
            id_end   = read_soa_u32(ranges_acc, tile_id * 2u + 1u, scr);
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

        // (3) Per-tile gaussian-row count (compute reads slot 0). One row is
        // emitted per candidate (mask==0 candidates dispatch nothing).
        cb_reserve_back(CB_MB_COUNTS, 1);
        {
            auto cnt_ptr = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
            cnt_ptr[0] = L;
        }
        cb_push_back(CB_MB_COUNTS, 1);

        // (4) Stream the candidate ids: gather attr, cull, emit coeff row.
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

                // Gather this gaussian's attributes (raw cov a,b,c, image-space
                // center, opacity, color rgb) as raw 32-bit words.
                uint32_t cov_a_bits, cov_b_bits, cov_c_bits;
                uint32_t mx_bits, my_bits, op_bits, cr, cg, cb;
#ifdef MB_RESIDENT
                // Gather straight from the resident per-component SoA buffers by
                // id g. Byte-identical to the values the host attr table carried
                // (proj_finish output == gather_visible output). colors are AoS
                // M*3, so element index is g*3 + channel.
                const uint32_t scr = get_write_ptr(CB_SCR_ATTR);
                cov_a_bits = read_soa_u32(a_acc,  g, scr);
                cov_b_bits = read_soa_u32(b_acc,  g, scr);
                cov_c_bits = read_soa_u32(c_acc,  g, scr);
                mx_bits    = read_soa_u32(px_acc, g, scr);
                my_bits    = read_soa_u32(py_acc, g, scr);
                op_bits    = read_soa_u32(op_acc, g, scr);
                cr         = read_soa_u32(col_acc, g * 3u + 0u, scr);
                cg         = read_soa_u32(col_acc, g * 3u + 1u, scr);
                cb         = read_soa_u32(col_acc, g * 3u + 2u, scr);
#else
                // Gather attr[g] (64B) into the private attr scratch.
                const uint32_t attr_scratch = get_write_ptr(CB_SCR_ATTR);
                auto attr_ptr = reinterpret_cast<volatile uint32_t*>(attr_scratch);
                noc_async_read_tile(g, attrs_acc, attr_scratch);
                noc_async_read_barrier();
                cov_a_bits = attr_ptr[0];
                cov_b_bits = attr_ptr[1];
                cov_c_bits = attr_ptr[2];
                mx_bits    = attr_ptr[3];
                my_bits    = attr_ptr[4];
                op_bits    = attr_ptr[5];
                cr         = attr_ptr[6];
                cg         = attr_ptr[7];
                cb         = attr_ptr[8];
#endif
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
    }
}
