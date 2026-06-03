// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gather_visible SCATTER kernel — residency pass R2b (MULTI-CORE).
//
// A single data-movement kernel replicated across the Blackhole core grid.
// The N-indexed device-resident project outputs (8 pfwc_* SoA fp32 tile streams
// + 4 scene streams: col_r, col_g, col_b, opacity) are split across cores by a
// contiguous TILE range [t_start, t_start + t_count). Each core applies the
// SAME valid_mask predicate as gsplat_cpu::project_finish_with_cov2d_radii and
// scatters the visible Gaussians it owns into the proj_m_* DRAM buffers.
//
// Stable global compaction is preserved by a two-pass scheme driven by the
// host:
//   * count_only pass: each core counts its visible quota over its tile range
//     and writes that scalar count into its own page of the per-core counts
//     DRAM buffer (counts[core_id]).
//   * scatter pass: the host computes the exclusive prefix-sum of the per-core
//     counts and passes each core its global output base offset `base`. Core c
//     writes its visible elements to global compact indices base, base+1, ...
//     in increasing source-index order. Because cores own disjoint, ordered
//     source ranges and disjoint, ordered output ranges, concatenating them in
//     core order reproduces exactly the single-core stable compaction.
//
// Boundary pages: a core's output range generally does not start/end on a 16-
// element (64B) page boundary, so the first/last output page of a core is
// shared with its neighbour. Each core writes ONLY the contiguous slot sub-
// range it owns within such a page. The two neighbouring cores write disjoint,
// page-congruent byte ranges (both L1 staging base and DRAM page base are 64B-
// aligned and offset by the same slot*4), so the partial NOC writes do not
// clobber one another and the union tiles the page exactly.
//
// INPUT PAGE LAYOUT: pfwc_* + scene_* are fp32 SoA, TILE-paged (1024 elems /
// 4096B page). OUTPUT PAGE LAYOUT: proj_m_* SoA are fp32 64B pages (16 elems);
// proj_m_colors is fp32 AoS (M*3) 64B pages (16 floats / page). proj_M is
// written by the host from the summed per-core counts.
//
// RUNTIME ARGS (all uint32):
//   0..7  : pfwc DRAM bases   m2x, m2y, depth, a, b, c, rx, ry
//   8..11 : scene DRAM bases  col_r, col_g, col_b, opacity
//   12..20: out  DRAM bases   px, py, rx, ry, a, b, c, depth, opacity (SoA)
//   21    : out colors DRAM base (AoS M*3)
//   22    : out proj_M DRAM base (unused by kernel; host writes proj_M)
//   23    : N
//   24    : num_tiles  (ceil(N/1024)) — informational
//   25    : min_opacity (fp32 bits)
//   26    : k_near      (fp32 bits, 0.2)
//   27    : image_width (fp32 bits)
//   28    : image_height(fp32 bits)
//   29    : max_radius  (fp32 bits, effective)
//   30    : count_only  (1 = only count this core's visible quota -> counts[core_id])
//   31    : t_start     (first tile this core owns)
//   32    : t_count     (number of tiles this core owns)
//   33    : base        (global compact output offset for this core, scatter pass)
//   34    : is_last     (1 = this core writes the final element; zero-pads tail)
//   35    : core_id     (slot in the per-core counts buffer)
//   36    : counts DRAM base (per-core counts, one 64B page per core)
//
// COMPILE-TIME ARGS: 24 TensorAccessorArgs in the order
//   m2x,m2y,depth,a,b,c,rx,ry, col_r,col_g,col_b,op,
//   o_px,o_py,o_rx,o_ry,o_a,o_b,o_c,o_depth,o_op, o_colors, o_M, o_counts.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4;  // 4096
constexpr uint32_t PAGE_ELEMS = 16;
constexpr uint32_t PAGE_BYTES = PAGE_ELEMS * 4;  // 64
constexpr uint32_t COLOR_GROUP_FLOATS = PAGE_ELEMS * 3;  // 48 floats / 16-Gaussian group

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline int clampi(int v, int lo, int hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

}  // namespace

void kernel_main() {
    const uint32_t count_only = get_arg_val<uint32_t>(30);
    // The Tracy device zone is opened inside each pass below (count vs scatter)
    // with a COMPILE-TIME-LITERAL name. DeviceZoneScopedN hashes its argument via
    // Hash16_CT(const char (&)[N]); a runtime ternary decays to const char* and
    // fails that template's N deduction (kernel_profiler.hpp:110) under
    // TT_METAL_DEVICE_PROFILER=1. Two static-named zones keep the per-pass labels.
    const uint32_t m2x_addr   = get_arg_val<uint32_t>(0);
    const uint32_t m2y_addr   = get_arg_val<uint32_t>(1);
    const uint32_t depth_addr = get_arg_val<uint32_t>(2);
    const uint32_t a_addr     = get_arg_val<uint32_t>(3);
    const uint32_t b_addr     = get_arg_val<uint32_t>(4);
    const uint32_t c_addr     = get_arg_val<uint32_t>(5);
    const uint32_t rx_addr    = get_arg_val<uint32_t>(6);
    const uint32_t ry_addr    = get_arg_val<uint32_t>(7);
    const uint32_t cr_addr    = get_arg_val<uint32_t>(8);
    const uint32_t cg_addr    = get_arg_val<uint32_t>(9);
    const uint32_t cb_addr    = get_arg_val<uint32_t>(10);
    const uint32_t op_addr    = get_arg_val<uint32_t>(11);
    const uint32_t o_px_addr     = get_arg_val<uint32_t>(12);
    const uint32_t o_py_addr     = get_arg_val<uint32_t>(13);
    const uint32_t o_rx_addr     = get_arg_val<uint32_t>(14);
    const uint32_t o_ry_addr     = get_arg_val<uint32_t>(15);
    const uint32_t o_a_addr      = get_arg_val<uint32_t>(16);
    const uint32_t o_b_addr      = get_arg_val<uint32_t>(17);
    const uint32_t o_c_addr      = get_arg_val<uint32_t>(18);
    const uint32_t o_depth_addr  = get_arg_val<uint32_t>(19);
    const uint32_t o_op_addr     = get_arg_val<uint32_t>(20);
    const uint32_t o_colors_addr = get_arg_val<uint32_t>(21);
    const uint32_t o_M_addr      = get_arg_val<uint32_t>(22);
    const uint32_t N          = get_arg_val<uint32_t>(23);
    const uint32_t num_tiles  = get_arg_val<uint32_t>(24);
    const float min_opacity   = bits_to_f(get_arg_val<uint32_t>(25));
    const float k_near        = bits_to_f(get_arg_val<uint32_t>(26));
    const float img_w         = bits_to_f(get_arg_val<uint32_t>(27));
    const float img_h         = bits_to_f(get_arg_val<uint32_t>(28));
    const float max_radius    = bits_to_f(get_arg_val<uint32_t>(29));
    const uint32_t t_start    = get_arg_val<uint32_t>(31);
    const uint32_t t_count    = get_arg_val<uint32_t>(32);
    const uint32_t base       = get_arg_val<uint32_t>(33);
    const uint32_t is_last    = get_arg_val<uint32_t>(34);
    const uint32_t core_id    = get_arg_val<uint32_t>(35);
    const uint32_t counts_addr= get_arg_val<uint32_t>(36);
#ifdef GATHER_EMIT_BLENDREC
    // S1 (GSPLAT_TT_BLEND_AOS): contiguous AoS blend record buffer. One 64B page
    // per compacted gaussian == {a,b,c,px,py,op,cr,cg,cb, 0..}. Lets the blend
    // reader fetch a candidate with ONE contiguous read instead of 7-9 SoA pages.
    const uint32_t o_blendrec_addr = get_arg_val<uint32_t>(37);
    // GSPLAT_TT_PROJ_BALANCE: tile stride. 1 = contiguous chunk (this core owns
    // tiles [t_start, t_start+t_count)); num_cores = interleaved/strided (this
    // core owns t_start, t_start+stride, t_start+2*stride, ...).
    const uint32_t t_stride = get_arg_val<uint32_t>(38);
    // GSPLAT_TT_PROJ_DEVICE_SCAN: 1 => read (base, is_last) from counts_addr
    // (repointed by the host at the device-scan base buffer) page core_id,
    // instead of the host-computed `base`/`is_last` args (which are 0).
    const uint32_t device_scan = get_arg_val<uint32_t>(39);
#else
    const uint32_t t_stride = get_arg_val<uint32_t>(37);
    const uint32_t device_scan = get_arg_val<uint32_t>(38);
#endif
#ifdef GATHER_EMIT_TPG
#ifdef GATHER_EMIT_BLENDREC
    const int tiles_x_f = static_cast<int>(get_arg_val<uint32_t>(40));
    const int tiles_y_f = static_cast<int>(get_arg_val<uint32_t>(41));
    const float tsf_f = static_cast<float>(get_arg_val<uint32_t>(42));
    const uint32_t tpg_addr = get_arg_val<uint32_t>(43);
#else
    const int tiles_x_f = static_cast<int>(get_arg_val<uint32_t>(39));
    const int tiles_y_f = static_cast<int>(get_arg_val<uint32_t>(40));
    const float tsf_f = static_cast<float>(get_arg_val<uint32_t>(41));
    const uint32_t tpg_addr = get_arg_val<uint32_t>(42);
#endif
#endif
    (void)num_tiles;
    (void)o_M_addr;

    constexpr auto a_m2x   = TensorAccessorArgs<0>();
    constexpr auto a_m2y   = TensorAccessorArgs<a_m2x.next_compile_time_args_offset()>();
    constexpr auto a_depth = TensorAccessorArgs<a_m2y.next_compile_time_args_offset()>();
    constexpr auto a_a     = TensorAccessorArgs<a_depth.next_compile_time_args_offset()>();
    constexpr auto a_b     = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto a_c     = TensorAccessorArgs<a_b.next_compile_time_args_offset()>();
    constexpr auto a_rx    = TensorAccessorArgs<a_c.next_compile_time_args_offset()>();
    constexpr auto a_ry    = TensorAccessorArgs<a_rx.next_compile_time_args_offset()>();
    constexpr auto a_cr    = TensorAccessorArgs<a_ry.next_compile_time_args_offset()>();
    constexpr auto a_cg    = TensorAccessorArgs<a_cr.next_compile_time_args_offset()>();
    constexpr auto a_cb    = TensorAccessorArgs<a_cg.next_compile_time_args_offset()>();
    constexpr auto a_op    = TensorAccessorArgs<a_cb.next_compile_time_args_offset()>();
    constexpr auto a_opx   = TensorAccessorArgs<a_op.next_compile_time_args_offset()>();
    constexpr auto a_opy   = TensorAccessorArgs<a_opx.next_compile_time_args_offset()>();
    constexpr auto a_orx   = TensorAccessorArgs<a_opy.next_compile_time_args_offset()>();
    constexpr auto a_ory   = TensorAccessorArgs<a_orx.next_compile_time_args_offset()>();
    constexpr auto a_oa    = TensorAccessorArgs<a_ory.next_compile_time_args_offset()>();
    constexpr auto a_ob    = TensorAccessorArgs<a_oa.next_compile_time_args_offset()>();
    constexpr auto a_oc    = TensorAccessorArgs<a_ob.next_compile_time_args_offset()>();
    constexpr auto a_odep  = TensorAccessorArgs<a_oc.next_compile_time_args_offset()>();
    constexpr auto a_oop   = TensorAccessorArgs<a_odep.next_compile_time_args_offset()>();
    constexpr auto a_ocol  = TensorAccessorArgs<a_oop.next_compile_time_args_offset()>();
    constexpr auto a_oM    = TensorAccessorArgs<a_ocol.next_compile_time_args_offset()>();
    constexpr auto a_counts= TensorAccessorArgs<a_oM.next_compile_time_args_offset()>();
#ifdef GATHER_EMIT_BLENDREC
    constexpr auto a_brec  = TensorAccessorArgs<a_counts.next_compile_time_args_offset()>();
#endif
#ifdef GATHER_EMIT_TPG
    constexpr auto a_tpg = TensorAccessorArgs<
#ifdef GATHER_EMIT_BLENDREC
        a_brec.next_compile_time_args_offset()
#else
        a_counts.next_compile_time_args_offset()
#endif
        >();
#endif
    (void)a_oM;

    const auto acc_m2x   = TensorAccessor(a_m2x,   m2x_addr,   TILE_BYTES);
    const auto acc_m2y   = TensorAccessor(a_m2y,   m2y_addr,   TILE_BYTES);
    const auto acc_depth = TensorAccessor(a_depth, depth_addr, TILE_BYTES);
    const auto acc_a     = TensorAccessor(a_a,     a_addr,     TILE_BYTES);
    const auto acc_b     = TensorAccessor(a_b,     b_addr,     TILE_BYTES);
    const auto acc_c     = TensorAccessor(a_c,     c_addr,     TILE_BYTES);
    const auto acc_rx    = TensorAccessor(a_rx,    rx_addr,    TILE_BYTES);
    const auto acc_ry    = TensorAccessor(a_ry,    ry_addr,    TILE_BYTES);
    const auto acc_cr    = TensorAccessor(a_cr,    cr_addr,    TILE_BYTES);
    const auto acc_cg    = TensorAccessor(a_cg,    cg_addr,    TILE_BYTES);
    const auto acc_cb    = TensorAccessor(a_cb,    cb_addr,    TILE_BYTES);
    const auto acc_op    = TensorAccessor(a_op,    op_addr,    TILE_BYTES);
    const auto acc_opx   = TensorAccessor(a_opx,   o_px_addr,    PAGE_BYTES);
    const auto acc_opy   = TensorAccessor(a_opy,   o_py_addr,    PAGE_BYTES);
    const auto acc_orx   = TensorAccessor(a_orx,   o_rx_addr,    PAGE_BYTES);
    const auto acc_ory   = TensorAccessor(a_ory,   o_ry_addr,    PAGE_BYTES);
    const auto acc_oa    = TensorAccessor(a_oa,    o_a_addr,     PAGE_BYTES);
    const auto acc_ob    = TensorAccessor(a_ob,    o_b_addr,     PAGE_BYTES);
    const auto acc_oc    = TensorAccessor(a_oc,    o_c_addr,     PAGE_BYTES);
    const auto acc_odep  = TensorAccessor(a_odep,  o_depth_addr, PAGE_BYTES);
    const auto acc_oop   = TensorAccessor(a_oop,   o_op_addr,    PAGE_BYTES);
    const auto acc_ocol  = TensorAccessor(a_ocol,  o_colors_addr, PAGE_BYTES);
    const auto acc_counts= TensorAccessor(a_counts, counts_addr,  PAGE_BYTES);
#ifdef GATHER_EMIT_BLENDREC
    const auto acc_brec  = TensorAccessor(a_brec,  o_blendrec_addr, PAGE_BYTES);
#endif
#ifdef GATHER_EMIT_TPG
    const auto acc_tpg = TensorAccessor(a_tpg, tpg_addr, PAGE_BYTES);
#endif

    constexpr uint32_t CB_M2X = 0, CB_M2Y = 1, CB_DEP = 2, CB_A = 3, CB_B = 4,
                       CB_C = 5, CB_RX = 6, CB_RY = 7, CB_CR = 8, CB_CG = 9,
                       CB_CB = 10, CB_OP = 11;
    constexpr uint32_t CB_OPX = 12, CB_OPY = 13, CB_ORX = 14, CB_ORY = 15,
                       CB_OA = 16, CB_OB = 17, CB_OC = 18, CB_ODEP = 19,
                       CB_OOP = 20, CB_OCOL = 21, CB_OM = 22;
#ifdef GATHER_EMIT_BLENDREC
    constexpr uint32_t CB_OREC = 23;  // 16 records x 64B AoS staging
    constexpr uint32_t REC_WORDS = 16;  // 64B / 4
    const uint32_t l1_orec = get_write_ptr(CB_OREC);
    auto o_rec = reinterpret_cast<volatile uint32_t*>(l1_orec);
#endif
#ifdef GATHER_EMIT_TPG
    constexpr uint32_t CB_OTPG = 24;
    const uint32_t l1_otpg = get_write_ptr(CB_OTPG);
    auto o_tpg = reinterpret_cast<volatile int32_t*>(l1_otpg);
#endif

    const uint32_t l1_m2x = get_write_ptr(CB_M2X);
    const uint32_t l1_m2y = get_write_ptr(CB_M2Y);
    const uint32_t l1_dep = get_write_ptr(CB_DEP);
    const uint32_t l1_a   = get_write_ptr(CB_A);
    const uint32_t l1_b   = get_write_ptr(CB_B);
    const uint32_t l1_c   = get_write_ptr(CB_C);
    const uint32_t l1_rx  = get_write_ptr(CB_RX);
    const uint32_t l1_ry  = get_write_ptr(CB_RY);
    const uint32_t l1_cr  = get_write_ptr(CB_CR);
    const uint32_t l1_cg  = get_write_ptr(CB_CG);
    const uint32_t l1_cb  = get_write_ptr(CB_CB);
    const uint32_t l1_op  = get_write_ptr(CB_OP);

    auto p_m2x = reinterpret_cast<volatile uint32_t*>(l1_m2x);
    auto p_m2y = reinterpret_cast<volatile uint32_t*>(l1_m2y);
    auto p_dep = reinterpret_cast<volatile uint32_t*>(l1_dep);
    auto p_a   = reinterpret_cast<volatile uint32_t*>(l1_a);
    auto p_b   = reinterpret_cast<volatile uint32_t*>(l1_b);
    auto p_c   = reinterpret_cast<volatile uint32_t*>(l1_c);
    auto p_rx  = reinterpret_cast<volatile uint32_t*>(l1_rx);
    auto p_ry  = reinterpret_cast<volatile uint32_t*>(l1_ry);
    auto p_cr  = reinterpret_cast<volatile uint32_t*>(l1_cr);
    auto p_cg  = reinterpret_cast<volatile uint32_t*>(l1_cg);
    auto p_cb  = reinterpret_cast<volatile uint32_t*>(l1_cb);
    auto p_op  = reinterpret_cast<volatile uint32_t*>(l1_op);

    const uint32_t l1_opx = get_write_ptr(CB_OPX);
    const uint32_t l1_opy = get_write_ptr(CB_OPY);
    const uint32_t l1_orx = get_write_ptr(CB_ORX);
    const uint32_t l1_ory = get_write_ptr(CB_ORY);
    const uint32_t l1_oa  = get_write_ptr(CB_OA);
    const uint32_t l1_ob  = get_write_ptr(CB_OB);
    const uint32_t l1_oc  = get_write_ptr(CB_OC);
    const uint32_t l1_odep= get_write_ptr(CB_ODEP);
    const uint32_t l1_oop = get_write_ptr(CB_OOP);
    const uint32_t l1_ocol= get_write_ptr(CB_OCOL);
    const uint32_t l1_oM  = get_write_ptr(CB_OM);

    auto o_px  = reinterpret_cast<volatile uint32_t*>(l1_opx);
    auto o_py  = reinterpret_cast<volatile uint32_t*>(l1_opy);
    auto o_rx  = reinterpret_cast<volatile uint32_t*>(l1_orx);
    auto o_ry  = reinterpret_cast<volatile uint32_t*>(l1_ory);
    auto o_a   = reinterpret_cast<volatile uint32_t*>(l1_oa);
    auto o_b   = reinterpret_cast<volatile uint32_t*>(l1_ob);
    auto o_c   = reinterpret_cast<volatile uint32_t*>(l1_oc);
    auto o_dep = reinterpret_cast<volatile uint32_t*>(l1_odep);
    auto o_op  = reinterpret_cast<volatile uint32_t*>(l1_oop);
    auto o_col = reinterpret_cast<volatile uint32_t*>(l1_ocol);
    auto o_Mp  = reinterpret_cast<volatile uint32_t*>(l1_oM);

    // Per-element source guard: only the GLOBAL-last partial tile can hold
    // i >= N; the tile loop itself bounds each core's tile set (contiguous or
    // strided), so N is the only clamp the inner loop needs.
    const uint32_t i_hi = N;

    // ── count_only pass: just tally this core's visible quota ───────────
    if (count_only) {
        DeviceZoneScopedN("proj_count");
        uint32_t vcount = 0;
        for (uint32_t kk = 0, t = t_start; kk < t_count; kk++, t += t_stride) {
            noc_async_read(get_noc_addr(t, acc_m2x),   l1_m2x, TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_m2y),   l1_m2y, TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_depth), l1_dep, TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_rx),    l1_rx,  TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_ry),    l1_ry,  TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_op),    l1_op,  TILE_BYTES);
            noc_async_read_barrier();

            const uint32_t tbase = t * TILE_ELEMS;
            for (uint32_t il = 0; il < TILE_ELEMS; il++) {
                const uint32_t i = tbase + il;
                if (i >= i_hi) break;
                const float tz = bits_to_f(p_dep[il]);
                const float op = bits_to_f(p_op[il]);
                if (tz <= k_near || op < min_opacity) continue;
                const float mx = bits_to_f(p_m2x[il]);
                const float my = bits_to_f(p_m2y[il]);
                const float rx = bits_to_f(p_rx[il]);
                const float ry = bits_to_f(p_ry[il]);
                const bool valid = (mx + rx > 0.0f) && (mx - rx < img_w) &&
                                   (my + ry > 0.0f) && (my - ry < img_h) &&
                                   (rx > 0.0f) && (ry > 0.0f) &&
                                   (rx <= max_radius) && (ry <= max_radius);
                if (valid) vcount++;
            }
        }
        o_Mp[0] = vcount;
        noc_async_write(l1_oM, get_noc_addr(core_id, acc_counts), 4);
        noc_async_write_barrier();
        return;
    }

    DeviceZoneScopedN("proj_scatter");
    // ── scatter pass: write this core's visible elements at [base, ...) ─
    // GSPLAT_TT_PROJ_DEVICE_SCAN: the scan kernel computed this core's base +
    // is_last on-device into the base buffer (host repoints counts_addr at it);
    // read them over NoC instead of from the host-computed args. Reuse the M
    // accumulator L1 staging (l1_oM / o_Mp), unused in the scatter pass.
    uint32_t base_eff = base;
    uint32_t is_last_eff = is_last;
    if (device_scan) {
        noc_async_read(get_noc_addr(core_id, acc_counts), l1_oM, PAGE_BYTES);
        noc_async_read_barrier();
        base_eff = o_Mp[0];
        is_last_eff = o_Mp[1];
    }
    uint32_t g = base_eff;              // global compact output index
    uint32_t cur_page = g / PAGE_ELEMS; // current output page
    uint32_t slot = g % PAGE_ELEMS;     // current slot within cur_page
    uint32_t flush_lo = slot;           // first slot this core wrote in cur_page

    auto flush_page = [&](uint32_t page, uint32_t lo, uint32_t hi) {
        const uint32_t off = lo * 4;
        const uint32_t sz  = (hi - lo) * 4;
        noc_async_write(l1_opx  + off, get_noc_addr(page, acc_opx)  + off, sz);
        noc_async_write(l1_opy  + off, get_noc_addr(page, acc_opy)  + off, sz);
        noc_async_write(l1_orx  + off, get_noc_addr(page, acc_orx)  + off, sz);
        noc_async_write(l1_ory  + off, get_noc_addr(page, acc_ory)  + off, sz);
        noc_async_write(l1_oa   + off, get_noc_addr(page, acc_oa)   + off, sz);
        noc_async_write(l1_ob   + off, get_noc_addr(page, acc_ob)   + off, sz);
        noc_async_write(l1_oc   + off, get_noc_addr(page, acc_oc)   + off, sz);
        noc_async_write(l1_odep + off, get_noc_addr(page, acc_odep) + off, sz);
        noc_async_write(l1_oop  + off, get_noc_addr(page, acc_oop)  + off, sz);
#ifdef GATHER_EMIT_TPG
        noc_async_write(l1_otpg + off, get_noc_addr(page, acc_tpg) + off, sz);
#endif
        // colors: contiguous float subrange [lo*3, hi*3) of the 48-float group,
        // split at the 16-float color-page boundaries.
        const uint32_t f0 = page * COLOR_GROUP_FLOATS + lo * 3;
        const uint32_t f1 = page * COLOR_GROUP_FLOATS + hi * 3;
        uint32_t f = f0;
        while (f < f1) {
            const uint32_t cpage = f / PAGE_ELEMS;
            const uint32_t coff  = f % PAGE_ELEMS;
            uint32_t n = PAGE_ELEMS - coff;
            if (n > f1 - f) n = f1 - f;
            const uint32_t lf = f - page * COLOR_GROUP_FLOATS;  // local float idx in staging
            noc_async_write(l1_ocol + lf * 4,
                            get_noc_addr(cpage, acc_ocol) + coff * 4, n * 4);
            f += n;
        }
#ifdef GATHER_EMIT_BLENDREC
        // Each AoS record is its OWN full 64B page (record page index == g ==
        // page*16 + slot). Cores own disjoint, ordered g-ranges so every record
        // page is written by exactly one core (no neighbour boundary sharing).
        for (uint32_t s = lo; s < hi; ++s) {
            noc_async_write(l1_orec + s * 64,
                            get_noc_addr(page * PAGE_ELEMS + s, acc_brec), 64);
        }
#endif
        noc_async_write_barrier();
    };

    for (uint32_t kk = 0, t = t_start; kk < t_count; kk++, t += t_stride) {
        noc_async_read(get_noc_addr(t, acc_m2x),   l1_m2x, TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_m2y),   l1_m2y, TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_depth), l1_dep, TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_a),     l1_a,   TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_b),     l1_b,   TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_c),     l1_c,   TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_rx),    l1_rx,  TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_ry),    l1_ry,  TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_cr),    l1_cr,  TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_cg),    l1_cg,  TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_cb),    l1_cb,  TILE_BYTES);
        noc_async_read(get_noc_addr(t, acc_op),    l1_op,  TILE_BYTES);
        noc_async_read_barrier();

        const uint32_t tbase = t * TILE_ELEMS;
        for (uint32_t il = 0; il < TILE_ELEMS; il++) {
            const uint32_t i = tbase + il;
            if (i >= i_hi) break;

            const float tz = bits_to_f(p_dep[il]);
            const float op = bits_to_f(p_op[il]);
            if (tz <= k_near || op < min_opacity) continue;

            const float mx = bits_to_f(p_m2x[il]);
            const float my = bits_to_f(p_m2y[il]);
            const float rx = bits_to_f(p_rx[il]);
            const float ry = bits_to_f(p_ry[il]);
            const bool valid = (mx + rx > 0.0f) && (mx - rx < img_w) &&
                               (my + ry > 0.0f) && (my - ry < img_h) &&
                               (rx > 0.0f) && (ry > 0.0f) &&
                               (rx <= max_radius) && (ry <= max_radius);
            if (!valid) continue;

            o_px[slot]  = p_m2x[il];
            o_py[slot]  = p_m2y[il];
            o_rx[slot]  = p_rx[il];
            o_ry[slot]  = p_ry[il];
            o_a[slot]   = p_a[il];
            o_b[slot]   = p_b[il];
            o_c[slot]   = p_c[il];
            o_dep[slot] = p_dep[il];
            o_op[slot]  = p_op[il];
            o_col[slot * 3 + 0] = p_cr[il];
            o_col[slot * 3 + 1] = p_cg[il];
            o_col[slot * 3 + 2] = p_cb[il];
#ifdef GATHER_EMIT_TPG
            {
                const float px = bits_to_f(p_m2x[il]);
                const float py = bits_to_f(p_m2y[il]);
                const float rxv = bits_to_f(p_rx[il]);
                const float ryv = bits_to_f(p_ry[il]);
                const int min_x =
                    clampi(static_cast<int>((px - rxv) / tsf_f), 0, tiles_x_f - 1);
                const int max_x =
                    clampi(static_cast<int>((px + rxv) / tsf_f), 0, tiles_x_f - 1);
                const int min_y =
                    clampi(static_cast<int>((py - ryv) / tsf_f), 0, tiles_y_f - 1);
                const int max_y =
                    clampi(static_cast<int>((py + ryv) / tsf_f), 0, tiles_y_f - 1);
                const int w = max_x - min_x + 1;
                const int h = max_y - min_y + 1;
                o_tpg[slot] = w * h;
            }
#endif
#ifdef GATHER_EMIT_BLENDREC
            {
                volatile uint32_t* r = o_rec + slot * REC_WORDS;
                r[0] = p_a[il];   r[1] = p_b[il];   r[2] = p_c[il];
                r[3] = p_m2x[il]; r[4] = p_m2y[il]; r[5] = p_op[il];
                r[6] = p_cr[il];  r[7] = p_cg[il];  r[8] = p_cb[il];
                r[9] = 0; r[10] = 0; r[11] = 0; r[12] = 0;
                r[13] = 0; r[14] = 0; r[15] = 0;
            }
#endif

            slot++;
            g++;
            if (slot == PAGE_ELEMS) {
                flush_page(cur_page, flush_lo, PAGE_ELEMS);
                slot = 0;
                flush_lo = 0;
                cur_page++;
            }
        }
    }

    // Tail: flush the partial final page this core owns. The single core that
    // writes the final global element (is_last) zero-pads the remaining slots
    // of the last page so [M, M_pad) is well-defined, matching the single-core
    // output byte-for-byte.
    uint32_t hi = slot;
    if (is_last_eff && slot != 0) {
        for (uint32_t s = slot; s < PAGE_ELEMS; s++) {
            o_px[s] = 0; o_py[s] = 0; o_rx[s] = 0; o_ry[s] = 0;
            o_a[s] = 0; o_b[s] = 0; o_c[s] = 0;             o_dep[s] = 0; o_op[s] = 0;
            o_col[s * 3 + 0] = 0; o_col[s * 3 + 1] = 0; o_col[s * 3 + 2] = 0;
#ifdef GATHER_EMIT_TPG
            o_tpg[s] = 0;
#endif
#ifdef GATHER_EMIT_BLENDREC
            volatile uint32_t* r = o_rec + s * REC_WORDS;
            for (uint32_t w = 0; w < REC_WORDS; ++w) r[w] = 0;
#endif
        }
        hi = PAGE_ELEMS;
    }
    if (hi > flush_lo) {
        flush_page(cur_page, flush_lo, hi);
    }
}
