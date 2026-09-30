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
//   35    : core_id     (count/base slot: 2*core + mover)
//   36    : counts DRAM base (per-slot counts, one 64B page per slot)
//   37    : AoS blend-record DRAM base
//   38    : tile stride
//   39    : device_scan (read base/is_last from counts_addr page core_id)
//   40    : mover       (0 = BRISC, 1 = NCRISC; selects the L1 staging copy)
//   41    : visibility-mask DRAM base (one 128B page = 1024 bits per tile)
//
// VISIBILITY MASK (task #75): the count pass evaluates the predicate once per
// Gaussian and writes a 1024-bit mask per tile (bit il of word il/32 set iff
// element t*1024+il is visible). The scatter pass reads the mask instead of
// re-running the soft-float predicate, walks only the set bits, and skips the
// 12 tile reads of tiles with no visible element. Same elements, same order,
// so every output byte is unchanged.
//
// DUAL DATA MOVER: the host gives each core's (strided) tile list to its two
// movers as two contiguous halves, BRISC the first, NCRISC the second, and
// numbers the slots 2*core + mover. The compaction order is the concatenation
// of the slots' visible elements in slot order, i.e. core 0's first half,
// core 0's second half, core 1's first half, ... — exactly the single-mover
// order. So every output byte is unchanged; the page shared by two adjacent
// slots (same core or not) is handled by the boundary-page rule below.
//
// COMPILE-TIME ARGS: 24 TensorAccessorArgs in the order
//   m2x,m2y,depth,a,b,c,rx,ry, col_r,col_g,col_b,op,
//   o_px,o_py,o_rx,o_ry,o_a,o_b,o_c,o_depth,o_op, o_colors, o_M, o_counts,
//   then o_blendrec and the visibility mask.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dm_fp32.h"
#include "gather_visible_pred.h"

namespace {

constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t TILE_BYTES = TILE_ELEMS * 4;  // 4096
constexpr uint32_t PAGE_ELEMS = 16;
constexpr uint32_t PAGE_BYTES = PAGE_ELEMS * 4;  // 64
constexpr uint32_t COLOR_GROUP_FLOATS = PAGE_ELEMS * 3;  // 48 floats / 16-Gaussian group
constexpr uint32_t MASK_WORDS = TILE_ELEMS / 32;          // 32 words / tile
constexpr uint32_t MASK_BYTES = MASK_WORDS * 4;           // 128

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
    // fp32 bits: compared with dm_fp32 (integer ops), see visible_bits().
    const uint32_t min_opacity = get_arg_val<uint32_t>(25);
    const uint32_t k_near      = get_arg_val<uint32_t>(26);
    const uint32_t img_w       = get_arg_val<uint32_t>(27);
    const uint32_t img_h       = get_arg_val<uint32_t>(28);
    const uint32_t max_radius  = get_arg_val<uint32_t>(29);
    const uint32_t t_start    = get_arg_val<uint32_t>(31);
    const uint32_t t_count    = get_arg_val<uint32_t>(32);
    const uint32_t base       = get_arg_val<uint32_t>(33);
    const uint32_t is_last    = get_arg_val<uint32_t>(34);
    const uint32_t core_id    = get_arg_val<uint32_t>(35);
    const uint32_t counts_addr= get_arg_val<uint32_t>(36);
    // S1 (BLEND_AOS): contiguous AoS blend record buffer. One 64B page per
    // compacted gaussian == {a,b,c,px,py,op,cr,cg,cb, 0..}. Lets the blend reader
    // fetch a candidate with ONE contiguous read instead of 7-9 SoA pages.
    const uint32_t o_blendrec_addr = get_arg_val<uint32_t>(37);
    // GSPLAT_TT_PROJ_BALANCE: tile stride. 1 = contiguous chunk (this core owns
    // tiles [t_start, t_start+t_count)); num_cores = interleaved/strided (this
    // core owns t_start, t_start+stride, t_start+2*stride, ...).
    const uint32_t t_stride = get_arg_val<uint32_t>(38);
    // GSPLAT_TT_PROJ_DEVICE_SCAN: 1 => read (base, is_last) from counts_addr
    // (repointed by the host at the device-scan base buffer) page core_id,
    // instead of the host-computed `base`/`is_last` args (which are 0).
    const uint32_t device_scan = get_arg_val<uint32_t>(39);
    // Dual data mover: 0 = BRISC, 1 = NCRISC. Both run this kernel on the same
    // core, each on its own slice of the core's tile list with its own count /
    // base slot (core_id is the (core, mover) slot). Every staging CB is
    // allocated twice as deep; mover m uses the m-th copy, so the two movers
    // share no L1.
    const uint32_t mover = get_arg_val<uint32_t>(40);
    const uint32_t mask_addr = get_arg_val<uint32_t>(41);
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
    constexpr auto a_brec  = TensorAccessorArgs<a_counts.next_compile_time_args_offset()>();
    constexpr auto a_mask  = TensorAccessorArgs<a_brec.next_compile_time_args_offset()>();
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
    const auto acc_brec  = TensorAccessor(a_brec,  o_blendrec_addr, PAGE_BYTES);
    const auto acc_mask  = TensorAccessor(a_mask,  mask_addr, MASK_BYTES);

    constexpr uint32_t CB_M2X = 0, CB_M2Y = 1, CB_DEP = 2, CB_A = 3, CB_B = 4,
                       CB_C = 5, CB_RX = 6, CB_RY = 7, CB_CR = 8, CB_CG = 9,
                       CB_CB = 10, CB_OP = 11;
    constexpr uint32_t CB_OPX = 12, CB_OPY = 13, CB_ORX = 14, CB_ORY = 15,
                       CB_OA = 16, CB_OB = 17, CB_OC = 18, CB_ODEP = 19,
                       CB_OOP = 20, CB_OCOL = 21, CB_OM = 22;
    constexpr uint32_t CB_OREC = 23;  // 16 records x 64B AoS staging
    constexpr uint32_t CB_MASK = 24;  // 2 x 128B visibility-mask staging
    const uint32_t l1_mask = get_write_ptr(CB_MASK) + mover * 2 * MASK_BYTES;
    constexpr uint32_t REC_WORDS = 16;  // 64B / 4
    const uint32_t l1_orec = get_write_ptr(CB_OREC) + mover * PAGE_ELEMS * PAGE_BYTES;
    auto o_rec = reinterpret_cast<volatile uint32_t*>(l1_orec);

    const uint32_t l1_m2x = get_write_ptr(CB_M2X) + mover * TILE_BYTES;
    const uint32_t l1_m2y = get_write_ptr(CB_M2Y) + mover * TILE_BYTES;
    const uint32_t l1_dep = get_write_ptr(CB_DEP) + mover * TILE_BYTES;
    const uint32_t l1_a   = get_write_ptr(CB_A) + mover * TILE_BYTES;
    const uint32_t l1_b   = get_write_ptr(CB_B) + mover * TILE_BYTES;
    const uint32_t l1_c   = get_write_ptr(CB_C) + mover * TILE_BYTES;
    const uint32_t l1_rx  = get_write_ptr(CB_RX) + mover * TILE_BYTES;
    const uint32_t l1_ry  = get_write_ptr(CB_RY) + mover * TILE_BYTES;
    const uint32_t l1_cr  = get_write_ptr(CB_CR) + mover * TILE_BYTES;
    const uint32_t l1_cg  = get_write_ptr(CB_CG) + mover * TILE_BYTES;
    const uint32_t l1_cb  = get_write_ptr(CB_CB) + mover * TILE_BYTES;
    const uint32_t l1_op  = get_write_ptr(CB_OP) + mover * TILE_BYTES;

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

    const uint32_t l1_opx = get_write_ptr(CB_OPX) + mover * PAGE_BYTES;
    const uint32_t l1_opy = get_write_ptr(CB_OPY) + mover * PAGE_BYTES;
    const uint32_t l1_orx = get_write_ptr(CB_ORX) + mover * PAGE_BYTES;
    const uint32_t l1_ory = get_write_ptr(CB_ORY) + mover * PAGE_BYTES;
    const uint32_t l1_oa  = get_write_ptr(CB_OA) + mover * PAGE_BYTES;
    const uint32_t l1_ob  = get_write_ptr(CB_OB) + mover * PAGE_BYTES;
    const uint32_t l1_oc  = get_write_ptr(CB_OC) + mover * PAGE_BYTES;
    const uint32_t l1_odep= get_write_ptr(CB_ODEP) + mover * PAGE_BYTES;
    const uint32_t l1_oop = get_write_ptr(CB_OOP) + mover * PAGE_BYTES;
    const uint32_t l1_ocol= get_write_ptr(CB_OCOL) + mover * COLOR_GROUP_FLOATS * 4;
    const uint32_t l1_oM  = get_write_ptr(CB_OM) + mover * PAGE_BYTES;

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
        // Two staging sets: the count pass only needs 6 streams, so the next
        // tile is prefetched into the a/b/c/col_r/col_g/col_b tiles (unused
        // here) while the current one is tested.
        const uint32_t set_m2x[2] = {l1_m2x, l1_a};
        const uint32_t set_m2y[2] = {l1_m2y, l1_b};
        const uint32_t set_dep[2] = {l1_dep, l1_c};
        const uint32_t set_rx[2]  = {l1_rx,  l1_cr};
        const uint32_t set_ry[2]  = {l1_ry,  l1_cg};
        const uint32_t set_op[2]  = {l1_op,  l1_cb};
        auto issue = [&](uint32_t t, uint32_t k) {
            noc_async_read(get_noc_addr(t, acc_m2x),   set_m2x[k], TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_m2y),   set_m2y[k], TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_depth), set_dep[k], TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_rx),    set_rx[k],  TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_ry),    set_ry[k],  TILE_BYTES);
            noc_async_read(get_noc_addr(t, acc_op),    set_op[k],  TILE_BYTES);
        };
        uint32_t vcount = 0;
        if (t_count > 0) issue(t_start, 0);
        for (uint32_t kk = 0, t = t_start; kk < t_count; kk++, t += t_stride) {
            const uint32_t k = kk & 1u;
            noc_async_read_barrier();
            if (kk + 1 < t_count) issue(t + t_stride, k ^ 1u);
            auto q_m2x = reinterpret_cast<volatile uint32_t*>(set_m2x[k]);
            auto q_m2y = reinterpret_cast<volatile uint32_t*>(set_m2y[k]);
            auto q_dep = reinterpret_cast<volatile uint32_t*>(set_dep[k]);
            auto q_rx  = reinterpret_cast<volatile uint32_t*>(set_rx[k]);
            auto q_ry  = reinterpret_cast<volatile uint32_t*>(set_ry[k]);
            auto q_op  = reinterpret_cast<volatile uint32_t*>(set_op[k]);
            const uint32_t tbase = t * TILE_ELEMS;
            const uint32_t n_el = (tbase >= i_hi) ? 0u
                : (i_hi - tbase < TILE_ELEMS ? i_hi - tbase : TILE_ELEMS);
            // Build the mask words in registers (no L1 read-after-write), then
            // store each word once. The mask page of this tile goes out while
            // the next tile is tested; the other staging copy is reused two
            // tiles later, after noc_async_writes_flushed.
            const uint32_t l1_m = l1_mask + k * MASK_BYTES;
            noc_async_writes_flushed();
            auto mw = reinterpret_cast<volatile uint32_t*>(l1_m);
            for (uint32_t w = 0; w < MASK_WORDS; w++) {
                uint32_t bits = 0;
                const uint32_t il0 = w * 32;
                const uint32_t il1 = (il0 + 32 < n_el) ? il0 + 32 : n_el;
                for (uint32_t il = il0; il < il1; il++) {
                    if (gather_pred::visible_at(q_dep, q_op, q_m2x, q_m2y, q_rx, q_ry, il,
                                   k_near, min_opacity, img_w, img_h, max_radius))
                        bits |= 1u << (il - il0);
                }
                vcount += static_cast<uint32_t>(__builtin_popcount(bits));
                mw[w] = bits;
            }
            noc_async_write(l1_m, get_noc_addr(t, acc_mask), MASK_BYTES);
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
        // Each AoS record is its OWN full 64B page (record page index == g ==
        // page*16 + slot). Cores own disjoint, ordered g-ranges so every record
        // page is written by exactly one core (no neighbour boundary sharing).
        for (uint32_t s = lo; s < hi; ++s) {
            noc_async_write(l1_orec + s * 64,
                            get_noc_addr(page * PAGE_ELEMS + s, acc_brec), 64);
        }
        noc_async_write_barrier();
    };

    // AoS record words 9..15 are always zero and nothing else writes them,
    // so they are zeroed once here instead of per visible element. iter 132
    // reverted the iter-131 birth-side UNORM16 pack: op/color are written as
    // fp32 (keeps the pack off the BRISC proj_scatter long pole); the UNORM16
    // pack runs once per gaussian on the NCRISC side (sort_bin
    // pack_invariants), which publishes the packed words into blendrec[10],[11]
    // for the materialize overflow gather.
    for (uint32_t s = 0; s < PAGE_ELEMS; ++s) {
        volatile uint32_t* r = o_rec + s * REC_WORDS;
        for (uint32_t w = 9; w < REC_WORDS; ++w) r[w] = 0;
    }

    auto p_mask = reinterpret_cast<volatile uint32_t*>(l1_mask);
    for (uint32_t kk = 0, t = t_start; kk < t_count; kk++, t += t_stride) {
        // The count pass's mask for this tile; tiles with no visible element
        // skip the 12 tile reads.
        noc_async_read(get_noc_addr(t, acc_mask), l1_mask, MASK_BYTES);
        noc_async_read_barrier();
        uint32_t mbits[MASK_WORDS];
        uint32_t any = 0;
        for (uint32_t w = 0; w < MASK_WORDS; w++) {
            mbits[w] = p_mask[w];
            any |= mbits[w];
        }
        if (any == 0) continue;
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
        for (uint32_t w = 0; w < MASK_WORDS; w++)
        for (uint32_t bits = mbits[w]; bits != 0; bits &= bits - 1) {
            const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
            const uint32_t dep = p_dep[il], op = p_op[il], mx = p_m2x[il];
            const uint32_t my = p_m2y[il], rx = p_rx[il], ry = p_ry[il];
            const uint32_t a = p_a[il], b = p_b[il], c = p_c[il];
            const uint32_t cr = p_cr[il], cg = p_cg[il], cb = p_cb[il];

            o_px[slot]  = mx;
            o_py[slot]  = my;
            o_rx[slot]  = rx;
            o_ry[slot]  = ry;
            o_a[slot]   = a;
            o_b[slot]   = b;
            o_c[slot]   = c;
            o_dep[slot] = dep;
            o_op[slot]  = op;
            o_col[slot * 3 + 0] = cr;
            o_col[slot * 3 + 1] = cg;
            o_col[slot * 3 + 2] = cb;
            {
                // Words 9..15 stay zero from the pre-fill above the loop.
                volatile uint32_t* r = o_rec + slot * REC_WORDS;
                r[0] = a;  r[1] = b;  r[2] = c;
                r[3] = mx; r[4] = my; r[5] = op;
                r[6] = cr; r[7] = cg; r[8] = cb;
            }

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
            volatile uint32_t* r = o_rec + s * REC_WORDS;
            for (uint32_t w = 0; w < REC_WORDS; ++w) r[w] = 0;
        }
        hi = PAGE_ELEMS;
    }
    if (hi > flush_lo) {
        flush_page(cur_page, flush_lo, hi);
    }
}
