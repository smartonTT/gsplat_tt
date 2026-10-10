// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// FUSED project(means_cam) + pfwc compute kernel — iter-133 (program fusion #1).
//
// This kernel merges the former standalone `project_means_cam` program and the
// `pfwc` program into ONE device program, eliminating one per-program launch's
// firmware (BRISC-FW long pole) and the fp32 means_cam DRAM store+reload between
// the two stages. The means_cam handoff is now a CORE-LOCAL L1 pass: the
// world→camera transform writes its result straight into the per-chunk L1 DEST
// and folds the pfwc translation in, emitting tx/ty/tz directly — no NoC / DRAM
// round-trip.
//
// BIT-IDENTICAL: the former means_cam stored mc_j = R·m to fp32 DRAM (lossless)
// and pfwc reloaded it then added t_j (tx = mc_j + t_j). Here mc_j stays in an
// fp32 DEST slot and add_unary(t_j) is applied to the same fp32 value, so the
// emitted tx/ty/tz are bit-for-bit the values the two-program path produced.
// Everything downstream (steps 2..11) is the verbatim pfwc body on the same
// inputs.
//
// LAYOUT
//   Inputs  CB_MX/MY/MZ (0,1,2)  — fp32 world-space means SoA (reader-filled).
//           CB_C00..CB_C22 (3..8) — fp32 cov3d unique SoA (reader-filled).
//   Outputs CB_M2X..CB_RY (9..16) — the 8 pfwc outputs (writer-drained).
//   Scratch CB 17..29             — tx/ty/tz/inv_tz, cc00..cc22, conic a/b/c.
//   (33→30 CBs: the old means_cam MCX/MCY/MCZ bridge is gone — tx/ty/tz IS the
//   L1 bridge now.)
//
// RUNTIME ARGS (uint32_t, layout identical to the former pfwc_device.cpp):
//   0           : num_chunks
//   1..9        : R bits (row-major r00..r22) — USED by the fused transform
//   10..12      : t0, t1, t2 (translation)    — USED by the fused transform
//   13..16      : fx, fy, cx, cy
//   17..52      : 36 cov_cam scales (6 entries × 6 scales)
//   53          : k_bits (fp32 of k = 3.0 default)
//   54          : neg_fx_bits (fp32 of -fx)
//   55          : neg_fy_bits (fp32 of -fy)
//   56..64      : PFWC_VIS only (task #99, lever 2): k_near, min_opacity, image
//                 width, image height, max_radius, 1/tile_size, tiles_x - 1,
//                 tiles_y - 1, edge tau, all fp32 bits.
//
// PFWC_VIS (task #99, GSPLAT_TT_SFPU_VIS): step 11.5 also evaluates the gather
// visibility predicate and the tile_assign K1 rectangle on the SFPU and emits
// two more tiles, the tpg word and the aabb word per gaussian (encoding in
// render/kernels/dataflow/vis_tile.h). Inputs: tz (CB_TMP_TZ), opacity (new
// reader stream CB_OP) and mean_x/mean_y/rx/ry, which steps 4/5/10/11 also pack
// into scratch CBs. tests/unit/test_vis_lever2.cpp models pfwc_vis_one lane by
// lane and checks it against gather_visible_pred.h + tile_assign_bbox.cpp.
//
// PFWC_PRECULL (task #140, lever C, GSPLAT_TT_PRECULL=1, needs PFWC_VIS): step
// 11.6 (inside step 11.5's DEST acquire) shrinks the radii that step 11.5 turns into the tile rectangle to the
// opacity-aware extent ceil(sqrt(max(2 ln(op) + c0, 0) * a) + 1), with c0 =
// 2 ln(1 / floor) + margin (the microblock band cull's floor; runtime arg 65
// holds it with the constants of the log2 upper bound folded in).
// The band cull keeps nothing outside that extent, so the records it drops
// are dead ones (mask 0) and the image is unchanged. The radius stays as is
// when the lane is ill-conditioned (64 det < a c), non-finite or over the
// radius limit (arg 66, min(max_radius, 4096)), so the max_radius test is
// unchanged. CB_RX / CB_RY (the radii outputs) keep the 3-sigma values.
// tests/unit/test_precull.cpp models the lane math against the band cull.
// PRECULL_PC (task #157, GSPLAT_TT_PRECULL=2): the pixel-centre rect, r' =
// max(sqrt(t cov) - 3/8, 1/8) (not an integer). A tile's pixel centres meet
// [m - e, m + e] iff the tile meets [m - (e - 1/2), m + (e - 1/2)], so this
// keeps the same tiles with a 1/8 px slack where the integer rect keeps 1.5 to
// 2.5 px more on each side: bicycle host model, 14.0% of the records dropped vs
// 3.9% (docs/precull-t157).

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/rounding.h"

// Task #197: PFWC_STEPRISC (0 unpack, 1 math, 2 pack, 9 all) picks the TRISC(s)
// that carry the counters (all five RISCs need GSPLAT_TT_KCFG_EXTRA_KB headroom).
#if defined(PFWC_STEPCYC) && (PFWC_STEPRISC == 9 || (PFWC_STEPRISC == 0 && defined(TRISC_UNPACK)) || \
                              (PFWC_STEPRISC == 1 && defined(TRISC_MATH)) ||   \
                              (PFWC_STEPRISC == 2 && defined(TRISC_PACK)))
#define PFWC_SC_ON 1
#endif

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#ifdef PFWC_VIS
#include "sfpu/ckernel_sfpu_converter.h"
#endif
#ifdef PFWC_COVCAM_SFPU
#include "pfwc_covcam_sfpu.h"
#endif
#ifdef PFWC_COV2D_SFPU
#include "pfwc_cov2d_sfpu.h"
#endif
#ifdef PFWC_RECIP_NEWTON
#include "pfwc_recip_nr.h"
#endif
#if defined(PFWC_FUSE_PROJ) || defined(PFWC_FUSE_CCAC)
#include "pfwc_fuse_sfpu.h"
#endif
#if defined(PFWC_FUSE_CCAC) && !(defined(PFWC_COVCAM_SFPU) && defined(PFWC_COV2D_SFPU))
#error "PFWC_FUSE_CCAC needs PFWC_COVCAM_SFPU and PFWC_COV2D_SFPU"
#endif
#endif

namespace {

// Means inputs (CBs 0,1,2) — reader fills with world-space means SoA. These are
// the same physical CBs the standalone means_cam read; the fused transform turns
// them into camera-space tx/ty/tz in L1.
constexpr uint32_t CB_MX = 0;
constexpr uint32_t CB_MY = 1;
constexpr uint32_t CB_MZ = 2;
// cov3d unique inputs.
constexpr uint32_t CB_C00 = 3;
constexpr uint32_t CB_C01 = 4;
constexpr uint32_t CB_C02 = 5;
constexpr uint32_t CB_C11 = 6;
constexpr uint32_t CB_C12 = 7;
constexpr uint32_t CB_C22 = 8;

// Outputs — 8 SoA streams.
constexpr uint32_t CB_M2X = 9;
constexpr uint32_t CB_M2Y = 10;
constexpr uint32_t CB_DEP = 11;
constexpr uint32_t CB_A   = 12;
constexpr uint32_t CB_B   = 13;
constexpr uint32_t CB_C   = 14;
constexpr uint32_t CB_RX  = 15;
constexpr uint32_t CB_RY  = 16;

// Scratch (per-chunk intermediates). CB_TMP_TX/TY/TZ double as the means_cam→pfwc
// L1 bridge: the fused transform emits camera-space tx/ty/tz (= R·m + t) here.
constexpr uint32_t CB_TMP_TX     = 17;
constexpr uint32_t CB_TMP_TY     = 18;
constexpr uint32_t CB_TMP_TZ     = 19;
constexpr uint32_t CB_TMP_INV_TZ = 20;
constexpr uint32_t CB_TMP_CC00   = 21;
constexpr uint32_t CB_TMP_CC01   = 22;
constexpr uint32_t CB_TMP_CC02   = 23;
constexpr uint32_t CB_TMP_CC11   = 24;
constexpr uint32_t CB_TMP_CC12   = 25;
constexpr uint32_t CB_TMP_CC22   = 26;
constexpr uint32_t CB_TMP_A      = 27;
constexpr uint32_t CB_TMP_B      = 28;
constexpr uint32_t CB_TMP_C      = 29;
#ifdef PFWC_VIS
// Lever 2 (task #99): opacity input, scratch copies of the outputs the
// predicate needs, and the two word tiles (vis_tile.h).
constexpr uint32_t CB_OP     = 30;
constexpr uint32_t CB_TMP_MX = 31;
constexpr uint32_t CB_TMP_MY = 32;
constexpr uint32_t CB_TMP_RX = 33;
constexpr uint32_t CB_TMP_RY = 34;
constexpr uint32_t CB_TPG    = 35;
constexpr uint32_t CB_AABB   = 36;
#endif

// ---- Per-step wall cycles (GSPLAT_TT_PFWC_STEPCYC=1, task #197; default OFF) ----
// Every TRISC (UNPACK, MATH, PACK) adds the wall cycles (1350 MHz) between step
// boundaries into one counter per step, summed over the core's chunks, and
// records "pfwc_pc" n wall init s0..s12 at kernel end. Steps: 0 input wait, 1 transform,
// 2 recip, 3 depth, 4 means, 5 cov_cam, 6 a, 7 b, 8 c, 9 conic, 10 radii x,
// 11 radii y, 12 vis + pops. =2 also splits step 5 (cov_cam: 36 copies, 36
// mul_unary, 30 add_binary, 6 packs) by call type on each thread:
// "PO copy mulu addb acq pack".
#ifdef PFWC_SC_ON
// One array, printed by one loop (keeps the binary small): n, wall, init,
// 13 steps, then 5 cov_cam parts when PFWC_STEPCYC >= 2.
constexpr uint32_t PC_N = 13;
constexpr uint32_t PC_NV = 3 + PC_N + (PFWC_STEPCYC >= 2 ? 5 : 0);
uint32_t g_v[3 + PC_N + 5];
uint32_t* const g_pc = g_v + 3;
uint32_t* const g_po = g_v + 3 + PC_N;
uint32_t g_pc_t = 0;
inline uint32_t pc_now() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
#define PC_MARK(i) do { const uint32_t n_ = pc_now(); g_pc[i] += n_ - g_pc_t; g_pc_t = n_; } while (0)
#if PFWC_STEPCYC >= 2
#define PO_T0() const uint32_t po_t_ = pc_now()
#define PO_ACC(i) g_po[i] += pc_now() - po_t_
#else
#define PO_T0() ((void)0)
#define PO_ACC(i) ((void)0)
#endif
#else
#define PC_MARK(i) ((void)0)
#define PO_T0() ((void)0)
#define PO_ACC(i) ((void)0)
#endif

constexpr uint32_t COV3D_CB[6] = {CB_C00, CB_C11, CB_C22, CB_C01, CB_C02, CB_C12};
constexpr uint32_t CC_SCRATCH[6] = {
    CB_TMP_CC00, CB_TMP_CC01, CB_TMP_CC02,
    CB_TMP_CC11, CB_TMP_CC12, CB_TMP_CC22};

// Task #207 (GSPLAT_TT_PFWC_WRITER_SPLIT): odd chunks go to the NCRISC
// writer's output set (writer_pfwc_split.cpp, pfwc_wsplit::odd_cb).
#ifdef PFWC_WSPLIT
#define OCB(cb) ((chunk & 1u) ? ((cb) < 32u ? (cb) + 32u : (cb) + 14u) : (cb))
#else
#define OCB(cb) (cb)
#endif

inline void emit_dst(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// Pack tile from DEST slot to scratch CB (no cb_pop_front; callers pop later).
inline void emit_scratch(uint32_t idst, uint32_t cb_out) {
    cb_reserve_back(cb_out, 1);
    pack_tile(idst, cb_out);
    cb_push_back(cb_out, 1);
}

// dst[idst] = sum_k scales[k] * COV3D_CB[k]  (cov_cam entry expansion).
inline void compute_cc_entry_to_scratch(uint32_t base_arg, uint32_t cb_out_scratch) {
    const uint32_t s0 = get_arg_val<uint32_t>(base_arg + 0);
    const uint32_t s1 = get_arg_val<uint32_t>(base_arg + 1);
    const uint32_t s2 = get_arg_val<uint32_t>(base_arg + 2);
    const uint32_t s3 = get_arg_val<uint32_t>(base_arg + 3);
    const uint32_t s4 = get_arg_val<uint32_t>(base_arg + 4);
    const uint32_t s5 = get_arg_val<uint32_t>(base_arg + 5);
    const uint32_t scales[6] = {s0, s1, s2, s3, s4, s5};

    { PO_T0(); tile_regs_acquire(); PO_ACC(3); }
    { PO_T0(); copy_tile_to_dst_init_short(COV3D_CB[0]); copy_tile(COV3D_CB[0], 0, 0); PO_ACC(0); }
    { PO_T0(); mul_unary_tile(0, scales[0]); PO_ACC(1); }
    for (uint32_t k = 1; k < 6; k++) {
        { PO_T0(); copy_tile_to_dst_init_short(COV3D_CB[k]); copy_tile(COV3D_CB[k], 0, 1); PO_ACC(0); }
        { PO_T0(); mul_unary_tile(1, scales[k]); PO_ACC(1); }
        { PO_T0(); add_binary_tile(0, 1, 0); PO_ACC(2); }
    }
    PO_T0();
    tile_regs_commit();
    tile_regs_wait();
    emit_scratch(0, cb_out_scratch);
    tile_regs_release();
    PO_ACC(4);
}

#ifdef PFWC_COVCAM_SFPU
#ifdef TRISC_MATH
inline void covcam_sfpu_math() {
    pfwc_covcam::run([](uint32_t j) { return get_arg_val<uint32_t>(17 + j); });
}
#endif

// Task #206 (GSPLAT_TT_PFWC_COVCAM_SFPU=1): all six cov_cam entries in one acquire:
// 6 copies, one SFPU pass (pfwc_covcam_sfpu.h), 6 packs. Bit-identical to six
// compute_cc_entry_to_scratch calls. PFWC_STEPCYC=2 books the SFPU pass as mulu.
inline void covcam_sfpu_to_scratch() {
    { PO_T0(); tile_regs_acquire(); PO_ACC(3); }
    {
        PO_T0();
        copy_tile_to_dst_init_short(COV3D_CB[0]);  // the six cov3d CBs share one format
        for (uint32_t k = 0; k < 6; k++) copy_tile(COV3D_CB[k], 0, k);
        PO_ACC(0);
    }
    {
        PO_T0();
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
        MATH((covcam_sfpu_math()));
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
        PO_ACC(1);
    }
    PO_T0();
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t e = 0; e < 6; e++) emit_scratch(e, CC_SCRATCH[e]);
    tile_regs_release();
    PO_ACC(4);
}
#endif

#ifdef TRISC_MATH
// A1 conic fold for ONE 32-lane vector V of the chunk tile. DEST tile 0 holds
// cov2d_a, tile 1 cov2d_b, tile 2 cov2d_c. Fold a,b,c -> A,B,C in place. Lifted
// verbatim from alpha_blend_compute_mb.cpp so the emitted A,B,C are bit-identical
// to what the blend kernel would have computed.
template <uint32_t V>
__attribute__((noinline, noipa)) void pfwc_conic_one() {
    using namespace sfpi;
    vFloat cov_a = dst_reg[0 * 32 + V];
    vFloat cov_b = dst_reg[1 * 32 + V];
    vFloat cov_c = dst_reg[2 * 32 + V];
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    vFloat det_floor = 1e-6f;
    vec_min_max(det_floor, det);          // det = max(det, 1e-6)
    vFloat inv = approx_recip(det);
    inv = inv * (vFloat(2.0f) - det * inv);
    inv = inv * (vFloat(2.0f) - det * inv);
    dst_reg[0 * 32 + V] = vFloat(-0.5f) * (cov_c * inv);  // A
    dst_reg[1 * 32 + V] = cov_b * inv;                    // B
    dst_reg[2 * 32 + V] = vFloat(-0.5f) * (cov_a * inv);  // C
}
#endif

template <uint32_t V>
inline void pfwc_conic_unroll() {
    if constexpr (V < 32) {
        MATH((pfwc_conic_one<V>()));
        pfwc_conic_unroll<V + 1>();
    }
}

#ifdef PFWC_VIS
// Lever 2 (task #99). DEST slots: 0 tz, 1 opacity, 2 mean_x, 3 mean_y, 4 rx,
// 5 ry; slot 6 holds the 8 runtime parameters as lane-broadcast vectors
// (staged once per tile, task #68 pattern); outputs go to slot 0 (tpg word)
// and slot 1 (aabb word).
constexpr uint32_t DR_VP = 6 * 32;
constexpr uint32_t VP_KN = 0, VP_MO = 1, VP_W = 2, VP_H = 3, VP_R = 4, VP_INV = 5, VP_TX1 = 6,
                   VP_TY1 = 7, VP_TAU = 8;
constexpr uint32_t VIS_PARAMS = 9;

#ifdef TRISC_MATH
// pfwc_vis_one<V> is instantiated for V < 16 only and run twice per tile, the
// second time with the DEST counter advanced by 16 vectors (pfwc_vis_half):
// 32 noinline copies made the PFWC_VIS program 83 KB, over the 70.6 KB kernel
// config buffer (task #102). The counter also offsets the slot-6 parameter
// reads, so the parameters are staged at rows p and 16 + p.
inline void pfwc_vis_stage(const uint32_t* b) {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
    for (uint32_t h = 0; h < 32; h += 16) {
        dst_reg[DR_VP + h + VP_KN] = Converter::as_float(b[VP_KN]);
        dst_reg[DR_VP + h + VP_MO] = Converter::as_float(b[VP_MO]);
        dst_reg[DR_VP + h + VP_W] = Converter::as_float(b[VP_W]);
        dst_reg[DR_VP + h + VP_H] = Converter::as_float(b[VP_H]);
        dst_reg[DR_VP + h + VP_R] = Converter::as_float(b[VP_R]);
        dst_reg[DR_VP + h + VP_INV] = Converter::as_float(b[VP_INV]);
        dst_reg[DR_VP + h + VP_TX1] = Converter::as_float(b[VP_TX1]);
        dst_reg[DR_VP + h + VP_TY1] = Converter::as_float(b[VP_TY1]);
        dst_reg[DR_VP + h + VP_TAU] = Converter::as_float(b[VP_TAU]);
    }
}

// Advance the SFPU DEST counter by 16 vectors (32 rows); the INCRWC field is
// signed 4-bit ([-8, 7] rows), so in 4-row steps.
inline void pfwc_vis_half() {
    for (uint32_t k = 0; k < 8; k++) sfpi::dst_reg += 2;
}

// Tile coordinate of a scaled edge q = fl(m +- r) * 2^-s: clamp to [0, hi],
// then floor. For q in [0, 2^23), q + 2^23 rounds to one of the two integers
// around q (any faithful rounding), and the r > q fix-up turns it into floor.
// trunc(q) clamped == floor(clamp(q)) for an integer hi >= 0, which is what
// tile_assign_bbox.cpp computes ((int) cast, then clampi).
sfpi_inline sfpi::vFloat pfwc_vis_cell(sfpi::vFloat q, sfpi::vFloat hi) {
    using namespace sfpi;
    vFloat lo = 0.0f;
    vec_min_max(lo, q);  // q = max(q, 0)
    vec_min_max(q, hi);  // q = min(q, hi)
    vFloat r = (q + 8388608.0f) - 8388608.0f;
    v_if(r > q) { r = r - 1.0f; }
    v_endif;
    return r;
}

// One 32-lane vector V. Every comparison is a sign/zero test of one rounded
// fp32 difference or sum, which has the sign of the exact value, so for finite
// inputs the predicate is exact whatever the rounding (fl(m - r) is exact near
// the image edge: r is an integer and |m| < 2^24 there). In the rectangle,
// fl(m + r) is the one rounding that matters: if SFPMAD rounded it differently
// from nearest-even, floor(fl(m + r) / tile) could only change when
// q = fl(m + r) / tile is within tau of an integer, so those still-visible
// lanes go to the writer's exact path too (tau = runtime arg 64,
// GSPLAT_TT_VIS_EDGE_TAU, default 2^-12; ~0.1% of visible lanes). inf/NaN
// inputs also give RECHECK. GSPLAT_TT_SFPU_VIS=2 cross-checks it all on device.
// The SFPU has 8 vector registers and sfpi cannot spill them, so consumed
// input rows are reused for intermediates (DEST round-trips fp32 exactly) and
// at most ~6 vectors are live: slot 7 = recheck flag, then x0 -> slot 0,
// y0 -> slot 1, x1 - x0 -> slot 2, y1 - y0 -> slot 3, sx / sy parked in 4 / 5.
#ifdef PFWC_VIS_FAST
// Task #488 (GSPLAT_TT_PFWC_VIS_FAST): the same words after the writer with
// 9 v_ifs instead of 25 (lane model: tests/unit/test_vis_lever2.cpp
// sfpu_model_fast). The "<= 0 fails" terms are folded with vec_min_max into
// m1 and the "> 0 fails" terms into m2; min / max return one of their inputs,
// so the sign / zero test of the result is exact. Lanes fail iff m1 <= 0.
// The non-finite flag is the exponent of s = tz + op + sx + sy (inf / NaN
// propagate; an overflow only adds RECHECK lanes). Both axes' edge-tau tests
// become e = min over (q - x1 - tau, max(d - tau, -d)), one test under
// "not failed"; RCK holds s, and inf once flagged.
// Slots after A: 1 m1, 2 dx, 3 dy, 4 sx, 5 sy, 7 s. After B: 0 x0, 2 w - 1,
// 3 h - 1, 4 e_x, 5 y0.
template <uint32_t V>
__attribute__((noinline, noipa)) void pfwc_vis_one() {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
    constexpr uint32_t TZ = 0 * 32 + V, OP = 1 * 32 + V, MX = 2 * 32 + V, MY = 3 * 32 + V,
                       RX = 4 * 32 + V, RY = 5 * 32 + V, RCK = 7 * 32 + V;
    // A. Predicate terms.
    {
        vFloat m1;
        {
            vFloat tz = dst_reg[TZ];
            vFloat op = dst_reg[OP];
            dst_reg[RCK] = tz + op;
            m1 = tz - vFloat(dst_reg[DR_VP + VP_KN]);
            vFloat m2 = vFloat(dst_reg[DR_VP + VP_MO]) - op;
            vFloat t = vFloat(dst_reg[RX]) - vFloat(dst_reg[DR_VP + VP_R]);
            vec_min_max(t, m2);
            t = vFloat(dst_reg[RY]) - vFloat(dst_reg[DR_VP + VP_R]);
            vec_min_max(t, m2);
            v_if(m2 > 0.0f) { m1 = 0.0f; }
            v_endif;
        }
        {
            vFloat rx = dst_reg[RX];
            vFloat mx = dst_reg[MX];
            vFloat sx = mx + rx;
            vFloat dx = mx - rx;
            dst_reg[RX] = sx;
            dst_reg[MX] = dx;
            dst_reg[RCK] = vFloat(dst_reg[RCK]) + sx;
            vec_min_max(m1, rx);
            vec_min_max(m1, sx);
            vFloat t = vFloat(dst_reg[DR_VP + VP_W]) - dx;
            vec_min_max(m1, t);
        }
        {
            vFloat ry = dst_reg[RY];
            vFloat my = dst_reg[MY];
            vFloat sy = my + ry;
            vFloat dy = my - ry;
            dst_reg[RY] = sy;
            dst_reg[MY] = dy;
            dst_reg[RCK] = vFloat(dst_reg[RCK]) + sy;
            vec_min_max(m1, ry);
            vec_min_max(m1, sy);
            vFloat t = vFloat(dst_reg[DR_VP + VP_H]) - dy;
            vec_min_max(m1, t);
        }
        dst_reg[OP] = m1;
    }
    // B. Cells and the edge-tau term per axis.
    {
        vFloat x0 = pfwc_vis_cell(vFloat(dst_reg[MX]) * vFloat(dst_reg[DR_VP + VP_INV]),
                                  vFloat(dst_reg[DR_VP + VP_TX1]));
        dst_reg[TZ] = x0;
        vFloat q0 = vFloat(dst_reg[RX]) * vFloat(dst_reg[DR_VP + VP_INV]);
        vFloat x1 = pfwc_vis_cell(q0, vFloat(dst_reg[DR_VP + VP_TX1]));
        dst_reg[MX] = x1 - x0;
        vFloat tau = dst_reg[DR_VP + VP_TAU];
        vFloat d = x1 + 1.0f - q0;
        vFloat e = q0 - x1 - tau;
        vFloat e2 = d - tau;
        d = -d;
        vec_min_max(d, e2);  // e2 = max(d - tau, -d)
        vec_min_max(e, e2);  // e = min(e, e2)
        dst_reg[RX] = e;
    }
    {
        vFloat y0 = pfwc_vis_cell(vFloat(dst_reg[MY]) * vFloat(dst_reg[DR_VP + VP_INV]),
                                  vFloat(dst_reg[DR_VP + VP_TY1]));
        vFloat q0 = vFloat(dst_reg[RY]) * vFloat(dst_reg[DR_VP + VP_INV]);
        dst_reg[RY] = y0;
        vFloat y1 = pfwc_vis_cell(q0, vFloat(dst_reg[DR_VP + VP_TY1]));
        dst_reg[MY] = y1 - y0;
        vFloat tau = dst_reg[DR_VP + VP_TAU];
        vFloat d = y1 + 1.0f - q0;
        vFloat e = q0 - y1 - tau;
        vFloat e2 = d - tau;
        d = -d;
        vec_min_max(d, e2);
        vec_min_max(e, e2);
        vFloat ex = dst_reg[RX];
        vec_min_max(e, ex);
        v_if(vFloat(dst_reg[OP]) > 0.0f && e <= 0.0f) {
            dst_reg[RCK] = Converter::as_float(0x7F800000u);
        }
        v_endif;
    }
    // C. Words (as below).
    {
        vFloat low = vFloat(dst_reg[TZ]) + vFloat(dst_reg[RY]) * 1024.0f;  // x0 | y0 << 10
        vInt aw = reinterpret<vInt>(low + 8388608.0f) - vInt(0x0B000000);
        vInt wb = reinterpret<vInt>(vFloat(dst_reg[MX]) + 8388608.0f) - vInt(0x4B000000);
        aw = aw + reinterpret<vInt>(reinterpret<vUInt>(wb) << 20);
        vFloat tpg = (vFloat(dst_reg[MX]) + 1.0f) * (vFloat(dst_reg[MY]) + 1.0f);
        vFloat tw = reinterpret<vFloat>(reinterpret<vInt>(tpg + 8388608.0f) - vInt(0x0B000000));
        v_if(vFloat(dst_reg[OP]) <= 0.0f) { tw = 0.0f; }
        v_endif;
        v_if(exexp(vFloat(dst_reg[RCK])) >= 128) { tw = Converter::as_float(0x20000000u); }
        v_endif;
        dst_reg[OP] = reinterpret<vFloat>(aw);  // output slot 1: aabb word
        dst_reg[TZ] = tw;                       // output slot 0: tpg word
    }
}
#else
template <uint32_t V>
__attribute__((noinline, noipa)) void pfwc_vis_one() {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
    constexpr uint32_t TZ = 0 * 32 + V, OP = 1 * 32 + V, MX = 2 * 32 + V, MY = 3 * 32 + V,
                       RX = 4 * 32 + V, RY = 5 * 32 + V, RCK = 7 * 32 + V;
    // 1. Recheck flag: exexp + 128 = biased exponent + 1 in [1, 256], 256 (bit 8)
    //    only for inf/NaN; OR over the six inputs.
    {
        vInt nf = exexp(vFloat(dst_reg[TZ])) + 128;
        nf = nf | (exexp(vFloat(dst_reg[OP])) + 128);
        nf = nf | (exexp(vFloat(dst_reg[MX])) + 128);
        nf = nf | (exexp(vFloat(dst_reg[MY])) + 128);
        nf = nf | (exexp(vFloat(dst_reg[RX])) + 128);
        nf = nf | (exexp(vFloat(dst_reg[RY])) + 128);
        dst_reg[RCK] = 0.0f;
        v_if(nf >= 256) { dst_reg[RCK] = 1.0f; }
        v_endif;
    }
    // 2. gather_pred::visible_bits, one failed test at a time.
    vFloat fail = 0.0f;
    v_if(vFloat(dst_reg[TZ]) - vFloat(dst_reg[DR_VP + VP_KN]) <= 0.0f) { fail = 1.0f; }  // tz <= k_near
    v_endif;
    v_if(vFloat(dst_reg[DR_VP + VP_MO]) - vFloat(dst_reg[OP]) > 0.0f) { fail = 1.0f; }  // op < min_op
    v_endif;
    // x: rx tests, sx = fl(mx + rx), dx = fl(mx - rx), their tests, then x0 / x1.
    {
        vFloat rx = dst_reg[RX];
        v_if(rx <= 0.0f) { fail = 1.0f; }
        v_endif;
        v_if(rx - vFloat(dst_reg[DR_VP + VP_R]) > 0.0f) { fail = 1.0f; }  // rx > max_radius
        v_endif;
        vFloat mx = dst_reg[MX];
        vFloat sx = mx + rx;
        vFloat dx = mx - rx;
        v_if(sx <= 0.0f) { fail = 1.0f; }  // !(-rx < mx)
        v_endif;
        v_if(vFloat(dst_reg[DR_VP + VP_W]) - dx <= 0.0f) { fail = 1.0f; }  // !(fl(mx - rx) < W)
        v_endif;
        dst_reg[RX] = sx;
        dst_reg[TZ] = pfwc_vis_cell(dx * vFloat(dst_reg[DR_VP + VP_INV]),
                                    vFloat(dst_reg[DR_VP + VP_TX1]));  // x0
    }
    {
        vFloat q0 = vFloat(dst_reg[RX]) * vFloat(dst_reg[DR_VP + VP_INV]);
        vFloat x1 = pfwc_vis_cell(q0, vFloat(dst_reg[DR_VP + VP_TX1]));
        v_if(fail <= 0.0f) {
            v_if(q0 - x1 <= vFloat(dst_reg[DR_VP + VP_TAU])) { dst_reg[RCK] = 1.0f; }
            v_endif;
            vFloat d = x1 + 1.0f - q0;  // < 0 when clamped at the far edge: no flag
            v_if(d >= 0.0f) {
                v_if(d <= vFloat(dst_reg[DR_VP + VP_TAU])) { dst_reg[RCK] = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
        dst_reg[MX] = x1 - vFloat(dst_reg[TZ]);  // w - 1
    }
    // y: same with ry / my.
    {
        vFloat ry = dst_reg[RY];
        v_if(ry <= 0.0f) { fail = 1.0f; }
        v_endif;
        v_if(ry - vFloat(dst_reg[DR_VP + VP_R]) > 0.0f) { fail = 1.0f; }
        v_endif;
        vFloat my = dst_reg[MY];
        vFloat sy = my + ry;
        vFloat dy = my - ry;
        v_if(sy <= 0.0f) { fail = 1.0f; }
        v_endif;
        v_if(vFloat(dst_reg[DR_VP + VP_H]) - dy <= 0.0f) { fail = 1.0f; }
        v_endif;
        dst_reg[RY] = sy;
        dst_reg[OP] = pfwc_vis_cell(dy * vFloat(dst_reg[DR_VP + VP_INV]),
                                    vFloat(dst_reg[DR_VP + VP_TY1]));  // y0
    }
    {
        vFloat q0 = vFloat(dst_reg[RY]) * vFloat(dst_reg[DR_VP + VP_INV]);
        vFloat y1 = pfwc_vis_cell(q0, vFloat(dst_reg[DR_VP + VP_TY1]));
        v_if(fail <= 0.0f) {
            v_if(q0 - y1 <= vFloat(dst_reg[DR_VP + VP_TAU])) { dst_reg[RCK] = 1.0f; }
            v_endif;
            vFloat d = y1 + 1.0f - q0;  // < 0 when clamped at the far edge: no flag
            v_if(d >= 0.0f) {
                v_if(d <= vFloat(dst_reg[DR_VP + VP_TAU])) { dst_reg[RCK] = 1.0f; }
                v_endif;
            }
            v_endif;
        }
        v_endif;
        dst_reg[MY] = y1 - vFloat(dst_reg[OP]);  // h - 1
    }
    // 3. Words. bits(x + 2^23) == 0x4B000000 + x for an integer x in [0, 2^23);
    //    subtracting 0x0B000000 instead leaves TAG (0x40000000) | x.
    {
        vFloat low = vFloat(dst_reg[TZ]) + vFloat(dst_reg[OP]) * 1024.0f;  // x0 | y0 << 10
        vInt aw = reinterpret<vInt>(low + 8388608.0f) - vInt(0x0B000000);
        vInt wb = reinterpret<vInt>(vFloat(dst_reg[MX]) + 8388608.0f) - vInt(0x4B000000);
        aw = aw + reinterpret<vInt>(reinterpret<vUInt>(wb) << 20);
        dst_reg[OP] = reinterpret<vFloat>(aw);  // output slot 1: aabb word
    }
    {
        vFloat tpg = (vFloat(dst_reg[MX]) + 1.0f) * (vFloat(dst_reg[MY]) + 1.0f);
        vFloat tw = reinterpret<vFloat>(reinterpret<vInt>(tpg + 8388608.0f) - vInt(0x0B000000));
        v_if(fail > 0.0f) { tw = 0.0f; }
        v_endif;
        v_if(vFloat(dst_reg[RCK]) > 0.0f) { tw = Converter::as_float(0x20000000u); }  // RECHECK
        v_endif;
        dst_reg[TZ] = tw;  // output slot 0: tpg word
    }
}
#endif  // PFWC_VIS_FAST
#ifdef PFWC_PRECULL
// Lever C (task #140). Runs in step 11.5's DEST acquire, before the TZ / MX /
// MY loads (task #142: a separate step with its own radii repack put the
// program over the 70.6 KB kernel config buffer). DEST slots: 0 cov b, 1
// opacity, 2 cov a, 3 cov c, 4 rx, 5 ry (1, 4 and 5 are 11.5's); rx / ry are replaced in place. One loop over the 32 vectors (not
// 32 instantiations, task #102). Intermediate (8 registers, no spill): t ->
// slot 6.
constexpr uint32_t PC_B = 0 * 32, PC_OP = 1 * 32, PC_A = 2 * 32, PC_C = 3 * 32, PC_RX = 4 * 32,
                   PC_RY = 5 * 32, PC_T = 6 * 32;

// sqrt(x), x >= 0, ~23-bit (microblock_band_cull_compute.cpp band_sqrt).
sfpi_inline sfpi::vFloat precull_sqrt(sfpi::vFloat x) {
    using namespace sfpi;
    vInt i = reinterpret<vInt>(reinterpret<vUInt>(x) >> 1);
    vFloat y = reinterpret<vFloat>(vInt(0x5f1110a0) - i);
    vFloat xy = x * y;
    vFloat c = (-y) * xy;
    y = y * (vFloat(2.2825186f) + c * (vFloat(2.2533049f) + c));
    xy = x * y;
    vFloat one_minus_xyy = vFloat(1.0f) - y * xy;
    return one_minus_xyy * (xy * 0.5f) + xy;
}

// r = an integer >= sqrt(t * cov) + 1 (the 1 px slack covers the sqrt and the
// band cull's fp32 conic error): q = sqrt(t * cov) + 2 rounded to an integer
// by the 2^23 trick, any faithful rounding (no ceil fix-up: task #142 code
// size; at most 1 px more than the ceil). PRECULL_PC: r = max(q - 3/8, 1/8)
// with q = sqrt(t * cov) (pixel-centre rect, see the top). r replaces the
// radius slot when r < radius (the caller's v_if holds the shrink-ok lanes).
sfpi_inline void precull_axis(uint32_t cov_slot, uint32_t r_slot) {
    using namespace sfpi;
    vFloat q = precull_sqrt(vFloat(dst_reg[PC_T]) * vFloat(dst_reg[cov_slot]));
#ifdef PRECULL_PC
    vFloat lo = 0.125f;
    vFloat r = q - 0.375f;
    vec_min_max(lo, r);  // r = max(r, 1/8)
#else
    vFloat r = (q + 8388610.0f) - 8388608.0f;
#endif
    v_if(r < vFloat(dst_reg[r_slot])) { dst_reg[r_slot] = r; }
    v_endif;
}

// 2 ln 2 / 2^14: t's slope in the top 23 bits of the opacity (step 1).
constexpr float PC_K = 1.3862944f / 16384.0f;

__attribute__((noinline)) void pfwc_precull_tile(uint32_t c0_bits, uint32_t rlim_bits) {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        // 1. t = max(2 ln(op) + c0, 0) with an upper bound of log2(op) (no SFPU
        //    log: task #142 needed the code size): for op = 2^e m, m in [1, 2),
        //    log2(op) <= e + (m - 1) + 0.0860713, and (m - 1) is read from the
        //    bits with 9 low bits truncated (+ 2^-14). Too large a t only keeps
        //    more records. The host folds the constants into arg 65; op = 0
        //    gives t < 0 -> 0.
        {
            vInt i = reinterpret<vInt>(reinterpret<vUInt>(vFloat(dst_reg[PC_OP])) >> 9) |
                     vInt(0x4B000000);
            vFloat t = (reinterpret<vFloat>(i) - 8388608.0f) * PC_K + Converter::as_float(c0_bits);
            vFloat z = 0.0f;
            vec_min_max(z, t);  // t = max(t, 0)
            dst_reg[PC_T] = t;
        }
        // 2. Shrink only when a, b, c are finite, 64 (a c - b^2) >= a c and
        //    rx, ry <= rlim (nested v_ifs: no flag slot).
        vInt nf = exexp(vFloat(dst_reg[PC_A])) + 128;
        nf = nf | (exexp(vFloat(dst_reg[PC_B])) + 128);
        nf = nf | (exexp(vFloat(dst_reg[PC_C])) + 128);
        vFloat ac = vFloat(dst_reg[PC_A]) * vFloat(dst_reg[PC_C]);
        vFloat b = dst_reg[PC_B];
        vFloat cond = (ac - b * b) * 64.0f - ac;
        vFloat rlo = dst_reg[PC_RX];
        vFloat rhi = dst_reg[PC_RY];
        vec_min_max(rlo, rhi);  // rhi = max(rx, ry)
        v_if(nf < 256) {
            v_if(cond >= 0.0f) {
                v_if(rhi - Converter::as_float(rlim_bits) <= 0.0f) {
                    precull_axis(PC_A, PC_RX);
                    precull_axis(PC_C, PC_RY);
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
        dst_reg++;
    }
}
#endif  // PFWC_PRECULL
#endif  // TRISC_MATH

template <uint32_t V>
inline void pfwc_vis_unroll() {
    if constexpr (V < 16) {
        MATH((pfwc_vis_one<V>()));
        pfwc_vis_unroll<V + 1>();
    }
}
#endif  // PFWC_VIS

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("pfwc");  // Tracy stage label (fused project+pfwc compute)
#ifdef PFWC_SC_ON
    const uint32_t pc_w0 = pc_now();
#endif
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    // R (row-major) + t for the fused world→camera transform.
    uint32_t r_bits[9];
    for (uint32_t k = 0; k < 9; k++) r_bits[k] = get_arg_val<uint32_t>(1 + k);
    const uint32_t t0 = get_arg_val<uint32_t>(10);
    const uint32_t t1 = get_arg_val<uint32_t>(11);
    const uint32_t t2 = get_arg_val<uint32_t>(12);
    const uint32_t t_bits[3] = {t0, t1, t2};

    const uint32_t fx       = get_arg_val<uint32_t>(13);
    const uint32_t fy       = get_arg_val<uint32_t>(14);
    const uint32_t cx       = get_arg_val<uint32_t>(15);
    const uint32_t cy       = get_arg_val<uint32_t>(16);
    const uint32_t k_bits   = get_arg_val<uint32_t>(53);  // radii scale k = 3.0
    const uint32_t neg_fx_bits = get_arg_val<uint32_t>(54);
    const uint32_t neg_fy_bits = get_arg_val<uint32_t>(55);
    constexpr uint32_t two_fp32_bits = 0x40000000U;
    constexpr uint32_t pt3_fp32_bits = 0x3E99999AU;  // 0.3f
#ifdef PFWC_VIS
    uint32_t vis_bits[VIS_PARAMS];
    for (uint32_t k = 0; k < VIS_PARAMS; k++) vis_bits[k] = get_arg_val<uint32_t>(56 + k);
#endif
#ifdef PFWC_PRECULL
    const uint32_t precull_c0_bits = get_arg_val<uint32_t>(65);
    const uint32_t precull_rlim_bits = get_arg_val<uint32_t>(66);
#endif

    init_sfpu(CB_MX, CB_M2X);
    add_binary_tile_init();
    mul_binary_tile_init();
    recip_tile_init();
    sqrt_tile_init();
    relu_tile_init();
    rounding_op_tile_init();

    if (num_chunks == 0) {
        return;
    }
#ifdef PFWC_SC_ON
    for (uint32_t i = 0; i < 3 + PC_N + 5; i++) g_v[i] = 0;
    g_pc_t = pc_now();
    const uint32_t pc_init = g_pc_t - pc_w0;
#endif

    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        cb_wait_front(CB_MX, 1);
        cb_wait_front(CB_MY, 1);
        cb_wait_front(CB_MZ, 1);
        cb_wait_front(CB_C00, 1);
        cb_wait_front(CB_C01, 1);
        cb_wait_front(CB_C02, 1);
        cb_wait_front(CB_C11, 1);
        cb_wait_front(CB_C12, 1);
        cb_wait_front(CB_C22, 1);
        PC_MARK(0);

#ifdef PFWC_FUSE_PROJ
        // ── 1-5 (task #489, GSPLAT_TT_PFWC_FUSE bit 0): one acquire. 3 copies,
        //      R·means + t, 1/tz, mean_x / mean_y in DEST (pfwc_fuse_sfpu.h), then
        //      the same 9 packs. Bit-identical to the seven acquires below.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MX);
            copy_tile(CB_MX, 0, 0);
            copy_tile(CB_MY, 0, 1);
            copy_tile(CB_MZ, 0, 2);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::xform<3, 3>(r_bits[0], r_bits[1], r_bits[2], t_bits[0])));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::xform<4, 4>(r_bits[3], r_bits[4], r_bits[5], t_bits[1])));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::xform<5, 6>(r_bits[6], r_bits[7], r_bits[8], t_bits[2])));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
#ifdef PFWC_RECIP_NEWTON
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_inv_tz_nr<6>()));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
#else
            recip_tile(6);
#endif
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::run_mean(fx, cx, fy, cy)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(3, CB_TMP_TX);
            emit_scratch(4, CB_TMP_TY);
            emit_scratch(5, CB_TMP_TZ);
            emit_scratch(6, CB_TMP_INV_TZ);
            emit_dst(5, OCB(CB_DEP));
            emit_dst(0, OCB(CB_M2X));
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_MX);
#endif
            emit_dst(1, OCB(CB_M2Y));
#ifdef PFWC_VIS
            emit_scratch(1, CB_TMP_MY);
#endif
            tile_regs_release();
        }
        cb_pop_front(CB_MX, 1);
        cb_pop_front(CB_MY, 1);
        cb_pop_front(CB_MZ, 1);
        PC_MARK(1);
        PC_MARK(2);
        PC_MARK(3);
#else
        // ── 1. FUSED transform: tx/ty/tz = R · means + t  (means_cam folded
        //      with the pfwc translation; L1 bridge into the pfwc body below).
        //      Per j the sum mx*r_j0 + my*r_j1 + mz*r_j2 is computed by the SAME
        //      SFPU op sequence the standalone means_cam used, in an fp32 DEST
        //      slot, then add_unary(t_j) — bit-identical to the old fp32-DRAM
        //      handoff + pfwc translate.
        for (uint32_t j = 0; j < 3; j++) {
            const uint32_t r_j0 = r_bits[j * 3 + 0];
            const uint32_t r_j1 = r_bits[j * 3 + 1];
            const uint32_t r_j2 = r_bits[j * 3 + 2];
            const uint32_t cb_out = CB_TMP_TX + j;

            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_MX);
            copy_tile(CB_MX, 0, 0);
            copy_tile(CB_MY, 0, 1);
            copy_tile(CB_MZ, 0, 2);
            mul_unary_tile(0, r_j0);
            mul_unary_tile(1, r_j1);
            mul_unary_tile(2, r_j2);
            add_binary_tile(0, 1, 0);
            add_binary_tile(0, 2, 0);
            add_unary_tile(0, t_bits[j]);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, cb_out);
            tile_regs_release();
        }

        // Means consumed — drain so the reader can refill for the next chunk.
        cb_pop_front(CB_MX, 1);
        cb_pop_front(CB_MY, 1);
        cb_pop_front(CB_MZ, 1);

        PC_MARK(1);
        // ── 2. inv_tz = 1/tz → scratch
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
#ifdef PFWC_RECIP_NEWTON
            // Task #266: SFPARECIP seed + 2 Newton steps (pfwc_recip_nr.h), ~1 ulp.
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_inv_tz_nr()));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
#else
            recip_tile(0);
#endif
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_INV_TZ);
            tile_regs_release();
        }

        PC_MARK(2);
        // ── 3. depth = tz → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_DEP));
            tile_regs_release();
        }

        PC_MARK(3);
        // ── 4. mean_x = fx · tx · inv_tz + cx → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 0);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_binary_tile(0, 1, 0);   // dst[0] = tx · inv_tz
            mul_unary_tile(0, fx);
            add_unary_tile(0, cx);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_M2X));
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_MX);
#endif
            tile_regs_release();
        }

        // ── 5. mean_y = fy · ty · inv_tz + cy → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 0);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_binary_tile(0, 1, 0);
            mul_unary_tile(0, fy);
            add_unary_tile(0, cy);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_M2Y));
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_MY);
#endif
            tile_regs_release();
        }

#endif  // PFWC_FUSE_PROJ
        PC_MARK(4);
#ifdef PFWC_FUSE_CCAC
        // ── 6-7 (task #489, GSPLAT_TT_PFWC_FUSE bit 1): cov_cam and S_AC in one
        //      acquire. cc00 / cc11 stay in DEST (S_AC was their only reader); the
        //      other four cc entries, a, c and the radii are packed as before.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(COV3D_CB[0]);
            for (uint32_t k = 0; k < 6; k++) copy_tile(COV3D_CB[k], 0, k);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((covcam_sfpu_math()));    // cc00..cc22 -> tiles 0..5
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 6);
            copy_tile(CB_TMP_TX, 0, 7);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::run_a(fx, neg_fx_bits)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            copy_tile(CB_TMP_TY, 0, 7);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_fuse::run_c(fy, neg_fy_bits)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            relu_tile(6);
            sqrt_tile(6);
            mul_unary_tile(6, k_bits);
            ceil_tile(6);
            relu_tile(7);
            sqrt_tile(7);
            mul_unary_tile(7, k_bits);
            ceil_tile(7);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(1, CB_TMP_CC01);
            emit_scratch(2, CB_TMP_CC02);
            emit_scratch(4, CB_TMP_CC12);
            emit_scratch(5, CB_TMP_CC22);
            emit_scratch(0, CB_TMP_A);
            emit_scratch(3, CB_TMP_C);
            emit_dst(6, OCB(CB_RX));
#ifdef PFWC_VIS
            emit_scratch(6, CB_TMP_RX);
#endif
            emit_dst(7, OCB(CB_RY));
#ifdef PFWC_VIS
            emit_scratch(7, CB_TMP_RY);
#endif
            tile_regs_release();
        }
        cb_pop_front(CB_C00, 1);
        cb_pop_front(CB_C01, 1);
        cb_pop_front(CB_C02, 1);
        cb_pop_front(CB_C11, 1);
        cb_pop_front(CB_C12, 1);
        cb_pop_front(CB_C22, 1);
        PC_MARK(5);
#else
        // ── 6. cov_cam (6 unique entries) → scratch CBs
#ifdef PFWC_COVCAM_SFPU
        covcam_sfpu_to_scratch();
#else
        for (uint32_t e = 0; e < 6; e++) {
            compute_cc_entry_to_scratch(17 + e * 6, CC_SCRATCH[e]);
        }
#endif

        // Drain cov3d inputs — done feeding cov_cam.
        cb_pop_front(CB_C00, 1);
        cb_pop_front(CB_C01, 1);
        cb_pop_front(CB_C02, 1);
        cb_pop_front(CB_C11, 1);
        cb_pop_front(CB_C12, 1);
        cb_pop_front(CB_C22, 1);

        PC_MARK(5);
#endif  // PFWC_FUSE_CCAC
#ifdef PFWC_COV2D_SFPU
#ifndef PFWC_FUSE_CCAC
        // ── 7-11 (task #228, P2): two acquires instead of six. S_AC: a, c and
        //      the radii (pfwc_cov2d::run_ac, then the same relu / sqrt / k /
        //      ceil tile ops on copies of a and c in slots 6 / 7).
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_CC00);  // all Float32: one init
            copy_tile(CB_TMP_CC00, 0, 0);
            copy_tile(CB_TMP_CC02, 0, 1);
            copy_tile(CB_TMP_CC22, 0, 2);
            copy_tile(CB_TMP_CC11, 0, 3);
            copy_tile(CB_TMP_CC12, 0, 4);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            copy_tile(CB_TMP_TX, 0, 6);
            copy_tile(CB_TMP_TY, 0, 7);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_cov2d::run_ac(fx, neg_fx_bits, fy, neg_fy_bits)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            relu_tile(6);
            sqrt_tile(6);
            mul_unary_tile(6, k_bits);
            ceil_tile(6);
            relu_tile(7);
            sqrt_tile(7);
            mul_unary_tile(7, k_bits);
            ceil_tile(7);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_A);
            emit_scratch(3, CB_TMP_C);
            emit_dst(6, OCB(CB_RX));
#ifdef PFWC_VIS
            emit_scratch(6, CB_TMP_RX);
#endif
            emit_dst(7, OCB(CB_RY));
#ifdef PFWC_VIS
            emit_scratch(7, CB_TMP_RY);
#endif
            tile_regs_release();
        }
#endif  // !PFWC_FUSE_CCAC

        PC_MARK(6);
        // S_BC: b (pfwc_cov2d::run_b), then the conic fold of a, b, c.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 0);
            copy_tile(CB_TMP_CC01, 0, 1);
            copy_tile(CB_TMP_CC12, 0, 2);
            copy_tile(CB_TMP_CC22, 0, 3);
            copy_tile(CB_TMP_INV_TZ, 0, 4);
            copy_tile(CB_TMP_TX, 0, 5);
            copy_tile(CB_TMP_TY, 0, 6);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_cov2d::run_b(fx, neg_fx_bits, fy, neg_fy_bits)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            copy_tile(CB_TMP_A, 0, 0);
            copy_tile(CB_TMP_C, 0, 2);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            pfwc_conic_unroll<0>();
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_A));
            emit_dst(1, OCB(CB_B));
            emit_dst(2, OCB(CB_C));
            emit_scratch(7, CB_TMP_B);
            tile_regs_release();
        }
        // Steps 8-11 have no work of their own here (STEPCYC books 0).
        PC_MARK(7);
        PC_MARK(8);
        PC_MARK(9);
        PC_MARK(10);
        PC_MARK(11);
#else
        // ── 7. cov2d_a = j00²·cc00 + 2·j00·j02·cc02 + j02²·cc22 + 0.3
        {
            tile_regs_acquire();

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00 = fx · inv_tz

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);    // dst[5] = inv_tz scratch
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz
            mul_binary_tile(2, 5, 2);          // dst[2] = -fx·tx·inv_tz² = j02

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11 = fy · inv_tz

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC00);
            copy_tile(CB_TMP_CC00, 0, 0);      // dst[0] = cc00
            mul_binary_tile(0, 1, 0);          // dst[0] = cc00 · j00
            mul_binary_tile(0, 1, 0);          // dst[0] = cc00 · j00²

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);          // dst[5] = cc02 · j00
            mul_binary_tile(5, 2, 5);          // dst[5] = cc02 · j00 · j02
            mul_unary_tile(5, two_fp32_bits);  // dst[5] = 2·cc02·j00·j02
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);          // dst[5] = cc22 · j02
            mul_binary_tile(5, 2, 5);          // dst[5] = cc22 · j02²
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);  // dst[0] = a

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_A);
            tile_regs_release();
        }

        PC_MARK(6);
        // ── 8. cov2d_b = j00·j11·cc01 + j00·j12·cc02 + j02·j11·cc12 + j02·j12·cc22
        {
            tile_regs_acquire();

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);    // dst[5] = inv_tz
            mul_binary_tile(2, 5, 2);
            mul_binary_tile(2, 5, 2);          // dst[2] = j02

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC01);
            copy_tile(CB_TMP_CC01, 0, 0);      // dst[0] = cc01
            mul_binary_tile(0, 1, 0);          // dst[0] *= j00
            mul_binary_tile(0, 3, 0);          // dst[0] *= j11

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);          // dst[5] = cc02·j00
            mul_binary_tile(5, 4, 5);          // dst[5] *= j12
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 3, 5);          // *= j11
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);          // *= j02
            mul_binary_tile(5, 4, 5);          // *= j12
            add_binary_tile(0, 5, 0);

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_B);
            tile_regs_release();
        }

        PC_MARK(7);
        // ── 9. cov2d_c = j11²·cc11 + 2·j11·j12·cc12 + j12²·cc22 + 0.3
        {
            tile_regs_acquire();

            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC11);
            copy_tile(CB_TMP_CC11, 0, 0);
            mul_binary_tile(0, 3, 0);
            mul_binary_tile(0, 3, 0);          // dst[0] = j11²·cc11

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 3, 5);
            mul_binary_tile(5, 4, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);          // dst[0] += 2·j11·j12·cc12

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 4, 5);
            mul_binary_tile(5, 4, 5);
            add_binary_tile(0, 5, 0);          // dst[0] += j12²·cc22

            add_unary_tile(0, pt3_fp32_bits);

            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_C);
            tile_regs_release();
        }

        PC_MARK(8);
        // ── 9.5 (A1). Conic fold: A,B,C from scratch a,b,c → cov2d outputs.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_A);
            copy_tile(CB_TMP_A, 0, 0);
            copy_tile_to_dst_init_short(CB_TMP_B);
            copy_tile(CB_TMP_B, 0, 1);
            copy_tile_to_dst_init_short(CB_TMP_C);
            copy_tile(CB_TMP_C, 0, 2);

            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            pfwc_conic_unroll<0>();
            MATH((_llk_math_eltwise_unary_sfpu_done_()));

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_A));
            emit_dst(1, OCB(CB_B));
            emit_dst(2, OCB(CB_C));
            tile_regs_release();
        }

        PC_MARK(9);
        // ── 10. radii.x = ceil(k · sqrt(max(a, 0))) — recompute a.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 1);
            mul_unary_tile(1, fx);             // dst[1] = j00

            copy_tile_to_dst_init_short(CB_TMP_TX);
            copy_tile(CB_TMP_TX, 0, 2);
            mul_unary_tile(2, neg_fx_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(2, 5, 2);
            mul_binary_tile(2, 5, 2);          // dst[2] = j02

            copy_tile_to_dst_init_short(CB_TMP_CC00);
            copy_tile(CB_TMP_CC00, 0, 0);
            mul_binary_tile(0, 1, 0);
            mul_binary_tile(0, 1, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC02);
            copy_tile(CB_TMP_CC02, 0, 5);
            mul_binary_tile(5, 1, 5);
            mul_binary_tile(5, 2, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 2, 5);
            mul_binary_tile(5, 2, 5);
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);  // dst[0] = a

            relu_tile(0);                       // dst[0] = max(a, 0)
            sqrt_tile(0);
            mul_unary_tile(0, k_bits);
            ceil_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_RX));
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_RX);
#endif
            tile_regs_release();
        }

        PC_MARK(10);
        // ── 11. radii.y = ceil(k · sqrt(max(c, 0))) — recompute c.
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 3);
            mul_unary_tile(3, fy);             // dst[3] = j11

            copy_tile_to_dst_init_short(CB_TMP_TY);
            copy_tile(CB_TMP_TY, 0, 4);
            mul_unary_tile(4, neg_fy_bits);
            copy_tile_to_dst_init_short(CB_TMP_INV_TZ);
            copy_tile(CB_TMP_INV_TZ, 0, 5);
            mul_binary_tile(4, 5, 4);
            mul_binary_tile(4, 5, 4);          // dst[4] = j12

            copy_tile_to_dst_init_short(CB_TMP_CC11);
            copy_tile(CB_TMP_CC11, 0, 0);
            mul_binary_tile(0, 3, 0);
            mul_binary_tile(0, 3, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC12);
            copy_tile(CB_TMP_CC12, 0, 5);
            mul_binary_tile(5, 3, 5);
            mul_binary_tile(5, 4, 5);
            mul_unary_tile(5, two_fp32_bits);
            add_binary_tile(0, 5, 0);

            copy_tile_to_dst_init_short(CB_TMP_CC22);
            copy_tile(CB_TMP_CC22, 0, 5);
            mul_binary_tile(5, 4, 5);
            mul_binary_tile(5, 4, 5);
            add_binary_tile(0, 5, 0);

            add_unary_tile(0, pt3_fp32_bits);

            relu_tile(0);
            sqrt_tile(0);
            mul_unary_tile(0, k_bits);
            ceil_tile(0);

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_RY));
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_RY);
#endif
            tile_regs_release();
        }

        PC_MARK(11);
#endif  // PFWC_COV2D_SFPU
#ifdef PFWC_VIS
        // ── 11.5 (task #99). Visibility predicate + tile rectangle on the SFPU.
        {
            cb_wait_front(CB_OP, 1);
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);  // all Float32: one init
#ifdef PFWC_PRECULL
            // 11.6 (task #140, lever C): opacity-aware radii for the rectangle,
            // in place in slots 4 / 5 (pfwc_precull_tile), before TZ / MX / MY
            // overwrite slots 0 / 2 / 3. CB_TMP_A/B/C are popped in step 12.
            copy_tile(CB_TMP_B, 0, 0);
            copy_tile(CB_OP, 0, 1);
            copy_tile(CB_TMP_A, 0, 2);
            copy_tile(CB_TMP_C, 0, 3);
            copy_tile(CB_TMP_RX, 0, 4);
            copy_tile(CB_TMP_RY, 0, 5);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_precull_tile(precull_c0_bits, precull_rlim_bits)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            copy_tile(CB_TMP_TZ, 0, 0);
            copy_tile(CB_TMP_MX, 0, 2);
            copy_tile(CB_TMP_MY, 0, 3);
#else
            copy_tile(CB_TMP_TZ, 0, 0);
            copy_tile(CB_OP, 0, 1);
            copy_tile(CB_TMP_MX, 0, 2);
            copy_tile(CB_TMP_MY, 0, 3);
            copy_tile(CB_TMP_RX, 0, 4);
            copy_tile(CB_TMP_RY, 0, 5);
#endif

            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((pfwc_vis_stage(vis_bits)));
            pfwc_vis_unroll<0>();
            MATH((pfwc_vis_half()));
            pfwc_vis_unroll<0>();
            MATH((_llk_math_eltwise_unary_sfpu_done_()));

            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, OCB(CB_TPG));
            emit_dst(1, OCB(CB_AABB));
            tile_regs_release();
            cb_pop_front(CB_OP, 1);
            cb_pop_front(CB_TMP_MX, 1);
            cb_pop_front(CB_TMP_MY, 1);
            cb_pop_front(CB_TMP_RX, 1);
            cb_pop_front(CB_TMP_RY, 1);
        }
#endif

        // ── 12. Drain scratch CBs.
        cb_pop_front(CB_TMP_TX, 1);
        cb_pop_front(CB_TMP_TY, 1);
        cb_pop_front(CB_TMP_TZ, 1);
        cb_pop_front(CB_TMP_INV_TZ, 1);
#ifndef PFWC_FUSE_CCAC
        cb_pop_front(CB_TMP_CC00, 1);
#endif
        cb_pop_front(CB_TMP_CC01, 1);
        cb_pop_front(CB_TMP_CC02, 1);
#ifndef PFWC_FUSE_CCAC
        cb_pop_front(CB_TMP_CC11, 1);
#endif
        cb_pop_front(CB_TMP_CC12, 1);
        cb_pop_front(CB_TMP_CC22, 1);
        cb_pop_front(CB_TMP_A, 1);
        cb_pop_front(CB_TMP_B, 1);
        cb_pop_front(CB_TMP_C, 1);
        PC_MARK(12);
    }
#ifdef PFWC_SC_ON
    {
        g_v[0] = num_chunks;
        g_v[1] = pc_now() - pc_w0;
        g_v[2] = pc_init;
        // Profiler builds: one timestamped-data marker per value, (index << 32) | value.
        for (uint32_t i = 0; i < PC_NV; i++) DeviceTimestampedData("pfwc_pc", (uint64_t(i) << 32) | g_v[i]);
    }
#endif
}
