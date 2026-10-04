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

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#ifdef PFWC_VIS
#include "sfpu/ckernel_sfpu_converter.h"
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

constexpr uint32_t COV3D_CB[6] = {CB_C00, CB_C11, CB_C22, CB_C01, CB_C02, CB_C12};
constexpr uint32_t CC_SCRATCH[6] = {
    CB_TMP_CC00, CB_TMP_CC01, CB_TMP_CC02,
    CB_TMP_CC11, CB_TMP_CC12, CB_TMP_CC22};

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

    tile_regs_acquire();
    copy_tile_to_dst_init_short(COV3D_CB[0]);
    copy_tile(COV3D_CB[0], 0, 0);
    mul_unary_tile(0, scales[0]);
    for (uint32_t k = 1; k < 6; k++) {
        copy_tile_to_dst_init_short(COV3D_CB[k]);
        copy_tile(COV3D_CB[k], 0, 1);
        mul_unary_tile(1, scales[k]);
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    emit_scratch(0, cb_out_scratch);
    tile_regs_release();
}

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
// size; at most 1 px more than the ceil). r replaces the radius slot when
// r < radius (the caller's v_if holds the shrink-ok lanes).
sfpi_inline void precull_axis(uint32_t cov_slot, uint32_t r_slot) {
    using namespace sfpi;
    vFloat q = precull_sqrt(vFloat(dst_reg[PC_T]) * vFloat(dst_reg[cov_slot]));
    vFloat r = (q + 8388610.0f) - 8388608.0f;
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

        // ── 2. inv_tz = 1/tz → scratch
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            recip_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            emit_scratch(0, CB_TMP_INV_TZ);
            tile_regs_release();
        }

        // ── 3. depth = tz → output
        {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_TMP_TZ);
            copy_tile(CB_TMP_TZ, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            emit_dst(0, CB_DEP);
            tile_regs_release();
        }

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
            emit_dst(0, CB_M2X);
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
            emit_dst(0, CB_M2Y);
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_MY);
#endif
            tile_regs_release();
        }

        // ── 6. cov_cam (6 unique entries) → scratch CBs
        for (uint32_t e = 0; e < 6; e++) {
            compute_cc_entry_to_scratch(17 + e * 6, CC_SCRATCH[e]);
        }

        // Drain cov3d inputs — done feeding cov_cam.
        cb_pop_front(CB_C00, 1);
        cb_pop_front(CB_C01, 1);
        cb_pop_front(CB_C02, 1);
        cb_pop_front(CB_C11, 1);
        cb_pop_front(CB_C12, 1);
        cb_pop_front(CB_C22, 1);

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
            emit_dst(0, CB_A);
            emit_dst(1, CB_B);
            emit_dst(2, CB_C);
            tile_regs_release();
        }

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
            emit_dst(0, CB_RX);
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_RX);
#endif
            tile_regs_release();
        }

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
            emit_dst(0, CB_RY);
#ifdef PFWC_VIS
            emit_scratch(0, CB_TMP_RY);
#endif
            tile_regs_release();
        }

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
            emit_dst(0, CB_TPG);
            emit_dst(1, CB_AABB);
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
        cb_pop_front(CB_TMP_CC00, 1);
        cb_pop_front(CB_TMP_CC01, 1);
        cb_pop_front(CB_TMP_CC02, 1);
        cb_pop_front(CB_TMP_CC11, 1);
        cb_pop_front(CB_TMP_CC12, 1);
        cb_pop_front(CB_TMP_CC22, 1);
        cb_pop_front(CB_TMP_A, 1);
        cb_pop_front(CB_TMP_B, 1);
        cb_pop_front(CB_TMP_C, 1);
    }
}
