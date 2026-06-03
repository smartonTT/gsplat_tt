// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU microblock-cull compute kernel (GSPLAT_TT_SFPU_CULL).
//
// WHY
// ---
// The microblock mask (32 bits, one per 8x4 microblock of a 32x32 tile) was
// computed as SCALAR SOFT-FLOAT on the data-mover RISC inside
// reader_alpha_blend_mb_devcull.cpp (compute_microblock_mask). That soft-float
// constrained-min Mahalanobis dominated the entire blend (~422 ms). This kernel
// moves the whole mask computation onto the SFPU: ONE 32-lane SFPU vector
// evaluates ALL 32 microblocks of a gaussian in parallel (lane == microblock).
//
// 32-LANE MICROBLOCK MAPPING (the "key primitive")
// ------------------------------------------------
// A tile has exactly 4x8 = 32 microblocks; an SFPU vector has 32 lanes. We pack
// 32 gaussians per tile-pass: dst_reg[V] (vector V) == gaussian V of the batch,
// and the 32 lanes of vector V == the 32 microblocks of that gaussian.
//
// The box-origin "ramps" (CB_BOX_OX / CB_BOX_OY) are constant tiles built host-
// side so that, for EVERY vector V, the lane that will end up holding microblock
// m carries microblock m's tile-local box origin ((m&3)*8, (m>>2)*4). We never
// need to know the SFPU's intra-vector lane order: copy_tile (CB->DEST) and
// pack_tile (DEST->CB) are exact inverses at the CB-linear level, so a value
// loaded from CB-linear position P returns to CB-linear position P after the
// (lane-local) SFPU math. The writer (writer_microblock_cull.cpp) reads the
// per-microblock keep flag back from the SAME CB-linear positions and packs the
// 32-bit mask with pure integer compares -> no float on any data mover.
//
// Per (gaussian, microblock) lane math mirrors compute_microblock_mask:
//   det  = max(cov_a*cov_c - cov_b*cov_b, 1e-6)
//   ci_a=cov_c/det, ci_b=-cov_b/det, ci_c=cov_a/det
//   constrained-min Mahalanobis^2 of the microblock box vs the gaussian center
//   keep = (opacity * exp(-0.5 * m2_min) >= contrib_floor)   (exp avoids log)
// keep (1.0/0.0) is written to dst_reg[KEEP][V]; the unused upper vectors of a
// short final batch are zero-filled and ignored by the writer.

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#if defined(CULL_DEBUG_VALS)
#include "api/debug/dprint.h"
#endif
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

namespace {

constexpr uint32_t CB_BOX_OX     = 0;   // fp32 tile: per-lane microblock box x-origin
constexpr uint32_t CB_BOX_OY     = 1;   // fp32 tile: per-lane microblock box y-origin
constexpr uint32_t CB_CULL_COEFF = 2;   // 64B row per gaussian: cov_a,cov_b,cov_c,mx,my,op
constexpr uint32_t CB_CULL_COUNTS= 3;   // per-tile [L, tx_pix, ty_pix]
constexpr uint32_t CB_KEEP       = 16;  // fp32 keep tile (one per 32-gaussian batch)

#ifdef FUSE_BLEND
// §8.4 L1 mask handoff: the SAME compute kernel runs the blend SFPU after the
// cull SFPU, per tile (cull-then-blend, DST slots reused). Blend CBs use ids
// disjoint from the cull set so both coexist in one fused program.
constexpr uint32_t CB_XRAMP     = 8;    // fp32 tile-local x ramp (c + 0.5)
constexpr uint32_t CB_YRAMP     = 9;    // fp32 tile-local y ramp (r + 0.5)
constexpr uint32_t CB_MB_COEFF  = 10;   // one 64B coeff row per gaussian (mb-major)
constexpr uint32_t CB_MB_COUNTS = 11;   // per-tile gaussian-row count (slot 0 == L)
constexpr uint32_t CB_COLOR_OUT = 17;   // 3 bf16 color tiles per screen tile
// Blend DST slot bases (ix*32 units). REUSED across the cull/blend tile_regs
// brackets (never co-resident): cull uses DR_BOX_OX..DR_QH (0..4), blend uses
// DR_R..DR_Y (0..5); max 6 live <= 8.
constexpr uint32_t DR_R = 0 * 32;
constexpr uint32_t DR_G = 1 * 32;
constexpr uint32_t DR_B = 2 * 32;
constexpr uint32_t DR_T = 3 * 32;
constexpr uint32_t DR_X = 4 * 32;
constexpr uint32_t DR_Y = 5 * 32;
#endif

constexpr uint32_t NUM_MB = 32;
constexpr uint32_t BATCH  = 32;  // gaussians per keep-tile (one per SFPU vector)

// STEP-WISE CULL BUILD-UP (user direction). Prove the SFPU mask PIPELINE
// end-to-end with a trivial keep-all before layering the Mahalanobis math back
// on. This constant is the OPERATIVE DEFAULT (no env/debug define): bump it only
// after the current level verifies on device.
//   0 = keep-all   : mask = 0xFFFFFFFF for every microblock/gaussian/tile (no
//                    det/inv/face/exp). Correct (un-culled) image => proves mask
//                    production, lane/microblock layout, CB_KEEP binding and the
//                    blend-reader consumption are all sound.
//   1 = bbox geom   : keep microblock iff it intersects the gaussian pixel bbox
//                    (mean/extent only, no conic).
//   2 = +m2 metric  : add conic det/inverse + the two face quadratics -> m2.
//   3 = full cull   : add opacity*exp(-0.5*m2) >= contrib_floor threshold.
// 0=keep-all, 1=bbox geometry, 2=math-free VARY probe, 3=full Mahalanobis.
constexpr int CULL_LEVEL = 3;
// Parity double-buffer of the packed keep tile (DR_KEEP / DR_KEEP_B). PROVEN a
// no-op: the math-free VARY probe binds with ZERO mismatch both with and without
// it, so the production pipeline is not racy. Left toggled OFF.
constexpr bool CULL_PARITY_DBUF = false;

// DEST slot bases (dst_reg ix units; a tile == 32 vectors).
constexpr uint32_t DR_BOX_OX = 0 * 32;
constexpr uint32_t DR_BOX_OY = 1 * 32;
constexpr uint32_t DR_KEEP   = 2 * 32;
constexpr uint32_t DR_QV     = 3 * 32;  // x-face UN-normalized Qraw (== det*m2_v)
constexpr uint32_t DR_QH     = 4 * 32;  // y-face UN-normalized Qraw (== det*m2_h)
// Parity double-buffer alternate for the packed keep tile. pack_tile(N) reads
// DEST asynchronously; if batch N+1 reuses the SAME DEST tile its fill/SFPU
// writes can clobber tile N's data before the async pack reads it (the proven
// +32 content skew). Alternating the keep tile per batch (tile 2 / tile 5) lets
// pack(N) drain tile A while batch N+1 writes tile B. Provably safe here because
// tile_regs_acquire(N+1) blocks on tile_regs_release(N), so at most ONE pack is
// ever in flight against the next batch's math (<=2 tiles outstanding).
constexpr uint32_t DR_KEEP_B = 5 * 32;

#ifdef TRISC_MATH
// EXACT box-constrained min Mahalanobis^2 (mirrors the soft-float reference
// compute_microblock_mask two-candidate edge projection). The WHOLE metric in
// one SFPU body (det+inv + cov_a/cov_c reciprocals + two face quadratics +
// clamps + exp) overflows the sfpu/gcc LRA ("reload insns per insn (90)") even
// as a single noinline function (verified on device). We therefore SPLIT it
// across three small `noinline,noipa` functions that hand intermediates through
// DEST scratch tiles (DR_QV / DR_QH); each body is light enough to compile.
//
//   v* = cov_b*u_c/cov_a (the 1/det cancels), likewise u*; a face needs only
//   cov_a or cov_c (bare approx_recip seed is fine: v*/u* get CLAMPED to the
//   box). m2 = min(Qv,Qh)/det; det/inv computed once in combine.
//   keep = opacity*exp(-0.5*m2) >= contrib_floor.
//
// THE BUG THAT WAS HERE (and the fix): combine<V> reading dst_reg[DR_QV/DR_QH+V]
// was returning ANOTHER gaussian's value (device mask == a different gaussian's
// correct mask; offline the formula and the CB_KEEP binding were both proven
// exact). Root cause: a DEST read-after-write HAZARD. The old dispatch ran
// face_x<V> (SFPSTORE DR_QV[V]) -> face_y<V> -> combine<V> (SFPLOAD DR_QV[V])
// only ~30 SFPU insns apart; before the store committed, the load returned the
// STALE value previously living in that DEST slot == the gaussian 32 positions
// back (same V lane, prior batch) == the "+32 / one-batch" skew from the doc.
// FIX: the dispatch is now PHASED -- ALL face_x, then ALL face_y, then ALL
// combine -- so 30..60+ SFPU insns separate every store from its matching load
// and the DEST write is long committed before it is read back.
//
// CRITICAL: every runtime scalar is materialized into a named vFloat BEFORE use
// (an inline Converter::as_float(...) inside a bigger expr emits a NON-uniform
// per-lane load instead of a broadcast -> per-lane garbage).

// x-face candidate: Qv = cov_c*u_c^2 - 2*cov_b*u_c*v* + cov_a*v*^2, where
// u_c=clamp(0,[u_lo,u_hi]) and v*=clamp(cov_b*u_c/cov_a,[v_lo,v_hi]). -> DR_QV.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_x(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat u_c = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    { vFloat uh = u_c + vFloat(8.0f); vFloat z = 0.0f; vec_min_max(z, u_c); vec_min_max(u_c, uh); }
    vFloat v_lo = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    vFloat v_hi = v_lo + vFloat(4.0f);
    vFloat ra = approx_recip(cov_a);
    ra = ra * (vFloat(2.0f) - cov_a * ra);
    ra = ra * (vFloat(2.0f) - cov_a * ra);
    vFloat vs = (cov_b * u_c) * ra;
    vec_min_max(v_lo, vs); vec_min_max(vs, v_hi);  // vs = clamp(vs, [v_lo, v_hi])
    vFloat qv = cov_c * (u_c * u_c) + (vFloat(-2.0f) * cov_b) * (u_c * vs) +
                cov_a * (vs * vs);
    dst_reg[DR_QV + V] = qv;
#ifdef CULL_DEBUG_FACE_QV
    // Diagnostic: write Qv STRAIGHT to the keep channel (no combine DEST read in
    // between). If the dumped value matches offline Qv, face_x math + its DEST
    // write are correct and the fault is the combine read-back of DR_QV.
    dst_reg[DR_KEEP + V] = qv;
#endif
}

// y-face candidate: Qh = cov_c*u*^2 - 2*cov_b*u**v_c + cov_a*v_c^2, where
// v_c=clamp(0,[v_lo,v_hi]) and u*=clamp(cov_b*v_c/cov_c,[u_lo,u_hi]). -> DR_QH.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_y(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat v_c = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    { vFloat vh = v_c + vFloat(4.0f); vFloat z = 0.0f; vec_min_max(z, v_c); vec_min_max(v_c, vh); }
    vFloat u_lo = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    vFloat u_hi = u_lo + vFloat(8.0f);
    vFloat rc = approx_recip(cov_c);
    rc = rc * (vFloat(2.0f) - cov_c * rc);
    rc = rc * (vFloat(2.0f) - cov_c * rc);
    vFloat us = (cov_b * v_c) * rc;
    vec_min_max(u_lo, us); vec_min_max(us, u_hi);  // us = clamp(us, [u_lo, u_hi])
    dst_reg[DR_QH + V] = cov_c * (us * us) + (vFloat(-2.0f) * cov_b) * (us * v_c) +
                         cov_a * (v_c * v_c);
}

// combine: m2 = min(DR_QV, DR_QH) / det; keep = opacity*exp(-0.5*m2) >= floor.
// Uses the SELF-CONTAINED 21-bit exp (the primitive the production blend kernel
// rides to 63.85 dB) rather than `_calculate_log_body_no_init_` (which assumes
// an SFPU log LUT init this kernel never performs). Runs ONLY after every
// face_x/face_y of the batch has committed its DEST store (phased dispatch).
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_combine(
    uint32_t keep_base, uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t thr_bits, bool cull_disabled,
    uint32_t pos_base) {
    using namespace sfpi;
    (void)pos_base;
    namespace cs = ckernel::sfpu;
    vFloat qmin = dst_reg[DR_QV + V];
    { vFloat qh = dst_reg[DR_QH + V]; vec_min_max(qmin, qh); }  // qmin = min(Qv, Qh) == det*m2_min
    if (cull_disabled) { qmin = vFloat(0.0f); }  // m2 := 0 -> kept iff thr >= 0
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    { vFloat det_floor = 1e-6f; vec_min_max(det_floor, det); }  // det = max(det, 1e-6)
    // DIVIDE-FREE, TRANSCENDENTAL-FREE threshold (mirrors tile_assign_cull.cpp,
    // the bit-faithful kept-pair cull): keep iff m2 = qmin/det <= thr, i.e.
    // qmin <= det*thr, with thr = -2*log(floor/op) precomputed per-gaussian on
    // the RISC (the soft-float reference's exact thresh_m2). The <0 sentinel for
    // op<=floor drops naturally: qmin>=0 is never <= det*thr<0. No approx_recip,
    // no bf16 SFPU exp -> removes the quantization that flipped borderline
    // microblocks and produced the ~28 dB over-cull.
    vFloat thr = cs::Converter::as_float(thr_bits);
    vFloat scaled = det * thr;
    vFloat keepv = 0.0f;
    v_if(qmin <= scaled) { keepv = vFloat(1.0f); } v_endif;
#ifdef CULL_DEBUG_EMIT_M2
    {  // emit raw m2 = qmin/det (read back by the writer for ground-truth compare)
        vFloat inv = approx_recip(det);
        inv = inv * (vFloat(2.0f) - det * inv);
        inv = inv * (vFloat(2.0f) - det * inv);
        keepv = qmin * inv;
    }
#endif
#ifdef CULL_DEBUG_COMBINE_QV
    // Diagnostic complement to CULL_DEBUG_FACE_QV: emit the value combine reads
    // BACK from DR_QV via SFPLOAD. PROVEN: matches face_x's Qv exactly (read OK).
    keepv = vFloat(dst_reg[DR_QV + V]);
#endif
#ifdef CULL_DEBUG_COMBINE_QH
    keepv = vFloat(dst_reg[DR_QH + V]);  // emit combine's read of the y-face Qh
#endif
#ifdef CULL_DEBUG_COMBINE_DET
    {
        vFloat ca = cs::Converter::as_float(a_bits);
        vFloat cb = cs::Converter::as_float(b_bits);
        vFloat cc = cs::Converter::as_float(c_bits);
        vFloat d = ca * cc - cb * cb;
        keepv = d;  // emit combine's det
    }
#endif
#ifdef CULL_DEBUG_EMIT_BOX
    keepv = vFloat(dst_reg[DR_BOX_OY + V]) * vFloat(32.0f) + vFloat(dst_reg[DR_BOX_OX + V]);
#endif
#ifdef CULL_DEBUG_KEEPALL
    keepv = vFloat(1.0f);
#endif
#ifdef CULL_DEBUG_RAMP
    keepv = vFloat(0.0f);
    v_if(vFloat(dst_reg[DR_BOX_OX + V]) > vFloat(12.0f)) { keepv = vFloat(1.0f); } v_endif;
#endif
#ifdef CULL_DEBUG_RAMPY
    keepv = vFloat(0.0f);
    v_if(vFloat(dst_reg[DR_BOX_OY + V]) > vFloat(14.0f)) { keepv = vFloat(1.0f); } v_endif;
#endif
#ifdef CULL_DEBUG_BIND
    // Pipeline-binding probe: emit a TRIVIAL deterministic value that depends
    // ONLY on this gaussian's tile-local position (pos_base + V), with NO cull
    // math involved. All 32 lanes carry the same value. The writer reads it back
    // off CB_KEEP and compares to its own (processed + g): any mismatch (esp. a
    // batch-granular +/-32) is a producer/consumer binding skew. Proven: zero skew.
    keepv = vFloat(static_cast<float>(pos_base + V));
#endif
#ifndef CULL_DEBUG_FACE_QV
    dst_reg[keep_base + V] = keepv;  // (skipped in FACE_QV probe: face_x owns keep)
#else
    (void)keepv;
#endif
}
// CULL_LEVEL 1 (bbox): register-only per-gaussian keep that uses NO DR_QV/DR_QH
// cross-function DEST scratch (the suspected +32 hazard). Keeps microblock m iff
// its box CENTER lies within a 3-sigma axis-aligned box of the gaussian:
//   |mb_cx - mlx|^2 <= 9*cov_a  AND  |mb_cy - mly|^2 <= 9*cov_c
// (cov_a/cov_c are the x/y variances; squared compare avoids a sqrt). Reads only
// the resident box ramps (DR_BOX_*, same-batch copy_tile) + coeff scalars and
// writes DR_KEEP[V] directly -> if THIS binds with no +32, the DEST scratch in
// the full metric is the culprit; if it ALSO skews, the hazard is broader.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_bbox(
    uint32_t keep_base, uint32_t a_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat dcx = (vFloat(dst_reg[DR_BOX_OX + V]) + vFloat(4.0f)) - mlx;  // mb center x - center
    vFloat dcy = (vFloat(dst_reg[DR_BOX_OY + V]) + vFloat(2.0f)) - mly;  // mb center y - center
    vFloat keepv = 0.0f;
    v_if(dcx * dcx <= vFloat(20.0f) * cov_a) {
        v_if(dcy * dcy <= vFloat(20.0f) * cov_c) {
            keepv = vFloat(1.0f);
        }
        v_endif;
    }
    v_endif;
    dst_reg[keep_base + V] = keepv;
}
// CULL_LEVEL 2 (VARY): MATH-FREE per-gaussian-slot deterministic mask routed
// through the EXACT real DR_KEEP production path (SFPU SFPSTORE -> tile_regs ->
// pack_tile -> CB_KEEP -> writer). NO det/inv/face/bbox/exp. All 32 lanes of
// gaussian V carry the same bit -> writer packs 0xFFFFFFFF or 0x00000000. The
// bit is bit(local) = (local ^ (local>>5)) & 1 with local = processed + V:
//   - alternates per gaussian within a batch (low bit of local), AND
//   - flips for EVERY gaussian under a +/-32 (one-batch) skew (the local>>5
//     term toggles), so the writer's direct value-compare to the SAME closed
//     form catches the race with zero reliance on PSNR or cull math.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_vary(uint32_t keep_base, uint32_t processed) {
    using namespace sfpi;
    const uint32_t local = processed + V;
    const uint32_t bit = (local ^ (local >> 5)) & 1u;
    dst_reg[keep_base + V] = vFloat(bit ? 1.0f : 0.0f);
}

#ifdef FUSE_BLEND
// One gaussian's contribution to a single microblock's 32-lane vector — the
// MB_DEVCONIC blend math copied bit-for-bit from alpha_blend_compute_mb.cpp.
// a/b/c = raw cov; d/e = mean x/y (tile-local); op/cr/cg/cb = opacity+colors.
template <uint32_t IX>
inline void blend_one_gaussian_math(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t d_bits, uint32_t e_bits,
    uint32_t op_bits, uint32_t cr_bits, uint32_t cg_bits, uint32_t cb_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat x = dst_reg[DR_X + IX];
    vFloat y = dst_reg[DR_Y + IX];
    vFloat cov_a = cs::Converter::as_float(a_bits);
    vFloat cov_b = cs::Converter::as_float(b_bits);
    vFloat cov_c = cs::Converter::as_float(c_bits);
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    vFloat det_floor = 1e-6f;
    vec_min_max(det_floor, det);  // det = max(det, 1e-6)
    vFloat inv = approx_recip(det);
    inv = inv * (vFloat(2.0f) - det * inv);
    inv = inv * (vFloat(2.0f) - det * inv);
    vFloat A = vFloat(-0.5f) * (cov_c * inv);
    vFloat B = cov_b * inv;
    vFloat C = vFloat(-0.5f) * (cov_a * inv);
    vFloat mx = cs::Converter::as_float(d_bits);
    vFloat my = cs::Converter::as_float(e_bits);
    vFloat dx = x - mx;
    vFloat dy = y - my;
    vFloat power = A * (dx * dx);
    power = power + B * (dx * dy);
    power = power + C * (dy * dy);
    vFloat zero = 0.0f;
    vec_min_max(power, zero);  // power = min(power, 0)
    vFloat weight = cs::_sfpu_exp_21f_bf16_</*is_fp32_dest_acc_en=*/true>(power);
    vFloat alpha = cs::Converter::as_float(op_bits) * weight;
    vFloat clamp = 0.99f;
    vec_min_max(alpha, clamp);  // alpha = min(alpha, 0.99)
    vFloat t = dst_reg[DR_T + IX];
    vFloat at = alpha * t;
    dst_reg[DR_R + IX] = vFloat(dst_reg[DR_R + IX]) + at * cs::Converter::as_float(cr_bits);
    dst_reg[DR_G + IX] = vFloat(dst_reg[DR_G + IX]) + at * cs::Converter::as_float(cg_bits);
    dst_reg[DR_B + IX] = vFloat(dst_reg[DR_B + IX]) + at * cs::Converter::as_float(cb_bits);
    dst_reg[DR_T + IX] = t * (vFloat(1.0f) - alpha);
}
#endif  // FUSE_BLEND
#endif  // TRISC_MATH

// PHASED compile-time-unrolled dispatch over the 32 gaussian vectors of a batch.
// Phase 1 runs ALL cull_face_x (-> DR_QV), phase 2 ALL cull_face_y (-> DR_QH),
// phase 3 ALL cull_combine (reads DR_QV/DR_QH -> DR_KEEP). Phasing is what fixes
// the DEST read-after-write hazard: by the time combine<V> loads DR_QV[V] the
// face_x<V> store happened a full phase (>=nb SFPU insns) earlier and is
// committed. Compile-time V offsets keep dst_reg addresses immediate.
// Phase 1: ALL x-faces (each writes DR_QV[V]).
template <uint32_t V>
inline void cull_phase_fx(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_face_x<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fx<V + 1>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    }
}

// Phase 2: ALL y-faces (each writes DR_QH[V]).
template <uint32_t V>
inline void cull_phase_fy(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_face_y<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fy<V + 1>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    }
}

// Phase 3: ALL combines (each reads DR_QV[V]/DR_QH[V] -> DR_KEEP[V]).
template <uint32_t V>
inline void cull_phase_combine(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* thr,
    bool cull_disabled) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_combine<V>(keep_base, a[V], b[V], c[V], thr[V], cull_disabled, pos_base)));
        }
        cull_phase_combine<V + 1>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
    }
}

// Level 1: ALL bbox keeps (register-only, no DR_QV/DR_QH scratch).
template <uint32_t V>
inline void cull_phase_bbox(
    uint32_t keep_base, uint32_t nb, const uint32_t* a, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_bbox<V>(keep_base, a[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_bbox<V + 1>(keep_base, nb, a, c, mx, my, txf_bits, tyf_bits);
    }
}

// Level 2: ALL math-free VARY keeps (each SFPSTOREs its closed-form bit).
template <uint32_t V>
inline void cull_phase_vary(uint32_t keep_base, uint32_t processed, uint32_t nb) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_vary<V>(keep_base, processed)));
        }
        cull_phase_vary<V + 1>(keep_base, processed, nb);
    }
}

inline void cull_dispatch(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits,
    bool cull_disabled) {
    cull_phase_fx<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_fy<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_combine<0>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
}

inline float u32_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}
inline uint32_t f_to_u32(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

#ifdef FUSE_BLEND
// Identity permutation (bit m -> vector m), as in alpha_blend_compute_mb.cpp.
template <uint32_t M>
inline void dispatch_blend_guarded(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    if constexpr (M < NUM_MB) {
        if (mask & (1u << M)) {
            MATH((blend_one_gaussian_math<M>(a, b, c, d, e, op, cr, cg, cbv)));
        }
        dispatch_blend_guarded<M + 1>(mask, a, b, c, d, e, op, cr, cg, cbv);
    }
}

// Stream all of a tile's gaussian rows from CB_MB_COEFF, dispatching each to its
// masked microblocks. One start_/done_ for the whole tile (proven safe).
inline void blend_tile_gaussians(uint32_t num_g) {
    if (num_g == 0) {
        return;
    }
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    for (uint32_t g = 0; g < num_g; g++) {
        cb_wait_front(CB_MB_COEFF, 1);
        const uint32_t* row = reinterpret_cast<const uint32_t*>(get_tile_address(CB_MB_COEFF, 0));
        const uint32_t a = row[0], b = row[1], c = row[2], d = row[3], e = row[4];
        const uint32_t op = row[6], cr = row[7], cg = row[8], cbv = row[9];
        const uint32_t mask = row[10];
        dispatch_blend_guarded<0>(mask, a, b, c, d, e, op, cr, cg, cbv);
        cb_pop_front(CB_MB_COEFF, 1);
    }
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
}
#endif  // FUSE_BLEND

}  // namespace

#ifdef CULL_LPT_CB
constexpr uint32_t CB_CORE_TILES = 7;
#endif

void kernel_main() {
    DeviceZoneScopedN("cull_global_mb");
    uint32_t num_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t floor_bits   = get_arg_val<uint32_t>(1);
    const bool cull_disabled    = get_arg_val<uint32_t>(2) != 0;

#ifdef CULL_LPT_CB
    if (num_tiles == 0) {
        cb_wait_front(CB_CORE_TILES, 1);
        num_tiles = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
        cb_pop_front(CB_CORE_TILES, 1);
    }
#endif

    init_sfpu(CB_BOX_OX, CB_KEEP);
    fill_tile_init();

    if (num_tiles == 0) {
        return;
    }

    // Constant box-origin ramps: produced once by the reader, kept resident in
    // the CB (waited once, never popped) and re-copied into DEST each batch.
    cb_wait_front(CB_BOX_OX, 1);
    cb_wait_front(CB_BOX_OY, 1);
#ifdef FUSE_BLEND
    // Pixel-center ramps for the blend phase: streamed once by the reader, kept
    // resident (waited once, never popped), re-copied into DEST each tile.
    cb_wait_front(CB_XRAMP, 1);
    cb_wait_front(CB_YRAMP, 1);
#endif

    // Parity for the keep-tile double-buffer: every batch (across tiles) flips
    // which DEST tile (DR_KEEP / DR_KEEP_B) holds + is packed, so the async pack
    // of batch N never collides with batch N+1's fill/SFPU writes.
    uint32_t batch_par = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_CULL_COUNTS, 1);
        uint32_t L, tx_pix, ty_pix;
        {
            auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CULL_COUNTS, 0));
            L      = cptr[0];
            tx_pix = cptr[1];
            ty_pix = cptr[2];
        }
        // Tile origin -> float (one scalar conversion per tile; on the compute
        // core, never on a data mover).
        const uint32_t txf_bits = f_to_u32(static_cast<float>(tx_pix));
        const uint32_t tyf_bits = f_to_u32(static_cast<float>(ty_pix));

#ifdef FUSE_BLEND
        // The previous tile's blend packed bf16 into CB_COLOR_OUT; restore the
        // fp32 keep-tile pack format before this tile's cull batches.
        pack_reconfig_data_format(CB_KEEP);
#endif
        uint32_t processed = 0;
        while (processed < L) {
            uint32_t nb = L - processed;
            if (nb > BATCH) nb = BATCH;

            // Pull this batch's coeff rows into L1-local arrays. row[6] is the
            // per-gaussian Mahalanobis threshold thr = -2*log(floor/op) (the
            // reference's exact thresh_m2, <0 sentinel for op<=floor) precomputed
            // by the reader on the data mover -- the compute TRISC's own logf
            // returns garbage, so the transcendental is kept off this core. The
            // SFPU keep test is the divide-free `qmin <= det*thr`.
            uint32_t a[BATCH], b[BATCH], c[BATCH], mx[BATCH], my[BATCH], thr[BATCH];
            for (uint32_t i = 0; i < nb; i++) {
                cb_wait_front(CB_CULL_COEFF, 1);
                auto row = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CULL_COEFF, 0));
                a[i]  = row[0];
                b[i]  = row[1];
                c[i]  = row[2];
                mx[i] = row[3];
                my[i] = row[4];
                thr[i] = row[6];
                cb_pop_front(CB_CULL_COEFF, 1);
            }

            // Alternate the packed keep tile per batch (parity double-buffer).
            const uint32_t keep_base = (CULL_PARITY_DBUF && (batch_par & 1u)) ? DR_KEEP_B : DR_KEEP;

            tile_regs_acquire();
            // Level 0 (keep-all): fill DR_KEEP with 1.0 for ALL 32 lanes of all
            // 32 vectors -> the writer packs 0xFFFFFFFF for every gaussian. No
            // box ramps, no SFPU math instantiated (if-constexpr discards the
            // dispatch, so the heavy metric never compiles at this level).
            fill_tile(keep_base / 32, (CULL_LEVEL == 0) ? 1.0f : 0.0f);
            if constexpr (CULL_LEVEL == 2) {
                // VARY: math-free per-slot bit through the real DR_KEEP path. No
                // box ramps, no geometry -> isolates the pack/CB_KEEP pipeline.
                MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
                cull_phase_vary<0>(keep_base, processed, nb);
                MATH((_llk_math_eltwise_unary_sfpu_done_()));
            } else if constexpr (CULL_LEVEL >= 1) {
                copy_tile_to_dst_init_short(CB_BOX_OX);
                copy_tile(CB_BOX_OX, 0, DR_BOX_OX / 32);
                copy_tile_to_dst_init_short(CB_BOX_OY);
                copy_tile(CB_BOX_OY, 0, DR_BOX_OY / 32);

                MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
                if constexpr (CULL_LEVEL == 1) {
                    cull_phase_bbox<0>(keep_base, nb, a, c, mx, my, txf_bits, tyf_bits);
                } else {
                    cull_dispatch(keep_base, nb, processed, a, b, c, mx, my, thr, txf_bits, tyf_bits, cull_disabled);
                }
                MATH((_llk_math_eltwise_unary_sfpu_done_()));
            }
#if defined(CULL_DEBUG_VALS)
            {
                static uint32_t dbg_vals = 0;
                if (dbg_vals < 3u) {
                    for (uint32_t V = 0; V < nb && V < 3u; V++) {
                        DPRINT << "CULLVAL V=" << V
                               << " a=" << F32(u32_to_f(a[V])) << " b=" << F32(u32_to_f(b[V]))
                               << " c=" << F32(u32_to_f(c[V]))
                               << " mx=" << F32(u32_to_f(mx[V])) << " my=" << F32(u32_to_f(my[V]))
                               << " thr=" << F32(u32_to_f(thr[V]))
                               << " tx=" << F32(u32_to_f(txf_bits)) << " ty=" << F32(u32_to_f(tyf_bits))
                               << " fl=" << F32(u32_to_f(floor_bits)) << ENDL();
                    }
                    dbg_vals++;
                }
            }
#endif

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_KEEP, 1);
            pack_tile(keep_base / 32, CB_KEEP);
            cb_push_back(CB_KEEP, 1);
            tile_regs_release();

            batch_par ^= 1u;
            processed += nb;
        }

        cb_pop_front(CB_CULL_COUNTS, 1);

#ifdef FUSE_BLEND
        // ---- BLEND PHASE (tile t) -------------------------------------------
        // The reader has, by now, emitted exactly L blend coeff rows into
        // CB_MB_COEFF (each carrying the L1-handoff mask in row[10]); composite
        // them into R/G/B/T and pack 3 bf16 color tiles. Runs AFTER the cull
        // phase so the cull (DR_BOX/QV/QH/KEEP) and blend (DR_R/G/B/T/X/Y) DST
        // slots are reused, never co-resident (max 6 live <= 8).
        cb_wait_front(CB_MB_COUNTS, 1);
        uint32_t num_g;
        {
            auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0));
            num_g = cptr[0];
        }
        tile_regs_acquire();
        fill_tile(DR_R / 32, 0.0f);
        fill_tile(DR_G / 32, 0.0f);
        fill_tile(DR_B / 32, 0.0f);
        fill_tile(DR_T / 32, 1.0f);
        copy_tile_to_dst_init_short(CB_XRAMP);
        copy_tile(CB_XRAMP, 0, DR_X / 32);
        copy_tile_to_dst_init_short(CB_YRAMP);
        copy_tile(CB_YRAMP, 0, DR_Y / 32);
        blend_tile_gaussians(num_g);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_COLOR_OUT);
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(DR_R / 32, CB_COLOR_OUT);
        pack_tile(DR_G / 32, CB_COLOR_OUT);
        pack_tile(DR_B / 32, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();
        cb_pop_front(CB_MB_COUNTS, 1);
#endif  // FUSE_BLEND
    }
    (void)u32_to_f;
    (void)floor_bits;
}
