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

constexpr uint32_t BATCH  = 32;  // gaussians per keep-tile (one per SFPU vector)

// DEST slot bases (dst_reg ix units; a tile == 32 vectors).
constexpr uint32_t DR_BOX_OX = 0 * 32;
constexpr uint32_t DR_BOX_OY = 1 * 32;
constexpr uint32_t DR_KEEP   = 2 * 32;
constexpr uint32_t DR_QV     = 3 * 32;  // x-face UN-normalized Qraw (== det*m2_v)
constexpr uint32_t DR_QH     = 4 * 32;  // y-face UN-normalized Qraw (== det*m2_h)

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
    dst_reg[DR_QV + V] = cov_c * (u_c * u_c) + (vFloat(-2.0f) * cov_b) * (u_c * vs) +
                         cov_a * (vs * vs);
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
    dst_reg[keep_base + V] = keepv;
}
#endif  // TRISC_MATH

// Reader precomputes thr = -1 for op<=floor; negative float => culled everywhere.
inline bool thr_pre_culled(uint32_t thr_bits) { return (thr_bits & 0x80000000u) != 0; }

inline bool batch_all_pre_culled(uint32_t nb, const uint32_t* thr) {
    for (uint32_t i = 0; i < nb; ++i) {
        if (!thr_pre_culled(thr[i])) {
            return false;
        }
    }
    return true;
}

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
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb && !thr_pre_culled(thr[V])) {
            MATH((cull_face_x<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fx<V + 1>(nb, a, b, c, mx, my, thr, txf_bits, tyf_bits);
    }
}

// Phase 2: ALL y-faces (each writes DR_QH[V]).
template <uint32_t V>
inline void cull_phase_fy(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb && !thr_pre_culled(thr[V])) {
            MATH((cull_face_y<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fy<V + 1>(nb, a, b, c, mx, my, thr, txf_bits, tyf_bits);
    }
}

// Phase 3: ALL combines (each reads DR_QV[V]/DR_QH[V] -> DR_KEEP[V]).
template <uint32_t V>
inline void cull_phase_combine(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t* thr,
    bool cull_disabled) {
    if constexpr (V < BATCH) {
        if (V < nb && !thr_pre_culled(thr[V])) {
            MATH((cull_combine<V>(keep_base, a[V], b[V], c[V], thr[V], cull_disabled, pos_base)));
        }
        cull_phase_combine<V + 1>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
    }
}

inline void cull_dispatch(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits,
    bool cull_disabled) {
    cull_phase_fx<0>(nb, a, b, c, mx, my, thr, txf_bits, tyf_bits);
    cull_phase_fy<0>(nb, a, b, c, mx, my, thr, txf_bits, tyf_bits);
    cull_phase_combine<0>(keep_base, nb, pos_base, a, b, c, thr, cull_disabled);
}

inline uint32_t f_to_u32(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return b;
}

}  // namespace

constexpr uint32_t CB_CORE_TILES = 7;

void kernel_main() {
#ifdef TILE_L1_CULL
    DeviceZoneScopedN("tile_mb_mask");
#else
    DeviceZoneScopedN("cull_global_mb");
#endif
    uint32_t num_tiles    = get_arg_val<uint32_t>(0);
    const uint32_t floor_bits   = get_arg_val<uint32_t>(1);
    const bool cull_disabled    = get_arg_val<uint32_t>(2) != 0;

    if (num_tiles == 0) {
        cb_wait_front(CB_CORE_TILES, 1);
        num_tiles = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
        cb_pop_front(CB_CORE_TILES, 1);
    }

    init_sfpu(CB_BOX_OX, CB_KEEP);
    fill_tile_init();

    if (num_tiles == 0) {
        return;
    }

    // Constant box-origin ramps: produced once by the reader, kept resident in
    // the CB (waited once, never popped) and re-copied into DEST each batch.
    cb_wait_front(CB_BOX_OX, 1);
    cb_wait_front(CB_BOX_OY, 1);

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

            const bool all_pc = batch_all_pre_culled(nb, thr);

            tile_regs_acquire();
            // Seed DR_KEEP to 0 (default-cull). Box-origin ramps are tile-constant:
            // copy once per tile (first batch only), not every 32-gaussian batch.
            fill_tile(DR_KEEP / 32, 0.0f);
            if (processed == 0) {
                copy_tile_to_dst_init_short(CB_BOX_OX);
                copy_tile(CB_BOX_OX, 0, DR_BOX_OX / 32);
                copy_tile_to_dst_init_short(CB_BOX_OY);
                copy_tile(CB_BOX_OY, 0, DR_BOX_OY / 32);
            }

            if (!all_pc) {
                MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
                cull_dispatch(
                    DR_KEEP, nb, processed, a, b, c, mx, my, thr, txf_bits, tyf_bits, cull_disabled);
                MATH((_llk_math_eltwise_unary_sfpu_done_()));
            }

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_KEEP, 1);
            pack_tile(DR_KEEP / 32, CB_KEEP);
            cb_push_back(CB_KEEP, 1);
            tile_regs_release();

            processed += nb;
        }

        cb_pop_front(CB_CULL_COUNTS, 1);
    }
    (void)floor_bits;
}
