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
//   A1 (iter 111): the record words now carry the PRE-FOLDED conic {A,B,C}
//   (hoisted into pfwc), so the precision matrix is recovered with two cheap
//   scalar multiplies — ci_a = -2A (= cov_c/det), ci_b = -B (= -cov_b/det),
//   ci_c = -2C (= cov_a/det) — instead of a per-face determinant + reciprocal
//   that re-derived conic from raw cov. The keep-test now runs DIRECTLY in
//   conic/precision space (no det recompute, no det*thr scaling):
//   constrained-min Mahalanobis^2 m2_min = min over the box of
//     m2(u,v) = ci_a*u^2 + 2*ci_b*u*v + ci_c*v^2,
//   keep = (m2_min <= thr) with thr = 2*ln(op/floor). iter 108: thr is computed
//   HERE on the SFPU (hardware log, _calculate_log_body_no_init_) from the raw
//   opacity the reader streams -- the per-pair soft-float __builtin_logf
//   (~74 ms) is removed from the NCRISC cull reader. The only per-face recip
//   that remains is the edge-projection optimum (1/ci_c for the x-face, 1/ci_a
//   for the y-face) -- intrinsic to the constrained min, not a conic re-derive.
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
constexpr uint32_t CB_CULL_COEFF = 2;   // (iter 109: retired — records read from CB_BUCKET)
constexpr uint32_t CB_CULL_COUNTS= 3;   // per-tile [L, tx_pix, ty_pix]
constexpr uint32_t CB_BUCKET     = 8;   // iter 109: bulk depth-sorted PACK2 slab (mirrors blend)
constexpr uint32_t CB_KEEP       = 16;  // fp32 keep tile (one per 32-gaussian batch)

constexpr uint32_t BATCH  = 32;  // gaussians per keep-tile (one per SFPU vector)

// iter 109: PACK2 slab geometry + fixed bulk slot (mirror reader_tile_l1_cull /
// alpha_blend_compute_mb). Two 32B splats per 64B page: record g -> page g/2,
// half g&1. The reader pushes ONE BULK_REC_SLOT slot per subchunk; the compute
// reads every record straight from this L1 slab (no per-record CB stream).
constexpr uint32_t L1_SPLAT_BYTES     = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;
constexpr uint32_t MB_BUCKET_FIT      = 8192u;
constexpr uint32_t BULK_REC_SLOT      = (MB_BUCKET_FIT + 1u) >> 1;  // 4096

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline const uint32_t* l1_splat_words(uint32_t buck, uint32_t g) {
    return reinterpret_cast<const uint32_t*>(
        buck + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
}

// L1 read-visibility fence: invalidate any stale cached line for the recycled
// bulk-CB slot so the freshly DMA'd slab records read coherently. ==
// invalidate_l1_cache(); mirrors alpha_blend_compute_mb::mb_cb_consume_fence.
inline void mb_cb_consume_fence() {
    asm volatile("fence" ::: "memory");
}

// DEST slot bases (dst_reg ix units; a tile == 32 vectors).
constexpr uint32_t DR_BOX_OX = 0 * 32;
constexpr uint32_t DR_BOX_OY = 1 * 32;
constexpr uint32_t DR_KEEP   = 2 * 32;
constexpr uint32_t DR_QV     = 3 * 32;  // x-face UN-normalized Qraw (== det*m2_v)
constexpr uint32_t DR_QH     = 4 * 32;  // y-face UN-normalized Qraw (== det*m2_h)
constexpr uint32_t DR_THR    = 5 * 32;  // iter 108: per-gaussian thr = 2*ln(op/floor) (SFPU log)

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

// x-face candidate (conic): m2_v = ci_a*u_c^2 + 2*ci_b*u_c*v* + ci_c*v*^2, where
// u_c=clamp(0,[u_lo,u_hi]) and v*=clamp(-ci_b*u_c/ci_c,[v_lo,v_hi]). -> DR_QV.
// a_bits/b_bits/c_bits carry the conic {A,B,C}: ci_a=-2A, ci_b=-B, ci_c=-2C.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_x(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat ci_a = vFloat(-2.0f) * cs::Converter::as_float(a_bits);  // = cov_c/det
    vFloat ci_b = vFloat(-1.0f) * cs::Converter::as_float(b_bits);  // = -cov_b/det
    vFloat ci_c = vFloat(-2.0f) * cs::Converter::as_float(c_bits);  // = cov_a/det
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat u_c = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    { vFloat uh = u_c + vFloat(8.0f); vFloat z = 0.0f; vec_min_max(z, u_c); vec_min_max(u_c, uh); }
    vFloat v_lo = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    vFloat v_hi = v_lo + vFloat(4.0f);
    // v* = -ci_b*u_c/ci_c minimizes m2 along the fixed-u edge; clamp to the box.
    vFloat rc = approx_recip(ci_c);
    rc = rc * (vFloat(2.0f) - ci_c * rc);
    rc = rc * (vFloat(2.0f) - ci_c * rc);
    vFloat vs = (vFloat(-1.0f) * ci_b * u_c) * rc;
    vec_min_max(v_lo, vs); vec_min_max(vs, v_hi);  // vs = clamp(vs, [v_lo, v_hi])
    dst_reg[DR_QV + V] = ci_a * (u_c * u_c) + (vFloat(2.0f) * ci_b) * (u_c * vs) +
                         ci_c * (vs * vs);
}

// y-face candidate (conic): m2_h = ci_a*u*^2 + 2*ci_b*u**v_c + ci_c*v_c^2, where
// v_c=clamp(0,[v_lo,v_hi]) and u*=clamp(-ci_b*v_c/ci_a,[u_lo,u_hi]). -> DR_QH.
// a_bits/b_bits/c_bits carry the conic {A,B,C}: ci_a=-2A, ci_b=-B, ci_c=-2C.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_face_y(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t mx_bits, uint32_t my_bits, uint32_t txf_bits, uint32_t tyf_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat ci_a = vFloat(-2.0f) * cs::Converter::as_float(a_bits);  // = cov_c/det
    vFloat ci_b = vFloat(-1.0f) * cs::Converter::as_float(b_bits);  // = -cov_b/det
    vFloat ci_c = vFloat(-2.0f) * cs::Converter::as_float(c_bits);  // = cov_a/det
    vFloat mlx = cs::Converter::as_float(mx_bits) - cs::Converter::as_float(txf_bits);
    vFloat mly = cs::Converter::as_float(my_bits) - cs::Converter::as_float(tyf_bits);
    vFloat v_c = vFloat(dst_reg[DR_BOX_OY + V]) - mly;
    { vFloat vh = v_c + vFloat(4.0f); vFloat z = 0.0f; vec_min_max(z, v_c); vec_min_max(v_c, vh); }
    vFloat u_lo = vFloat(dst_reg[DR_BOX_OX + V]) - mlx;
    vFloat u_hi = u_lo + vFloat(8.0f);
    // u* = -ci_b*v_c/ci_a minimizes m2 along the fixed-v edge; clamp to the box.
    vFloat ra = approx_recip(ci_a);
    ra = ra * (vFloat(2.0f) - ci_a * ra);
    ra = ra * (vFloat(2.0f) - ci_a * ra);
    vFloat us = (vFloat(-1.0f) * ci_b * v_c) * ra;
    vec_min_max(u_lo, us); vec_min_max(us, u_hi);  // us = clamp(us, [u_lo, u_hi])
    dst_reg[DR_QH + V] = ci_a * (us * us) + (vFloat(2.0f) * ci_b) * (us * v_c) +
                         ci_c * (v_c * v_c);
}

// iter 108: per-gaussian Mahalanobis threshold thr = -2*ln(floor/op) computed
// ON THE SFPU (hardware log), replacing the NCRISC reader's per-pair soft-float
// __builtin_logf (~74 ms). thr depends ONLY on opacity, so one broadcast log per
// gaussian suffices. op and inv_floor are uniform => ratio/log are uniform across
// the 32 lanes; thr is stored broadcast into DR_THR[V] for cull_combine.
//   thr = 2*ln(op/floor). For op<=floor: ratio<=1 -> ln<=0 -> thr<=0, which makes
//   the keep test (qmin<=det*thr, qmin>=0, det>0) FALSE => culled (the old <0
//   sentinel drops out naturally, no special-case needed).
// _calculate_log_body_no_init_ uses literal cheby coeffs (no vConstFloatPrgm LUT
// init), so it is safe inside this kernel's custom SFPU dispatch.
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_thr(uint32_t op_bits, uint32_t inv_floor_bits) {
    using namespace sfpi;
    namespace cs = ckernel::sfpu;
    vFloat op = cs::Converter::as_float(op_bits);
    vFloat inv_floor = cs::Converter::as_float(inv_floor_bits);
    vFloat ratio = op * inv_floor;
    vFloat logv = cs::_calculate_log_body_no_init_(ratio);
    dst_reg[DR_THR + V] = logv + logv;  // 2*ln(op/floor)
}

// combine (conic): m2_min = min(DR_QV, DR_QH); keep iff m2_min <= thr. A1: the
// faces now write m2 DIRECTLY (conic space), so there is NO determinant recompute
// and NO det*thr scaling here — just a min + a compare. thr is read from
// DR_THR[V] (committed a full phase earlier by cull_thr). Runs ONLY after every
// face_x/face_y/thr of the batch has committed its DEST store (phased dispatch ->
// no DEST read-after-write hazard).
template <uint32_t V>
__attribute__((noinline, noipa)) void cull_combine(
    uint32_t keep_base, bool cull_disabled) {
    using namespace sfpi;
    vFloat qmin = dst_reg[DR_QV + V];
    { vFloat qh = dst_reg[DR_QH + V]; vec_min_max(qmin, qh); }  // qmin = min(m2_v, m2_h)
    if (cull_disabled) { qmin = vFloat(0.0f); }  // m2 := 0 -> kept iff thr >= 0
    vFloat thr = dst_reg[DR_THR + V];
    vFloat keepv = 0.0f;
    v_if(qmin <= thr) { keepv = vFloat(1.0f); } v_endif;
    dst_reg[keep_base + V] = keepv;
}
#endif  // TRISC_MATH

// PHASED compile-time-unrolled dispatch over the 32 gaussian vectors of a batch.
// Phase 0 runs ALL cull_thr (-> DR_THR), phase 1 ALL cull_face_x (-> DR_QV),
// phase 2 ALL cull_face_y (-> DR_QH), phase 3 ALL cull_combine (reads
// DR_QV/DR_QH/DR_THR -> DR_KEEP). Phasing is what fixes the DEST read-after-write
// hazard: by the time combine<V> loads DR_*[V] the matching store happened a full
// phase (>=nb SFPU insns) earlier and is committed. Compile-time V offsets keep
// dst_reg addresses immediate. iter 108: the per-pair soft-float pre-cull skip is
// gone -- the SFPU log/combine culls op<=floor naturally (thr<=0 => keep=0), so
// every in-batch gaussian runs all phases (consistent DEST writes/reads).
// Phase 0: ALL thr (each writes DR_THR[V]).
template <uint32_t V>
inline void cull_phase_thr(
    uint32_t nb, const uint32_t* op, uint32_t inv_floor_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_thr<V>(op[V], inv_floor_bits)));
        }
        cull_phase_thr<V + 1>(nb, op, inv_floor_bits);
    }
}

// Phase 1: ALL x-faces (each writes DR_QV[V]).
template <uint32_t V>
inline void cull_phase_fx(
    uint32_t nb, const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my,
    uint32_t txf_bits, uint32_t tyf_bits) {
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
    const uint32_t* mx, const uint32_t* my,
    uint32_t txf_bits, uint32_t tyf_bits) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_face_y<V>(a[V], b[V], c[V], mx[V], my[V], txf_bits, tyf_bits)));
        }
        cull_phase_fy<V + 1>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    }
}

// Phase 3: ALL combines (each reads DR_QV[V]/DR_QH[V]/DR_THR[V] -> DR_KEEP[V]).
// A1: combine is conic-space (m2 direct), so it no longer needs a/b/c.
template <uint32_t V>
inline void cull_phase_combine(
    uint32_t keep_base, uint32_t nb, bool cull_disabled) {
    if constexpr (V < BATCH) {
        if (V < nb) {
            MATH((cull_combine<V>(keep_base, cull_disabled)));
        }
        cull_phase_combine<V + 1>(keep_base, nb, cull_disabled);
    }
}

inline void cull_dispatch(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* op,
    uint32_t inv_floor_bits, uint32_t txf_bits, uint32_t tyf_bits,
    bool cull_disabled) {
    (void)pos_base;
    cull_phase_thr<0>(nb, op, inv_floor_bits);
    cull_phase_fx<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_fy<0>(nb, a, b, c, mx, my, txf_bits, tyf_bits);
    cull_phase_combine<0>(keep_base, nb, cull_disabled);
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
    // iter 108: thr = -2*ln(floor/op) = 2*ln(op/floor) computed on the SFPU.
    // Precompute inv_floor = 1/floor once per core (single soft-float reciprocal,
    // NOT per-pair) so the per-gaussian SFPU log multiplies by a broadcast scalar.
    float floor_f;
    __builtin_memcpy(&floor_f, &floor_bits, 4);
    const float inv_floor_f = (floor_f > 0.0f) ? (1.0f / floor_f) : 0.0f;
    uint32_t inv_floor_bits;
    __builtin_memcpy(&inv_floor_bits, &inv_floor_f, 4);

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

        // iter 109: the reader hands the whole depth-sorted PACK2 slab for this
        // subchunk via CB_BUCKET in ONE bulk push (mirrors the blend reader). The
        // compute reads each record straight from L1 here — deriving the SAME
        // values the old per-record CB_CULL_COEFF stream carried — instead of
        // ~3.37M per-record CB handshakes (the ~40 ms cull-vs-blend load delta).
        if (L == 0) {
            cb_pop_front(CB_CULL_COUNTS, 1);
            continue;
        }
        cb_wait_front(CB_BUCKET, BULK_REC_SLOT);
        const uint32_t buck = get_tile_address(CB_BUCKET, 0);
        mb_cb_consume_fence();
        const float tx_tile_f = bits_to_f(txf_bits);
        const float ty_tile_f = bits_to_f(tyf_bits);
        constexpr float kUnormInv = 1.0f / 65535.0f;

        uint32_t processed = 0;
        while (processed < L) {
            uint32_t nb = L - processed;
            if (nb > BATCH) nb = BATCH;

            // Reproduce emit_cull_row_from_l1_splat (A1: rec[0..2] now carry the
            // pre-folded conic {A,B,C} instead of raw cov; the faces recover the
            // precision matrix as ci=-2A,-B,-2C). center = tile-local mean
            // (rec[4],rec[5]) + tile origin (the SFPU subtracts it back via
            // txf/tyf); op = UNORM16 (rec[6]&0xffff)/65535. thr is SFPU (cull_thr).
            uint32_t a[BATCH], b[BATCH], c[BATCH], mx[BATCH], my[BATCH], op[BATCH];
            for (uint32_t i = 0; i < nb; i++) {
                const uint32_t* rec = l1_splat_words(buck, processed + i);
                a[i]  = rec[0];
                b[i]  = rec[1];
                c[i]  = rec[2];
                mx[i] = f_to_u32(bits_to_f(rec[4]) + tx_tile_f);
                my[i] = f_to_u32(bits_to_f(rec[5]) + ty_tile_f);
                op[i] = f_to_u32(static_cast<float>(rec[6] & 0xffffu) * kUnormInv);
            }

            tile_regs_acquire();
            // Seed DR_KEEP to 0 (default-cull), copy the resident box-origin
            // ramps into DEST, then run the full Mahalanobis cull dispatch
            // (phased face_x -> face_y -> combine) which writes the per-microblock
            // keep flag (1.0/0.0) into DR_KEEP for every gaussian of the batch.
            fill_tile(DR_KEEP / 32, 0.0f);
            copy_tile_to_dst_init_short(CB_BOX_OX);
            copy_tile(CB_BOX_OX, 0, DR_BOX_OX / 32);
            copy_tile_to_dst_init_short(CB_BOX_OY);
            copy_tile(CB_BOX_OY, 0, DR_BOX_OY / 32);

            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            cull_dispatch(DR_KEEP, nb, processed, a, b, c, mx, my, op,
                          inv_floor_bits, txf_bits, tyf_bits, cull_disabled);
            MATH((_llk_math_eltwise_unary_sfpu_done_()));

            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_KEEP, 1);
            pack_tile(DR_KEEP / 32, CB_KEEP);
            cb_push_back(CB_KEEP, 1);
            tile_regs_release();

            processed += nb;
        }

        // MATH->UNPACK back-pressure ack (mirrors alpha_blend_compute_mb): UNPACK
        // would otherwise cb_pop_front this CB_BUCKET slot the instant it ran the
        // batch loop (its MATH() reads are no-ops), letting the FAST reader DMA
        // recycle the slot to the next subchunk before the MATH thread finished
        // reading every record => torn records. Block UNPACK on MATH completion.
        MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, L + 1u)));
        UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId));
        cb_pop_front(CB_BUCKET, BULK_REC_SLOT);

        cb_pop_front(CB_CULL_COUNTS, 1);
    }
}
