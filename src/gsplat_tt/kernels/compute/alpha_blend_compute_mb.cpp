// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Microblock-major (4x8) alpha-blend compute kernel — amendment-003 step 3.
//
// WHY THIS KERNEL
// ---------------
// The full-tile DST-persistent kernel processes EVERY (gaussian, tile) pair on
// all 1024 pixels of a tile, even though a gaussian's 3-sigma footprint usually
// touches only a handful of the tile's 32 microblocks. This kernel instead
// processes one 4-row x 8-col microblock at a time and loops only over THAT
// microblock's culled gaussian list (built host-side, mb_payload.cpp).
//
// HARDWARE GROUND TRUTH (Blackhole, quoted from tt-llk ckernel_sfpu_binary_bcast.h):
//   * One SFPU 32-lane vector == 4 dest rows x 8 cols == exactly one microblock.
//   * dst_reg[ix] addresses the vector at DEST addr ix*SFP_DESTREG_STRIDE
//     (STRIDE=2). With _llk_math_eltwise_unary_sfpu_start_(0) the base is DEST 0,
//     so dst_reg[slot*32 + (MB_TO_DST_ADDR[m]>>1)] reaches (slot, microblock m).
//     The dst_reg index MUST be a compile-time constant (SFPLOAD/SFPSTORE encode
//     the address as an immediate), so the 32-microblock loop is unrolled with a
//     templated vector index.
//
// DST SLOT MAP (fp32, dst_full_sync_en -> 8 fp32 tiles available):
//   0=R  1=G  2=B  3=T   (per-pixel running state, persistent across the tile)
//   4=XRAMP  5=YRAMP      (tile-local pixel-center coords, x=c+0.5 / y=r+0.5)
//
// Per-gaussian math (mirrors gsplat_tt::blend_from_mb_payload_cpu exactly):
//   power  = A*x*x + B*x*y + C*y*y + D*x + E*y + F   (-0.5 already folded host-side)
//   weight = exp(min(power, 0))
//   alpha  = min(opacity*weight, 0.99)
//   at = alpha*T ; R += at*cr ; G += at*cg ; B += at*cb ; T *= (1 - alpha)
//
// Runtime per-gaussian coefficients broadcast into all 32 lanes via
// Converter::as_float (the proven binop_with_scalar mechanism). exp uses the
// 21-bit-accurate _sfpu_exp_21f_bf16_ so we track the fp32 CPU oracle to >=60 dB.

#include <cstdint>

#include "api/compute/common.h"
#include "tools/profiler/kernel_profiler.hpp"  // DeviceZoneScopedN (compute include-order: define before kernel_main)
#if defined(MB_DEBUG_DPRINT) || defined(MB_BUCKET_AB_PROBE)
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
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

namespace {

constexpr uint32_t CB_XRAMP     = 0;   // fp32 tile-local x ramp (c + 0.5)
constexpr uint32_t CB_YRAMP     = 1;   // fp32 tile-local y ramp (r + 0.5)
constexpr uint32_t CB_MB_COEFF  = 2;   // one 48B coeff row per gaussian (mb-major)
constexpr uint32_t CB_MB_COUNTS = 3;   // 32 uint32 per tile (per-microblock count)
constexpr uint32_t CB_CORE_TILES = 7;  // MB_RESIDENT: tile count from reader (no host arg)
constexpr uint32_t CB_COLOR_OUT = 16;

#if defined(MB_BLEND_TAILSKIP)
// M2 §6 reader-side tile-saturation tail-skip handshake (shared L1 with the
// reader; see CB_HS in blend_device.cpp + the protocol in
// reader_alpha_blend_mb_devcull.cpp). MB_BLEND_TAILSKIP implies MB_BLEND_EARLYOUT.
constexpr uint32_t CB_HS       = 17;
constexpr uint32_t HS_DONE_REQ = 0;  // compute -> reader (epoch e on saturation)
// EPOCH/PUSHED/REND are PING-PONGED by tile parity (the reader may be one tile
// ahead): per-parity triplet base = 1 + (parity*3). Must match the reader.
inline uint32_t hs_base(uint32_t parity) { return 1u + parity * 3u; }
constexpr uint32_t HS_EPOCH    = 0;  // offset within a parity triplet
constexpr uint32_t HS_PUSHED   = 1;
constexpr uint32_t HS_REND     = 2;
#endif

#if defined(MB_BLEND_EARLYOUT)
// M2 §6 transmittance early-out (gated). For a DENSE tile we cannot break the
// gaussian loop on "all pixels saturated" purely on the SFPU — sfpi has a lane
// reduce but no cheap scalar extract, and the DST-persistent R/G/B/T cannot be
// inspected mid-bracket. So we process the tile in BLOCKS: each block runs the
// existing gaussian-major SFPU blend, then SPILLS R/G/B/T to fp32 L1 CBs. The
// spilled T tile is plain L1 memory, so ALL three TRISC threads read it and
// compute the SAME global max(T) -> a UNIFORM scalar break (front-to-back, once
// every pixel has T < eps the remaining splats add < eps everywhere, so the
// image is unchanged within the gate and we skip the deep tail). Small/medium
// tiles (num_g <= MB_EO_BLK) keep the original single-bracket path (no spill).
constexpr uint32_t CB_FB_R = 12;   // fp32 R accumulator spill
constexpr uint32_t CB_FB_G = 13;   // fp32 G accumulator spill
constexpr uint32_t CB_FB_B = 14;   // fp32 B accumulator spill
constexpr uint32_t CB_FB_T = 15;   // fp32 T (transmittance) spill — scanned for max
#ifndef MB_EO_BLK
#define MB_EO_BLK 768u
#endif
constexpr float MB_EO_EPS = 1e-4f;  // matches the CPU reference (max_t < 0.0001f)
#endif

// MEASUREMENT: per-tile-class blend split. Tiles with num_g<=FIT are served
// L1-resident by the reader (bucket path); num_g>FIT take the DRAM-gather
// fallback. Mirrors GSPLAT_TT_BUCKET_FIT (default 8192).
#ifndef MB_BUCKET_FIT
#define MB_BUCKET_FIT 8192u
#endif

constexpr uint32_t NUM_MB = 32;

// Volatile sink so profiling variants can't dead-code-eliminate the coeff loads.
volatile uint32_t g_prof_sink = 0;

// L1 read-visibility fence. On Blackhole L1 is a small write-THROUGH cache; the
// producer's CB-row stores reach L1, but THIS reader (the compute) may hold a
// stale cached line for the recycled CB slot address. A `fence` invalidates it
// so the freshly produced row is read coherently. == invalidate_l1_cache().
[[maybe_unused]] inline void mb_cb_consume_fence() {
    asm volatile("fence" ::: "memory");
}

#if defined(MB_BUCKET_AB_PROBE) && defined(TRISC_MATH)
// A/B PROBE (consumer side): verify the seq index + payload checksum the reader
// stamped into row[11]/row[12]. Runs on the MATH thread — the same thread whose
// row[] loads feed the blend — so it observes exactly the (possibly stale) bytes
// the blend math consumes. Per-tile counters; one summary line per dense tile.
inline void mb_ab_probe_check(const uint32_t* row, uint32_t g, uint32_t num_g) {
    static uint32_t _ab_t = 0;        // per-tile ordinal
    static uint32_t _ab_seq_mm = 0;   // rows whose seq != g  (STALE whole row)
    static uint32_t _ab_chk_mm = 0;   // rows whose checksum is wrong (TORN row)
    static uint32_t _ab_rows = 0;
    static uint32_t _ab_printed = 0;
    if (g == 0u) { _ab_seq_mm = 0; _ab_chk_mm = 0; _ab_rows = 0; }
    const uint32_t seq = row[11];
    const uint32_t chk = row[12];
    const uint32_t chk_ref = row[0] ^ row[1] ^ row[2] ^ row[3] ^ row[4] ^ row[5] ^
                             row[6] ^ row[7] ^ row[8] ^ row[9] ^ row[10];
    _ab_rows++;
    if (seq != g) _ab_seq_mm++;
    if (chk != chk_ref) _ab_chk_mm++;
    // Print the ACTUAL values for the first few records of the first dense tile:
    // is the read seq the slot's PREVIOUS occupant (off by a constant => the
    // producer's write has not landed / consumer reads an unwritten slot) or
    // garbage (torn)? Decides the exact mechanism + the right fix.
    static uint32_t _ab_val_t = 0xffffffffu;
    if (num_g > 64u && (_ab_val_t == 0xffffffffu || _ab_val_t == _ab_t) && g < 6u) {
        _ab_val_t = _ab_t;
        DPRINT << "ABVAL ti=" << _ab_t << " g=" << g << " seq=" << seq
               << " chk_ok=" << (uint32_t)(chk == chk_ref)
               << " r0=" << row[0] << " r10=" << row[10] << ENDL();
    }
    if (g + 1u == num_g) {
        if (num_g > 64u && _ab_printed < 40u) {
            _ab_printed++;
            DPRINT << "ABCK ti=" << _ab_t << " L=" << num_g
                   << " seqmm=" << _ab_seq_mm << " chkmm=" << _ab_chk_mm
                   << " rows=" << _ab_rows << ENDL();
        }
        _ab_t++;
    }
}
#endif

// DST slot bases (in dst_reg ix units; tile = 32 vectors).
constexpr uint32_t DR_R = 0 * 32;
constexpr uint32_t DR_G = 1 * 32;
constexpr uint32_t DR_B = 2 * 32;
constexpr uint32_t DR_T = 3 * 32;
constexpr uint32_t DR_X = 4 * 32;
constexpr uint32_t DR_Y = 5 * 32;

// Host microblock m -> dst_reg vector index within a tile slot.
//
// GEOMETRY (empirically verified on Blackhole via the VECMAP probe, fp32 dest
// + dst_full_sync_en): a single SFPU 32-lane vector dst_reg[V] is NOT a
// contiguous 4x8 raster block. It owns the 2 tile rows {2*(r/2), +1} at the
// 16 columns of parity (V&1), i.e. V = 2*(r/2) + (c&1). A contiguous 4x8
// microblock therefore spans HALF of four different vectors and cannot be
// addressed as one vector directly.
//
// We bridge this with a fixed host-side position permutation (see
// mb_perm_img_of_dev() in blend_device.cpp): the X/Y ramp upload places
// microblock m's 32 pixel coordinates into the raster slots that load into
// vector m, and the output is un-permuted on download. With that permutation
// in place the mapping host-m -> vector is the IDENTITY.
constexpr uint32_t MB_IX[NUM_MB] = {
    0,  1,  2,  3,  4,  5,  6,  7,
    8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31,
};

#ifdef TRISC_MATH
// One gaussian's contribution to a single microblock's 32-lane vector.
// IX is the dst_reg vector index (compile-time so SFPLOAD/SFPSTORE addresses
// are immediates). Coeff bits are runtime fp32 reinterpreted as uint32.
template <uint32_t IX>
inline void blend_one_gaussian_math(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t d_bits, uint32_t e_bits, uint32_t f_bits,
    uint32_t op_bits, uint32_t cr_bits, uint32_t cg_bits, uint32_t cb_bits) {
    using namespace sfpi;
    vFloat x = dst_reg[DR_X + IX];
    vFloat y = dst_reg[DR_Y + IX];

#if defined(MB_DEBUG_ANALYTIC) || defined(MB_DEBUG_RTCENTER)
    // Diagnostic: ignore A..C. Fixed-scale radial gaussian. ANALYTIC centers at
    // tile center (16,16) via compile-time constants. RTCENTER instead centers
    // at the gaussian mean via RUNTIME scalars d_bits(mx), e_bits(my): if the
    // blobs follow the gaussians coherently, runtime scalar->vFloat broadcast
    // works; if garbage, runtime scalar delivery in this sfpi block is broken.
    {
#if defined(MB_DEBUG_RTCENTER)
        vFloat dx = x - ckernel::sfpu::Converter::as_float(d_bits);
        vFloat dy = y - ckernel::sfpu::Converter::as_float(e_bits);
#else
        vFloat dx = x - vFloat(16.0f);
        vFloat dy = y - vFloat(16.0f);
#endif
        vFloat power = (dx * dx + dy * dy) * vFloat(-0.125f);
        vFloat zero = 0.0f;
        vec_min_max(power, zero);
        vFloat weight = ckernel::sfpu::_sfpu_exp_21f_bf16_</*is_fp32_dest_acc_en=*/true>(power);
        vFloat alpha = ckernel::sfpu::Converter::as_float(op_bits) * weight;
        vFloat clamp = 0.99f;
        vec_min_max(alpha, clamp);
        vFloat t = dst_reg[DR_T + IX];
        vFloat at = alpha * t;
        dst_reg[DR_R + IX] = vFloat(dst_reg[DR_R + IX]) + at * ckernel::sfpu::Converter::as_float(cr_bits);
        dst_reg[DR_G + IX] = vFloat(dst_reg[DR_G + IX]) + at * ckernel::sfpu::Converter::as_float(cg_bits);
        dst_reg[DR_B + IX] = vFloat(dst_reg[DR_B + IX]) + at * ckernel::sfpu::Converter::as_float(cb_bits);
        dst_reg[DR_T + IX] = t * (vFloat(1.0f) - alpha);
        (void)a_bits; (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits; (void)f_bits;
    }
    return;
#endif

    // Conic (centered) form: power = A dx^2 + B dx dy + C dy^2, dx=x-mx, dy=y-my.
    // (a_bits=A, b_bits=B, c_bits=C; d_bits=mx, e_bits=my; f_bits unused.)
    // Centered form avoids the catastrophic fp32 cancellation of the expanded
    // A x^2+...+F polynomial: near the gaussian center dx,dy are small, so all
    // terms stay small and `power` is computed accurately exactly where it
    // matters. The expanded form's ~1e8 terms cancelling to ~1 destroyed SFPU
    // precision and made the weight degenerate to binary noise.
    // Materialize each runtime scalar into a named vFloat BEFORE use — this is
    // the pattern the production SFPU activation functions (elu/celu/selu) use.
    // Using Converter::as_float(...) inline inside a larger expression makes
    // sfpi emit a non-uniform per-lane load (the value is not broadcast to all
    // 32 lanes), which is harmless for the gentle center subtraction but, once
    // multiplied by dx^2 (up to ~1e3), explodes into per-lane noise.
#if defined(MB_DEVCONIC)
    // DEVCONIC: a_bits/b_bits/c_bits carry the RAW 2D covariance {cov_a,cov_b,
    // cov_c}. Reproduce the host conic math bit-for-bit on the SFPU:
    //   det  = max(cov_a*cov_c - cov_b*cov_b, 1e-6)
    //   ci_a = cov_c/det, ci_b = -cov_b/det, ci_c = cov_a/det
    //   A = -0.5 ci_a, B = -ci_b (= cov_b/det), C = -0.5 ci_c
    // Reciprocal: 7-bit seed (approx_recip) + 2 Newton-Raphson iters (~24-bit),
    // using literal 2.0 so we do NOT depend on the programmed vConstFloatPrgm0.
    vFloat cov_a = ckernel::sfpu::Converter::as_float(a_bits);
    vFloat cov_b = ckernel::sfpu::Converter::as_float(b_bits);
    vFloat cov_c = ckernel::sfpu::Converter::as_float(c_bits);
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    vFloat det_floor = 1e-6f;
    vec_min_max(det_floor, det);  // det = max(det, 1e-6)
    vFloat inv = approx_recip(det);
    inv = inv * (vFloat(2.0f) - det * inv);
    inv = inv * (vFloat(2.0f) - det * inv);
    vFloat A = vFloat(-0.5f) * (cov_c * inv);
    vFloat B = cov_b * inv;
    vFloat C = vFloat(-0.5f) * (cov_a * inv);
#else
    vFloat A  = ckernel::sfpu::Converter::as_float(a_bits);
    vFloat B  = ckernel::sfpu::Converter::as_float(b_bits);
    vFloat C  = ckernel::sfpu::Converter::as_float(c_bits);
#endif
    vFloat mx = ckernel::sfpu::Converter::as_float(d_bits);
    vFloat my = ckernel::sfpu::Converter::as_float(e_bits);
    vFloat dx = x - mx;
    vFloat dy = y - my;
    vFloat power = A * (dx * dx);                                                    // A dx^2
    power = power + B * (dx * dy);                                                   // + B dx dy
    power = power + C * (dy * dy);                                                   // + C dy^2
    (void)f_bits;

    // weight = exp(min(power, 0))
    vFloat zero = 0.0f;
    vec_min_max(power, zero);  // power = min(power, 0)
    vFloat weight = ckernel::sfpu::_sfpu_exp_21f_bf16_</*is_fp32_dest_acc_en=*/true>(power);

#if defined(MB_DEBUG_WEIGHT)
    // Diagnostic: write this gaussian's real weight (from delivered A,B,C,mx,my)
    // to R/G/B, no composite. Combined with cnt clamped to 1 in process_microblock
    // this visualizes the FRONTMOST gaussian's falloff -> validates coeff delivery.
    dst_reg[DR_R + IX] = weight;
    dst_reg[DR_G + IX] = weight;
    dst_reg[DR_B + IX] = weight;
    (void)op_bits; (void)cr_bits; (void)cg_bits; (void)cb_bits;
    return;
#endif
#if defined(MB_DEBUG_SHOWA)
    // Diagnostic: visualize the delivered conic A (=-0.5 ci_a, negative) as a
    // FLAT per-microblock value via exp(A*scale) in [0,1]. A is constant across
    // the 32 lanes, so each microblock should be flat. Coherent patches that
    // track gaussian tightness => A reaches the kernel correctly; noise => not.
    {
#if defined(MB_DEBUG_SHOWA_CONST)
        vFloat aw = 0.5f;  // compile-time constant; output MUST be flat per mb
#else
        vFloat av = ckernel::sfpu::Converter::as_float(a_bits);  // A < 0
        vFloat ascaled = av * vFloat(2.0f);                      // still <= 0
        vFloat azero = 0.0f;
        vec_min_max(ascaled, azero);                             // min(2A,0)=2A
        vFloat aw = ckernel::sfpu::_sfpu_exp_21f_bf16_<true>(ascaled);  // exp(2A) in (0,1]
#endif
        dst_reg[DR_R + IX] = aw;
        dst_reg[DR_G + IX] = aw;
        dst_reg[DR_B + IX] = aw;
        (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits; (void)f_bits;
        (void)op_bits; (void)cr_bits; (void)cg_bits; (void)cb_bits; (void)power;
    }
    return;
#endif

    // alpha = min(opacity * weight, 0.99)
    vFloat alpha = ckernel::sfpu::Converter::as_float(op_bits) * weight;
    vFloat clamp = 0.99f;
    vec_min_max(alpha, clamp);  // alpha = min(alpha, 0.99)

    vFloat t = dst_reg[DR_T + IX];
    vFloat at = alpha * t;

    dst_reg[DR_R + IX] = vFloat(dst_reg[DR_R + IX]) + at * ckernel::sfpu::Converter::as_float(cr_bits);
    dst_reg[DR_G + IX] = vFloat(dst_reg[DR_G + IX]) + at * ckernel::sfpu::Converter::as_float(cg_bits);
    dst_reg[DR_B + IX] = vFloat(dst_reg[DR_B + IX]) + at * ckernel::sfpu::Converter::as_float(cb_bits);

    vFloat one_minus = vFloat(1.0f) - alpha;
    dst_reg[DR_T + IX] = t * one_minus;
}

#endif

// Dispatch one gaussian (coeffs in GPRs) to every microblock its mask selects,
// each blend as its OWN MATH() invocation (mirrors the working mb-major kernel:
// one blend_one_gaussian_math per MATH call). Compile-time unrolled; the runtime
// mask test skips untouched microblocks. Identity permutation: bit m -> vector m.
template <uint32_t M>
inline void dispatch_blend_guarded(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t fc, uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    if constexpr (M < NUM_MB) {
        if (mask & (1u << M)) {
            MATH((blend_one_gaussian_math<M>(a, b, c, d, e, fc, op, cr, cg, cbv)));
        }
        dispatch_blend_guarded<M + 1>(mask, a, b, c, d, e, fc, op, cr, cg, cbv);
    }
}

// Stream all of a tile's gaussian-major rows. Each row's 10 coeffs + mask word
// are read ONCE (not once per microblock), then dispatched to the masked
// microblocks. One start_/done_ for the whole tile (proven safe by VECMAP).
//
// mb_keep (default all-ones): per-microblock ACTIVE mask. Bit M clear => that
// 4x8 microblock has already saturated (all 32 px T < eps), so this gaussian's
// contribution to it is dropped (the §6 per-microblock transmittance early-out).
// With mb_keep == 0xFFFFFFFF this is a no-op AND => bit-identical to the
// un-gated path. The coeff row is still consumed (CB flow) even when fully
// masked off; only the SFPU exp/blend dispatch is skipped — and the blend is
// SFPU-bound, so skipping the saturated-microblock dispatch is the real lever.
inline void process_tile_gaussians(uint32_t num_g, uint32_t mb_keep = 0xFFFFFFFFu) {
    if (num_g == 0) {
        return;
    }
#if defined(MB_DEBUG_DPRINT)
    MATH((DPRINT << "G num_g=" << num_g << ENDL()));
#endif
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    uint32_t shown = 0;  // debug "frontmost-per-microblock" tracking only
    for (uint32_t g = 0; g < num_g; g++) {
        cb_wait_front(CB_MB_COEFF, 1);
        const uint32_t* row = reinterpret_cast<const uint32_t*>(get_tile_address(CB_MB_COEFF, 0));
#if defined(MB_BUCKET_CB_FENCE)
        // FIX (HYPOTHESIS A). The blend math reads each coeff row from L1 DIRECTLY
        // on the MATH thread, but CB flow control (cb_wait_front/cb_pop_front) runs
        // on UNPACK, which frees the slot the instant it has mailboxed MATH the
        // address — without waiting for the (slow, SFPU-bound) MATH read. On the
        // fast bucket feed the producer then recycles the slot to a LATER record
        // before MATH reads it => torn/stale rows (proven: seqmm~=rows). Two parts:
        //  (1) invalidate MATH's write-through L1 cache so it re-reads this slot
        //      fresh (the slot was last cached 8 records ago, depth=8); and
        //  (2) a bounded MATH->UNPACK back-pressure ack (below, around the row
        //      loads) so UNPACK cannot pop/free the slot until MATH has loaded it.
        MATH((mb_cb_consume_fence()));  // == invalidate_l1_cache() on Blackhole
#endif
#if defined(MB_DEBUG_PROF_NOREAD)
        // Profiling: skip the 10 coeff L1 loads (use constants) but keep the mask
        // read + the full 32-way dispatch + SFPU blend. Isolates SFPU/dispatch cost.
        const uint32_t mask = row[10] & mb_keep;
        const uint32_t a = 0xBDCCCCCDu, b = 0u, c = 0xBDCCCCCDu, d = 0x40800000u, e = 0x40000000u;
        const uint32_t fc = 0u, op = 0x3F000000u, cr = 0x3F000000u, cg = 0x3F000000u, cbv = 0x3F000000u;
        dispatch_blend_guarded<0>(mask, a, b, c, d, e, fc, op, cr, cg, cbv);
#elif defined(MB_DEBUG_PROF_NOBLEND)
        // Profiling: do the 10 coeff L1 loads + loop, but skip the SFPU dispatch.
        // Isolates the per-gaussian coeff-read + loop cost. Volatile sink defeats
        // dead-code elimination of the loads.
        g_prof_sink = row[0] + row[1] + row[2] + row[3] + row[4] +
                      row[5] + row[6] + row[7] + row[8] + row[9] + row[10];
        (void)shown;
#else
        const uint32_t a = row[0], b = row[1], c = row[2], d = row[3], e = row[4];
        const uint32_t fc = row[5], op = row[6], cr = row[7], cg = row[8], cbv = row[9];
        // Per-microblock early-out: drop this gaussian's contribution to any
        // microblock that has already saturated (mb_keep bit clear).
        uint32_t mask = row[10] & mb_keep;
#if defined(MB_COEFF_DEBUG)
        {
            static uint32_t dbg_cmp = 0;
            if (dbg_cmp < 4u) {
                dbg_cmp++;
                MATH((DPRINT << "CMPROW g=" << g << " a=" << F32(*reinterpret_cast<const float*>(&row[0]))
                      << " mxl=" << F32(*reinterpret_cast<const float*>(&row[3]))
                      << " op=" << F32(*reinterpret_cast<const float*>(&row[6]))
                      << " mask=" << mask << ENDL()));
            }
        }
#endif
#if defined(MB_DEBUG_WEIGHT) || defined(MB_DEBUG_SHOWA)
        mask &= ~shown;
        shown |= row[10];
#else
        (void)shown;
#endif
#if defined(MB_BUCKET_AB_PROBE)
        MATH((mb_ab_probe_check(row, g, num_g)));
#endif
#if defined(MB_BUCKET_CB_FENCE)
        // Pin all coeff loads into registers BEFORE the ack. They are plain
        // (non-volatile) L1 reads only consumed by the blend below; without this
        // the compiler could legally sink them past the ack, after which UNPACK
        // frees the slot and the producer overwrites it — reintroducing the race.
        asm volatile("" ::"r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(fc), "r"(op),
                     "r"(cr), "r"(cg), "r"(cbv), "r"(mask) : "memory");
        // Back-pressure ack: MATH has now loaded every word of this row into
        // registers, so the slot may be freed. UNPACK blocks on this hardware
        // mailbox before cb_pop_front, so it can never recycle a slot the producer
        // would overwrite before MATH read it. Bounded (one blocking mailbox
        // round-trip per row) — NOT a spin or a latency pad.
        MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, g + 1u)));
#endif
        dispatch_blend_guarded<0>(mask, a, b, c, d, e, fc, op, cr, cg, cbv);
#if defined(MB_BUCKET_CB_FENCE)
        UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId));
#endif
#endif
        cb_pop_front(CB_MB_COEFF, 1);
    }
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
}

#ifdef TRISC_MATH
// DIAGNOSTIC: write x/32 -> R, y/32 -> G for microblock vector IX. Confirms the
// ramp load + dst_reg addressing + pack path independent of the gaussian math.
template <uint32_t IX>
inline void debug_ramp_math() {
    using namespace sfpi;
    _llk_math_eltwise_unary_sfpu_start_(0);
    vFloat x = dst_reg[DR_X + IX];
    vFloat y = dst_reg[DR_Y + IX];
    dst_reg[DR_R + IX] = x * 0.03125f;  // /32
    dst_reg[DR_G + IX] = y * 0.03125f;
    dst_reg[DR_B + IX] = 0.0f;
    _llk_math_eltwise_unary_sfpu_done_();
}
#endif

template <uint32_t M>
inline void debug_all_microblocks() {
    if constexpr (M < NUM_MB) {
        MATH((debug_ramp_math<MB_IX[M]>()));
        debug_all_microblocks<M + 1>();
    }
}

#ifdef TRISC_MATH
// DIAGNOSTIC (VECMAP): write the RAW vector index i (0..31) as i/32 into the R
// slot vector i, under a SINGLE start_(0)/done_ bracket and LINEAR dst_reg[i]
// indexing (NO inc_dst_face_addr). Reading the raster R channel of any tile
// reveals exactly which (row,col) pixels each SFPU vector dst_reg[i] covers,
// and whether linear indexing crosses all 4 faces. Pure compile-time constants
// => no runtime-scalar broadcast involved. This isolates geometry+addressing.
template <uint32_t IX>
inline void vecmap_write() {
    using namespace sfpi;
    if constexpr (IX < 32) {
        dst_reg[DR_R + IX] = vFloat((float)IX * 0.03125f);  // i/32
        dst_reg[DR_G + IX] = vFloat(0.0f);
        dst_reg[DR_B + IX] = vFloat(0.0f);
        vecmap_write<IX + 1>();
    }
}
inline void debug_vecmap() {
    _llk_math_eltwise_unary_sfpu_start_(0);
    vecmap_write<0>();
    _llk_math_eltwise_unary_sfpu_done_();
}
#endif

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_blend_sfpu");
#ifdef MB_RESIDENT
    cb_wait_front(CB_CORE_TILES, 1);
    const uint32_t num_tiles =
        reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
    cb_pop_front(CB_CORE_TILES, 1);
#else
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);
#endif

    init_sfpu(CB_XRAMP, CB_COLOR_OUT);
    fill_tile_init();

    if (num_tiles == 0) {
        return;
    }

    // Ramps are constant across tiles; the reader streams them once per core.
    cb_wait_front(CB_XRAMP, 1);
    cb_wait_front(CB_YRAMP, 1);

#if defined(MB_BLEND_TAILSKIP)
    // Fixed L1 handshake base (CB_HS is never push/pop'd; for an unused CB the
    // rd_ptr == its configured base, i.e. the SAME L1 byte address the reader
    // gets via get_write_ptr(CB_HS)). get_tile_address mailbox-broadcasts the
    // UNPACK value so all three compute threads see the identical address.
    volatile uint32_t* hs = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_HS, 0));
#endif

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_MB_COUNTS, 1);
        // Gaussian-major: counts page slot 0 holds this tile's gaussian-row count.
        uint32_t num_g;
        {
            auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0));
            num_g = cptr[0];
        }

#if defined(MB_COEFF_DEBUG)
        {
            static uint32_t dbg_t = 0;
            if (dbg_t < 8u) {
                dbg_t++;
                auto xr = reinterpret_cast<volatile float*>(get_tile_address(CB_XRAMP, 0));
                auto yr = reinterpret_cast<volatile float*>(get_tile_address(CB_YRAMP, 0));
                MATH((DPRINT << "TILEDBG t=" << t << " num_g=" << num_g
                      << " xr0=" << F32(xr[0]) << " xr1=" << F32(xr[1])
                      << " xr8=" << F32(xr[8]) << " xr32=" << F32(xr[32])
                      << " yr0=" << F32(yr[0]) << " yr32=" << F32(yr[32]) << ENDL()));
            }
        }
#endif

#if defined(MB_BLEND_EARLYOUT)
        // ---- DENSE-TILE transmittance early-out (block spill/reload) ----------
        if (num_g > MB_EO_BLK) {
            uint32_t processed = 0;
            bool first = true;
            bool broke = false;
            // Per-microblock ACTIVE mask: bit M set => microblock M still has at
            // least one pixel with T >= eps. Cleared as each 4x8 microblock
            // saturates, so subsequent gaussians skip it (§6 per-microblock
            // transmittance early-out). When it reaches 0 the whole tile is done.
            uint32_t keep_mask = 0xFFFFFFFFu;
            while (processed < num_g && !broke) {
                uint32_t blk = num_g - processed;
                if (blk > MB_EO_BLK) blk = MB_EO_BLK;

                tile_regs_acquire();
                if (first) {
                    fill_tile(0, 0.0f);
                    fill_tile(1, 0.0f);
                    fill_tile(2, 0.0f);
                    fill_tile(3, 1.0f);
                } else {
                    // Reload R/G/B/T accumulators from the fp32 spill CBs.
                    copy_tile_to_dst_init_short(CB_FB_R); copy_tile(CB_FB_R, 0, 0);
                    copy_tile_to_dst_init_short(CB_FB_G); copy_tile(CB_FB_G, 0, 1);
                    copy_tile_to_dst_init_short(CB_FB_B); copy_tile(CB_FB_B, 0, 2);
                    copy_tile_to_dst_init_short(CB_FB_T); copy_tile(CB_FB_T, 0, 3);
                    cb_pop_front(CB_FB_R, 1); cb_pop_front(CB_FB_G, 1);
                    cb_pop_front(CB_FB_B, 1); cb_pop_front(CB_FB_T, 1);
                }
                copy_tile_to_dst_init_short(CB_XRAMP); copy_tile(CB_XRAMP, 0, 4);
                copy_tile_to_dst_init_short(CB_YRAMP); copy_tile(CB_YRAMP, 0, 5);

                // Blend `blk` rows into the still-ACTIVE microblocks only.
                process_tile_gaussians(blk, keep_mask);

                tile_regs_commit();
                tile_regs_wait();
                // Spill accumulators to fp32 CBs (so they survive tile_regs_release
                // and so T becomes plain L1 memory we can scan per-microblock).
                pack_reconfig_data_format(CB_FB_R);
                cb_reserve_back(CB_FB_R, 1); pack_tile(0, CB_FB_R); cb_push_back(CB_FB_R, 1);
                cb_reserve_back(CB_FB_G, 1); pack_tile(1, CB_FB_G); cb_push_back(CB_FB_G, 1);
                cb_reserve_back(CB_FB_B, 1); pack_tile(2, CB_FB_B); cb_push_back(CB_FB_B, 1);
                cb_reserve_back(CB_FB_T, 1); pack_tile(3, CB_FB_T); cb_push_back(CB_FB_T, 1);
                tile_regs_release();

                processed += blk;
                first = false;

                if (processed < num_g) {
                    // PER-MICROBLOCK max(T). The spilled T tile is in DEVICE
                    // raster order; device position p=(r*32+c) is owned by SFPU
                    // vector / microblock V = (r & ~1) | (c & 1) (the fixed ramp
                    // permutation mb_perm_img_of_dev: V = 2*(r/2) + (c&1)). Reduce
                    // T over each microblock's 32 pixels and clear its keep bit
                    // once they ALL fall below eps. Identical on every TRISC
                    // thread (same L1 bytes) => a uniform per-microblock decision.
                    cb_wait_front(CB_FB_T, 1);
                    mb_cb_consume_fence();
                    volatile float* tp =
                        reinterpret_cast<volatile float*>(get_tile_address(CB_FB_T, 0));
                    float mbmax[NUM_MB];
                    for (uint32_t m = 0; m < NUM_MB; m++) mbmax[m] = 0.0f;
                    for (uint32_t p = 0; p < 1024u; p++) {
                        const uint32_t r = p >> 5;
                        const uint32_t c = p & 31u;
                        const uint32_t V = (r & ~1u) | (c & 1u);
                        const float v = tp[p];
                        if (v > mbmax[V]) mbmax[V] = v;
                    }
                    uint32_t done = 0u;
                    for (uint32_t m = 0; m < NUM_MB; m++) {
                        if (mbmax[m] < MB_EO_EPS) done |= (1u << m);
                    }
                    keep_mask = ~done;
                    if (keep_mask == 0u) {
                        broke = true;
                        cb_pop_front(CB_FB_T, 1);  // R/G/B stay front for final pack
                    }
                }
            }
            // Drain any coeff rows the reader pushed past the break point.
#if defined(MB_BLEND_TAILSKIP)
            if (broke) {
                // Reader-side tail-skip: tell the reader to STOP emitting this
                // tile's (depth-sorted, far) tail, then drain exactly the rows it
                // actually pushed. Deadlock-free: only cb_wait_front when a row is
                // KNOWN pushed (processed < PUSHED); otherwise spin on REND. The
                // reader can only block on a full CB (=> PUSHED > processed => we
                // pop and unblock it), so it always reaches its break + sets REND.
                const uint32_t e = t + 1u;
                const uint32_t hb = hs_base(t & 1u);
                hs[HS_DONE_REQ] = e;
                mb_cb_consume_fence();
                for (;;) {
                    mb_cb_consume_fence();
                    const uint32_t rend   = hs[hb + HS_REND];
                    const uint32_t ep     = hs[hb + HS_EPOCH];
                    const uint32_t pushed = (ep == e) ? hs[hb + HS_PUSHED] : 0u;
                    if (processed < pushed) {
                        cb_wait_front(CB_MB_COEFF, 1);
                        cb_pop_front(CB_MB_COEFF, 1);
                        processed++;
                        continue;
                    }
                    if (rend == e) break;  // reader done this tile + we caught up
                }
            } else {
                for (; processed < num_g; processed++) {
                    cb_wait_front(CB_MB_COEFF, 1);
                    cb_pop_front(CB_MB_COEFF, 1);
                }
            }
#else
            for (; processed < num_g; processed++) {
                cb_wait_front(CB_MB_COEFF, 1);
                cb_pop_front(CB_MB_COEFF, 1);
            }
#endif
            // Final: reload R/G/B and pack bf16 color out for the writer.
            tile_regs_acquire();
            copy_tile_to_dst_init_short(CB_FB_R); copy_tile(CB_FB_R, 0, 0);
            copy_tile_to_dst_init_short(CB_FB_G); copy_tile(CB_FB_G, 0, 1);
            copy_tile_to_dst_init_short(CB_FB_B); copy_tile(CB_FB_B, 0, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(CB_COLOR_OUT);
            cb_reserve_back(CB_COLOR_OUT, 3);
            pack_tile(0, CB_COLOR_OUT);
            pack_tile(1, CB_COLOR_OUT);
            pack_tile(2, CB_COLOR_OUT);
            cb_push_back(CB_COLOR_OUT, 3);
            tile_regs_release();
            cb_pop_front(CB_FB_R, 1); cb_pop_front(CB_FB_G, 1); cb_pop_front(CB_FB_B, 1);
            if (!broke) cb_pop_front(CB_FB_T, 1);
            cb_pop_front(CB_MB_COUNTS, 1);
            continue;
        }
#endif

        tile_regs_acquire();

        // Init running state: R=G=B=0, T=1 over the whole tile.
        fill_tile(0, 0.0f);
        fill_tile(1, 0.0f);
        fill_tile(2, 0.0f);
        fill_tile(3, 1.0f);

        // Load tile-local coordinate ramps into DEST slots 4, 5.
        copy_tile_to_dst_init_short(CB_XRAMP);
        copy_tile(CB_XRAMP, 0, 4);
        copy_tile_to_dst_init_short(CB_YRAMP);
        copy_tile(CB_YRAMP, 0, 5);

#if defined(MB_DEBUG_VECMAP)
        MATH((debug_vecmap()));
        // Still drain CB_MB_COEFF so the reader doesn't deadlock.
        for (uint32_t i = 0; i < num_g; i++) {
            cb_wait_front(CB_MB_COEFF, 1);
            cb_pop_front(CB_MB_COEFF, 1);
        }
#elif defined(MB_DEBUG_RAMP)
        debug_all_microblocks<0>();
        for (uint32_t i = 0; i < num_g; i++) {
            cb_wait_front(CB_MB_COEFF, 1);
            cb_pop_front(CB_MB_COEFF, 1);
        }
#else
        // MEASUREMENT zones (axis B): compute SFPU-blend, one zone per tile, split
        // by class. process_tile_gaussians drains CB_MB_COEFF (cb_wait_front per
        // row), so this captures SFPU work + any back-pressure wait on the reader
        // emit. One zone per tile (cp_inb XOR cp_ovf) keeps the marker count
        // bounded; durations are summed in post-processing.
        if (num_g <= MB_BUCKET_FIT) {
            DeviceZoneScopedN("cp_inb");
            process_tile_gaussians(num_g);
        } else {
            DeviceZoneScopedN("cp_ovf");
            process_tile_gaussians(num_g);
        }
#endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        cb_pop_front(CB_MB_COUNTS, 1);
    }

    cb_pop_front(CB_XRAMP, 1);
    cb_pop_front(CB_YRAMP, 1);
}
