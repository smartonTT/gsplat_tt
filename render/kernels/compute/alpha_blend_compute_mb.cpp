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
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "../dataflow/dm_fp32.h"
#include "blend_t_live.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

#if defined(GSPLAT_TT_MB_STATS) || defined(GSPLAT_TT_MB_TILECYC)
#include "api/debug/dprint.h"
#endif

namespace {

constexpr uint32_t CB_XRAMP     = 0;   // fp32 tile-local x ramp (c + 0.5)
constexpr uint32_t CB_YRAMP     = 1;   // fp32 tile-local y ramp (r + 0.5)
constexpr uint32_t CB_MB_COUNTS = 3;   // 32 uint32 per tile (per-microblock count)
constexpr uint32_t CB_CORE_TILES = 7;  // MB_RESIDENT: tile count from reader (no host arg)
constexpr uint32_t CB_BUCKET_BULK = 12; // subchunk L1 records (slab carries mask in word3)
constexpr uint32_t CB_COLOR_OUT = 16;
constexpr uint32_t CB_T_RB = 2;        // iter 107: mid-accumulation T readback (bf16, 1 tile)

// MB_COUNTS flags (slot 1): bit0=emit_tile, bit1=continue_blend, bit2=l1_bulk.
constexpr uint32_t MB_FLAG_EMIT = 1u;
constexpr uint32_t MB_FLAG_CONTINUE = 2u;
constexpr uint32_t MB_FLAG_L1_BULK = 4u;
constexpr uint32_t MB_FLAG_DONE = 8u;   // task #60: reader ran out of tiles to claim

// Tiles with num_g<=FIT are served L1-resident by the reader (bucket path);
// num_g>FIT take the DRAM-gather fallback. Mirrors the host BUCKET_FIT (8192).
constexpr uint32_t MB_BUCKET_FIT = 8192u;

constexpr uint32_t NUM_MB = 32;

// L1 read-visibility fence. On Blackhole L1 is a small write-THROUGH cache; the
// producer's CB-row stores reach L1, but THIS reader (the compute) may hold a
// stale cached line for the recycled CB slot address. A `fence` invalidates it
// so the freshly produced row is read coherently. == invalidate_l1_cache().
inline void mb_cb_consume_fence() {
    asm volatile("fence" ::: "memory");
}

// DST slot bases (in dst_reg ix units; tile = 32 vectors).
constexpr uint32_t DR_R = 0 * 32;
constexpr uint32_t DR_G = 1 * 32;
constexpr uint32_t DR_B = 2 * 32;
constexpr uint32_t DR_T = 3 * 32;
constexpr uint32_t DR_X = 4 * 32;
constexpr uint32_t DR_Y = 5 * 32;

// Task #68: per-gaussian coefficient staging in DEST slot 6 (BLEND_COEF_DEST,
// host default ON). A runtime scalar reaches the SFPU as two SFPLOADIs whose
// instruction words TRISC1 builds in GPRs (zext.h/srli + lui + add + sw per
// half): 8 RISC instructions per scalar. The dispatch bodies re-materialized
// ~10 scalars per microblock (pair) dispatch, so 42% (pair) / 54% (single) of
// their RISC instructions went to that and TRISC1 was issue-bound (objdump,
// docs/blend-coef-dest-t68). Now each gaussian's scalars are built ONCE and
// SFPSTOREd to lane-broadcast vectors in slot 6; the dispatch reads them with
// SFPLOAD (one native instruction, immediate address). fp32 DEST round-trips
// exactly (same path as the R/G/B/T accumulators), so output is byte-identical.
constexpr uint32_t DR_S = 6 * 32;
constexpr uint32_t S_MX = 0, S_MY = 1, S_A = 2, S_B = 3, S_C = 4, S_OP = 5, S_FL = 6, S_CR = 7,
                   S_CG = 8, S_CB = 9, S_INV = 10;
#ifndef BLEND_COEF_DEST
#define BLEND_COEF_DEST 0
#endif
#if BLEND_COEF_DEST
#define BLEND_COEF(slot, bits) ((void)(bits), sfpi::vFloat(sfpi::dst_reg[DR_S + (slot)]))
#else
#define BLEND_COEF(slot, bits) ckernel::sfpu::Converter::as_float(bits)
#endif

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
// in place the mapping host-m -> vector is the IDENTITY (dispatch_blend_guarded
// blends microblock bit M directly into vector M).

// Per-pixel contribution floor (task #41, BLEND_PIXEL_FLOOR, host default ON): the
// GPU 3DGS rule alpha < floor -> skip, applied per pixel. The microblock mask is
// a conservative per-8x4-block superset, so without this a block where ANY pixel
// reaches the floor also keeps every sub-floor tail in its other 31 pixels. That
// is harmless at a 1/16384 floor, but at 1/255 the faint haze in front of thin
// structures is kept in some blocks and dropped in the next -> seams on
// microblock lines (task #28). Floor bits come from compute runtime-arg 1.
static uint32_t g_pixel_floor_bits = 0u;
#if defined(BLEND_PIXEL_FLOOR)
#define BLEND_APPLY_PIXEL_FLOOR(al)                                               \
    do {                                                                          \
        sfpi::vFloat fl_ = BLEND_COEF(S_FL, g_pixel_floor_bits);                   \
        v_if((al) < fl_) { (al) = 0.0f; }                                         \
        v_endif;                                                                  \
    } while (0)
#else
#define BLEND_APPLY_PIXEL_FLOOR(al) ((void)0)
#endif

// Task #146 (BLEND_CONST_HOIST, host default 1, needs BLEND_COEF_DEST): the
// bodies' constants are staged once per subchunk instead of built per body with
// 2 SFPLOADIs each. exp's ONE_LN2, c2, c1 live in the programmable const regs
// (L12-L14, MAD operands directly); c0 and the 0.99 alpha clamp in DEST slot 6
// (one SFPLOAD each; vec_min_max writes both args, so the clamp needs a copy
// anyway). blend_exp21f is _sfpu_exp_21f_bf16_<true> op for op minus the upper
// clamp to 255, which is dead: the input is min(power, 0) <= 0, so xlog2 <= 127.
#ifndef BLEND_CONST_HOIST
#define BLEND_CONST_HOIST 0
#endif
#if BLEND_CONST_HOIST && BLEND_COEF_DEST && defined(TRISC_MATH)
constexpr uint32_t S_C0 = 11, S_K99 = 12;
sfpi_inline sfpi::vFloat blend_exp21f(sfpi::vFloat val) {
    using namespace sfpi;
    vFloat xlog2 = val * vConstFloatPrgm0 + 127.f;
    vFloat threshold_low = 0.f;
    vec_min_max(threshold_low, xlog2);
    vInt z = ckernel::sfpu::_float_to_int32_for_exp_21f_(xlog2);
    vInt exponential_part = exexp(reinterpret<vFloat>(z), ExponentMode::NoDebias);
    vInt fractional_part = exman(reinterpret<vFloat>(z));
    vFloat frac = int32_to_float(fractional_part, RoundMode::NearestEven);
    vFloat t = frac * vConstFloatPrgm1 + vConstFloatPrgm2;  // c2 x + c1
    frac = frac * t + vFloat(dst_reg[DR_S + S_C0]);         // (.) x + c0
    return setexp(frac, exponential_part);
}
inline void blend_stage_hoist() {
    sfpi::vConstFloatPrgm0 = 1.4426950216293334961f;   // ONE_LN2
    sfpi::vConstFloatPrgm1 = 4.791750143340323e-15f;   // c2
    sfpi::vConstFloatPrgm2 = 7.839635491371155e-08f;   // c1
    sfpi::dst_reg[DR_S + S_C0] = 1.0017248f;
    sfpi::dst_reg[DR_S + S_K99] = 0.99f;
}
#define BLEND_EXP(v) blend_exp21f(v)
#define BLEND_K99() sfpi::vFloat(sfpi::dst_reg[DR_S + S_K99])
#else
#define BLEND_EXP(v) ckernel::sfpu::_sfpu_exp_21f_bf16_</*is_fp32_dest_acc_en=*/true>(v)
#define BLEND_K99() sfpi::vFloat(0.99f)
#endif

// Task #111 (lever 4 probe, host env GSPLAT_TT_BLEND_FPU_QF_ABL, default 0 =
// compiled out): 1 = the bodies skip the conic (dx, dy, A dx^2 + B dx dy +
// C dy^2) and SFPLOAD the x ramp as "power", i.e. the SFPU side of an ideal
// FPU quadratic form with the FPU work free. Timing only; output is wrong.
// Run with BLEND_T_PERIOD=0 in both arms: a fake power changes T saturation.
#ifndef BLEND_FPU_QF_ABL
#define BLEND_FPU_QF_ABL 0
#endif

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
#if BLEND_FPU_QF_ABL
    // Task #111 timing-only ablation: power arrives precomputed in DEST (what an
    // FPU quadratic form would leave the SFPU: one SFPLOAD). Output is wrong.
    (void)a_bits; (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits;
    vFloat power = dst_reg[DR_X + IX];
#else
    vFloat x = dst_reg[DR_X + IX];
    vFloat y = dst_reg[DR_Y + IX];

    // A1 (iter 111): a_bits/b_bits/c_bits now carry the PRE-FOLDED conic
    // {A,B,C}, hoisted once-per-gaussian into pfwc_compute.cpp (bit-identical to
    // the det/recip/-0.5-fold that USED to run here for every (gaussian ×
    // microblock) pair). The redundant SFPU recompute is gone — read straight.
    vFloat A = BLEND_COEF(S_A, a_bits);
    vFloat B = BLEND_COEF(S_B, b_bits);
    vFloat C = BLEND_COEF(S_C, c_bits);
    vFloat mx = BLEND_COEF(S_MX, d_bits);
    vFloat my = BLEND_COEF(S_MY, e_bits);
    vFloat dx = x - mx;
    vFloat dy = y - my;
    vFloat power = A * (dx * dx);                                                    // A dx^2
    power = power + B * (dx * dy);                                                   // + B dx dy
    power = power + C * (dy * dy);                                                   // + C dy^2
#endif
    (void)f_bits;

    // weight = exp(min(power, 0))
    vFloat zero = 0.0f;
    vec_min_max(power, zero);  // power = min(power, 0)
    vFloat weight = BLEND_EXP(power);

    // alpha = min(opacity * weight, 0.99)
    vFloat alpha = BLEND_COEF(S_OP, op_bits) * weight;
    vFloat clamp = BLEND_K99();
    vec_min_max(alpha, clamp);  // alpha = min(alpha, 0.99)
    BLEND_APPLY_PIXEL_FLOOR(alpha);

    vFloat t = dst_reg[DR_T + IX];
    vFloat at = alpha * t;

    dst_reg[DR_R + IX] = vFloat(dst_reg[DR_R + IX]) + at * BLEND_COEF(S_CR, cr_bits);
    dst_reg[DR_G + IX] = vFloat(dst_reg[DR_G + IX]) + at * BLEND_COEF(S_CG, cg_bits);
    dst_reg[DR_B + IX] = vFloat(dst_reg[DR_B + IX]) + at * BLEND_COEF(S_CB, cb_bits);

    vFloat one_minus = vFloat(1.0f) - alpha;
    dst_reg[DR_T + IX] = t * one_minus;
}

// CHEAP MICROBLOCK ILP (Track 1b): one gaussian's contribution to TWO covered
// microblocks (IXA, IXB), with the per-pair dependent chain issued STAGE-
// INTERLEAVED so the two INDEPENDENT microblock recurrences overlap and hide the
// SFPU op-latency stalls that dominate the blend (iter-115 ablation: blend is
// dependency-STALL-bound, NOT SFPU-throughput-bound — a cheaper exp does not
// help, but overlapping two independent chains fills the latency bubbles).
//
// CRITICAL: every intermediate stays in an LREG (vFloat local) — NOTHING spills
// to L1/DR_SCR (that spill is exactly what killed the iter-112 phasing). The two
// microblocks have INDEPENDENT accumulators (distinct DEST vectors IXA/IXB), so
// reordering ops across them cannot change any per-microblock result => output is
// BIT-IDENTICAL to the per-microblock full-chain form. Per-gaussian coeffs are
// re-materialized per stage (as_float == a cheap SFPLOADI immediate, no DEST
// traffic) to keep peak LREG pressure low (no spill). vec_min_max writes BOTH
// args (a=min,b=max), so each chain uses its OWN zero/clamp constant.
template <uint32_t IXA, uint32_t IXB>
inline void blend_pair_gaussian_math(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits,
    uint32_t d_bits, uint32_t e_bits, uint32_t f_bits,
    uint32_t op_bits, uint32_t cr_bits, uint32_t cg_bits, uint32_t cb_bits) {
    using namespace sfpi;
    (void)f_bits;

#if BLEND_FPU_QF_ABL
    (void)a_bits; (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits;
    vFloat pa = dst_reg[DR_X + IXA];
    vFloat pb = dst_reg[DR_X + IXB];
#else
    vFloat mx = BLEND_COEF(S_MX, d_bits);
    vFloat my = BLEND_COEF(S_MY, e_bits);
    vFloat dxa = vFloat(dst_reg[DR_X + IXA]) - mx;
    vFloat dxb = vFloat(dst_reg[DR_X + IXB]) - mx;
    vFloat dya = vFloat(dst_reg[DR_Y + IXA]) - my;
    vFloat dyb = vFloat(dst_reg[DR_Y + IXB]) - my;

    // power = A dx^2 + B dx dy + C dy^2 (both microblocks, interleaved).
    vFloat A = BLEND_COEF(S_A, a_bits);
    vFloat pa = A * (dxa * dxa);
    vFloat pb = A * (dxb * dxb);
    vFloat B = BLEND_COEF(S_B, b_bits);
    pa = pa + B * (dxa * dya);
    pb = pb + B * (dxb * dyb);
    vFloat C = BLEND_COEF(S_C, c_bits);
    pa = pa + C * (dya * dya);
    pb = pb + C * (dyb * dyb);
#endif

    // weight = exp(min(power, 0)) (own zero const per chain).
    vFloat zeroA = 0.0f;
    vFloat zeroB = 0.0f;
    vec_min_max(pa, zeroA);
    vec_min_max(pb, zeroB);
    vFloat wa = BLEND_EXP(pa);
    vFloat wb = BLEND_EXP(pb);

    // alpha = min(opacity * weight, 0.99) (own clamp const per chain).
    vFloat op = BLEND_COEF(S_OP, op_bits);
    vFloat aa = op * wa;
    vFloat ab = op * wb;
    vFloat clampA = BLEND_K99();
    vFloat clampB = BLEND_K99();
    vec_min_max(aa, clampA);
    vec_min_max(ab, clampB);
    BLEND_APPLY_PIXEL_FLOOR(aa);
    BLEND_APPLY_PIXEL_FLOOR(ab);

    // at = alpha * T.
    vFloat ta = dst_reg[DR_T + IXA];
    vFloat tb = dst_reg[DR_T + IXB];
    vFloat ata = aa * ta;
    vFloat atb = ab * tb;

    // R/G/B += at * color ; T *= (1 - alpha). Interleaved across A and B.
    vFloat cr = BLEND_COEF(S_CR, cr_bits);
    dst_reg[DR_R + IXA] = vFloat(dst_reg[DR_R + IXA]) + ata * cr;
    dst_reg[DR_R + IXB] = vFloat(dst_reg[DR_R + IXB]) + atb * cr;
    vFloat cg = BLEND_COEF(S_CG, cg_bits);
    dst_reg[DR_G + IXA] = vFloat(dst_reg[DR_G + IXA]) + ata * cg;
    dst_reg[DR_G + IXB] = vFloat(dst_reg[DR_G + IXB]) + atb * cg;
    vFloat cbc = BLEND_COEF(S_CB, cb_bits);
    dst_reg[DR_B + IXA] = vFloat(dst_reg[DR_B + IXA]) + ata * cbc;
    dst_reg[DR_B + IXB] = vFloat(dst_reg[DR_B + IXB]) + atb * cbc;
    dst_reg[DR_T + IXA] = ta * (vFloat(1.0f) - aa);
    dst_reg[DR_T + IXB] = tb * (vFloat(1.0f) - ab);
}

#if BLEND_COEF_DEST
// Build each runtime scalar once per gaussian and park it, lane-broadcast, in
// DEST slot 6. Stored in first-use order so each store is several instructions
// ahead of the dispatch's SFPLOAD of it.
inline void blend_stage_coeffs(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits, uint32_t d_bits, uint32_t e_bits,
    uint32_t op_bits, uint32_t cr_bits, uint32_t cg_bits, uint32_t cb_bits) {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
#if BLEND_FPU_QF_ABL
    (void)a_bits; (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits;
#else
    dst_reg[DR_S + S_MX] = Converter::as_float(d_bits);
    dst_reg[DR_S + S_MY] = Converter::as_float(e_bits);
    dst_reg[DR_S + S_A] = Converter::as_float(a_bits);
    dst_reg[DR_S + S_B] = Converter::as_float(b_bits);
    dst_reg[DR_S + S_C] = Converter::as_float(c_bits);
#endif
    dst_reg[DR_S + S_OP] = Converter::as_float(op_bits);
    dst_reg[DR_S + S_CR] = Converter::as_float(cr_bits);
    dst_reg[DR_S + S_CG] = Converter::as_float(cg_bits);
    dst_reg[DR_S + S_CB] = Converter::as_float(cb_bits);
}

#ifndef BLEND_SFPU_UNORM
#define BLEND_SFPU_UNORM 0
#endif
#if BLEND_SFPU_UNORM
// Task #80: UNORM16 op/colour decoded on the SFPU instead of the RISC. The
// reference is fl((float)q * fl(1/65535)): q (< 2^16) loads with one SFPLOADI
// (USHORT), converts exactly, and one fp32 multiply by the staged 1/65535 rounds
// it (SFPMAD, nearest-even). Replaces ~22 RISC insns + 2 SFPLOADIs per value.
// BLEND_SFPU_UNORM=2 is a check mode: stage the RISC decode as before, and if
// the SFPU decode differs in any bit, poison microblock 0's R (visible in md5).
inline sfpi::vFloat unorm16_sfpu(uint32_t q) {
    return sfpi::int32_to_float(sfpi::vInt(static_cast<uint16_t>(q)), sfpi::RoundMode::NearestEven) *
           sfpi::vFloat(sfpi::dst_reg[DR_S + S_INV]);
}
#if BLEND_SFPU_UNORM == 2
inline void unorm16_sfpu_check(uint32_t q, uint32_t ref_bits) {
    using namespace sfpi;
    vInt got = reinterpret<vInt>(unorm16_sfpu(q));
    v_if(got != vInt(ref_bits)) { dst_reg[DR_R] = 1.0e30f; }
    v_endif;
}
#endif
inline void blend_stage_coeffs_q(
    uint32_t a_bits, uint32_t b_bits, uint32_t c_bits, uint32_t d_bits, uint32_t e_bits,
    uint32_t w6, uint32_t w7) {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
#if BLEND_FPU_QF_ABL
    (void)a_bits; (void)b_bits; (void)c_bits; (void)d_bits; (void)e_bits;
#else
    dst_reg[DR_S + S_MX] = Converter::as_float(d_bits);
    dst_reg[DR_S + S_MY] = Converter::as_float(e_bits);
    dst_reg[DR_S + S_A] = Converter::as_float(a_bits);
    dst_reg[DR_S + S_B] = Converter::as_float(b_bits);
    dst_reg[DR_S + S_C] = Converter::as_float(c_bits);
#endif
    dst_reg[DR_S + S_OP] = unorm16_sfpu(w6 & 0xffffu);
    dst_reg[DR_S + S_CR] = unorm16_sfpu(w6 >> 16);
    dst_reg[DR_S + S_CG] = unorm16_sfpu(w7 & 0xffffu);
    dst_reg[DR_S + S_CB] = unorm16_sfpu(w7 >> 16);
}
// Task #146 (BLEND_RAW_STAGE, host default 0: -0.23 ms, under the 0.3 ms bar): the same staging as
// blend_stage_coeffs_q, written as raw instructions so the RISC side is
// shorter: the two SFPLOADI opcode words stay in registers (the compiler
// rebuilt them with a lui per half), and 1/65535 is loaded into L1 once per
// record instead of once per value. Same SFPU ops on the same values in the
// same order per value: fp32 = SFPLOADI USHORT lo + HI16_ONLY hi, SFPSTORE;
// UNORM16 = SFPLOADI USHORT q, SFPCAST (int -> fp32 RNE), SFPMUL by 1/65535,
// SFPSTORE. The UNORM values alternate L0/L2 so each cast is one instruction
// ahead of its multiply, as in the compiled form. LREGs are free here (the
// bodies keep nothing live across records).
#ifndef BLEND_RAW_STAGE
#define BLEND_RAW_STAGE 0
#endif
#if BLEND_RAW_STAGE && !BLEND_FPU_QF_ABL
#define BLEND_USE_RAW_STAGE 1
inline void blend_stage_coeffs_raw(const uint32_t* rec, uint32_t klo, uint32_t khi, uint32_t klo2) {
    volatile uint32_t* ib = ckernel::instrn_buffer;
    const uint32_t a = rec[0], b = rec[1], c = rec[2], d = rec[4], e = rec[5];
    const uint32_t w6 = rec[6], w7 = rec[7];
    constexpr uint32_t A0 = (DR_S) * 2u;  // SFPLOAD/SFPSTORE address of slot index 0
    ib[0] = (d & 0xffffu) + klo; ib[0] = (d >> 16) + khi; TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_MX);
    ib[0] = (e & 0xffffu) + klo; ib[0] = (e >> 16) + khi; TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_MY);
    ib[0] = (a & 0xffffu) + klo; ib[0] = (a >> 16) + khi; TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_A);
    ib[0] = (b & 0xffffu) + klo; ib[0] = (b >> 16) + khi; TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_B);
    ib[0] = (c & 0xffffu) + klo; ib[0] = (c >> 16) + khi; TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_C);
    TTI_SFPLOAD(1, 0, 7, A0 + 2u * S_INV);
    ib[0] = (w6 & 0xffffu) + klo; TTI_SFPCAST(0, 0, 0);
    ib[0] = (w6 >> 16) + klo2; TTI_SFPMUL(0, 1, 9, 0, 0); TTI_SFPCAST(2, 2, 0);
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_OP);
    ib[0] = (w7 & 0xffffu) + klo; TTI_SFPMUL(2, 1, 9, 2, 0); TTI_SFPCAST(0, 0, 0);
    TTI_SFPSTORE(2, 0, 7, A0 + 2u * S_CR);
    ib[0] = (w7 >> 16) + klo2; TTI_SFPMUL(0, 1, 9, 0, 0); TTI_SFPCAST(2, 2, 0);
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_CG);
    TTI_SFPMUL(2, 1, 9, 2, 0);
    TTI_SFPSTORE(2, 0, 7, A0 + 2u * S_CB);
}
#else
#define BLEND_USE_RAW_STAGE 0
#endif
// Bits of fl(1/65535) (0x37800080), staged once per subchunk like the floor.
inline void blend_stage_inv() {
    sfpi::dst_reg[DR_S + S_INV] = ckernel::sfpu::Converter::as_float(0x37800080u);
}
#endif

// The pixel floor is constant per launch; slot 6 survives the non-zeroing T
// readback and the subchunks of a tile, and is rewritten per subchunk call.
inline void blend_stage_floor() {
    sfpi::dst_reg[DR_S + S_FL] = ckernel::sfpu::Converter::as_float(g_pixel_floor_bits);
}
#endif

#endif

// Dispatch one gaussian (coeffs in GPRs) to every microblock its mask selects,
// CHEAP MICROBLOCK ILP (Track 1b). SINGLE mask-scan: the 32-bit live-microblock
// mask is examined ONCE here, two bits (one PAIR of adjacent microblock vectors)
// per recursion step — NOT re-scanned per phase (that 4x re-traversal + DR_SCR
// spill is what made the iter-112 phasing a net regression). For each pair:
//   - both covered  -> blend_pair_gaussian_math<A,B> issues the two INDEPENDENT
//                       dependent chains STAGE-INTERLEAVED so the SFPU op-latency
//                       stalls of one microblock are hidden by the other's ops;
//   - only one      -> blend_one_gaussian_math<A|B> (no wasted SFPU work, no ILP
//                       — but no penalty vs the old per-microblock form either);
//   - neither       -> skipped.
// Everything is compile-time unrolled (SFPLOAD/SFPSTORE addresses must be
// immediates); the runtime mask test selects the path. Each microblock keeps its
// own DEST accumulators, so pairing/interleaving is BIT-IDENTICAL. Identity
// permutation: bit m -> vector m.
// Task #78 timing-only ablation (host env GSPLAT_TT_BLEND_ABL, default 0 =
// compiled out). 1 = skip the SFPU blend bodies (per-record cost only; output
// is wrong). 2 = pad each pair body with 24 SFPNOPs and each single with 12
// (output unchanged): the cost of the issue slots a const-load hoist would free.
// Task #80 (floor attribution, output wrong for all three): 3 = 1 + skip the
// UNORM decode and coefficient staging (loop + mask read + T readbacks only);
// 4 = skip the whole per-record loop (reader/CB handshake floor); 5 = 1 but keep
// the 16-step mask walk (one nop per taken branch), so a5 - a1 = walk cost.
// Task #83: fine per-tile zones for floor attribution (host env
// GSPLAT_TT_BLEND_PROF=1; compiled out by default).
// Task #172: GSPLAT_TT_BLEND_PROF=2 swaps the (near-zero) wait zones for the
// per-tile fixed-cost sub-zones (cmp_stage, cmp_trb, cmp_sc_tail; cmp_init and
// cmp_emit at both levels) so one launch stays inside the 250-marker buffer.
#if defined(BLEND_PROF) && BLEND_PROF
#define BLEND_PZ(name) DeviceZoneScopedN(name)
#else
#define BLEND_PZ(name) ((void)0)
#endif
#if defined(BLEND_PROF) && BLEND_PROF == 1
#define BLEND_PZ_WAIT(name) DeviceZoneScopedN(name)
#else
#define BLEND_PZ_WAIT(name) ((void)0)
#endif
#if defined(BLEND_PROF) && BLEND_PROF >= 2
#define BLEND_PZ_SUB(name) DeviceZoneScopedN(name)
#else
#define BLEND_PZ_SUB(name) ((void)0)
#endif

#ifndef BLEND_ABL
#define BLEND_ABL 0
#endif
#if BLEND_ABL == 2 && defined(TRISC_MATH)
template <uint32_t N>
inline void blend_abl_pad() {
    if constexpr (N > 0) {
        TTI_SFPNOP;
        blend_abl_pad<N - 1>();
    }
}
#define BLEND_ABL_PAD(n) MATH((blend_abl_pad<n>()))
#else
#define BLEND_ABL_PAD(n) ((void)0)
#endif

template <uint32_t J>
inline void dispatch_blend_pairs(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e,
    uint32_t fc, uint32_t op, uint32_t cr, uint32_t cg, uint32_t cbv) {
    if constexpr (J < (NUM_MB / 2)) {
        constexpr uint32_t A = 2u * J;
        constexpr uint32_t B = 2u * J + 1u;
        const uint32_t pm = (mask >> (2u * J)) & 3u;
#if BLEND_ABL == 0 || BLEND_ABL == 2
        if (pm == 3u) {
            MATH((blend_pair_gaussian_math<A, B>(a, b, c, d, e, fc, op, cr, cg, cbv)));
            BLEND_ABL_PAD(24);
        } else if (pm == 1u) {
            MATH((blend_one_gaussian_math<A>(a, b, c, d, e, fc, op, cr, cg, cbv)));
            BLEND_ABL_PAD(12);
        } else if (pm == 2u) {
            MATH((blend_one_gaussian_math<B>(a, b, c, d, e, fc, op, cr, cg, cbv)));
            BLEND_ABL_PAD(12);
        }
#elif BLEND_ABL == 5 && defined(TRISC_MATH)
        if (pm == 3u) {
            asm volatile("nop");
        } else if (pm == 1u) {
            asm volatile("nop; nop");
        } else if (pm == 2u) {
            asm volatile("nop; nop; nop");
        }
#else
        (void)pm;
#endif
        dispatch_blend_pairs<J + 1>(mask, a, b, c, d, e, fc, op, cr, cg, cbv);
    }
}

// Task #80: jump-table mask walk (host env GSPLAT_TT_BLEND_JUMP_WALK, default 1;
// needs BLEND_COEF_DEST, so the bodies take no arguments). The recursive walk
// above tests all 16 pairs with compare/branch steps (2.6 ms/view measured with
// empty bodies). This one visits only the pairs with a set bit: ctz finds the
// lowest one, the 2-bit pair value picks the out-of-line body from a table.
// Same bodies, same ascending pair order -> bit-identical.
#ifndef BLEND_JUMP_WALK
#define BLEND_JUMP_WALK 0
#endif
#if BLEND_JUMP_WALK && BLEND_COEF_DEST && BLEND_ABL == 0 && defined(TRISC_MATH)
#define BLEND_USE_JUMP_WALK 1
// Task #219 (t205 lever; host env GSPLAT_TT_BLEND_SCHED, default 0): the bodies
// below as raw SFPU instructions in a hand schedule. Every op is the compiled
// body's op (opcode, immediate and mode as in docs/blend-dispatch-t189/out/
// trisc1-default.dis); only the order and the LREGs differ, so that no op reads
// the result of the MAD-class op right before it (a 1-cycle stall; the compiled
// single has 11, the pair 22, these have 0). Bit-identical: docs/blend-loop-
// model-t205/schedule.py runs every body against the compiled one, and
// docs/blend-sched-t219/check_cpp.py checks this source against those listings.
// 1 = F: straight-line bodies (single 60 ops, pair 109).
// 2 = A2: singles push only x/y, the middle 27 ops and the D steps; the front 13
//     ops and the tail 18 ops (+ SETRWC D=0) replay from the TRISC1 replay
//     buffer, recorded per records loop by blend_sched_record(). A pair is two
//     A2 singles. The tail addresses are relative to RWC D, which is 0 outside.
// BS_* take their operands in disassembly order (destination first). LREG 9 is
// 0.0 (the swaps park their dead output there; writes to it are dropped), 10 is
// 1.0, 11 is -1.0, 12-14 are the exp constants of blend_stage_hoist.
#ifndef BLEND_SCHED
#define BLEND_SCHED 0
#endif
#if BLEND_SCHED
#if !BLEND_CONST_HOIST || !defined(BLEND_PIXEL_FLOOR) || BLEND_FPU_QF_ABL
#error "BLEND_SCHED bodies are the CONST_HOIST + PIXEL_FLOOR bodies"
#endif
#define BS_V(r, i) (2u * ((r) + (i)))  // SFPLOAD/SFPSTORE address of dst_reg[r + i]
#define BS_S(s) BS_V(DR_S, s)
#define BS_LD(l, a, m, am) TTI_SFPLOAD(l, m, am, a)
#define BS_ST(l, a, m, am) TTI_SFPSTORE(l, m, am, a)
#define BS_MAD(d, a, b, c, m) TTI_SFPMAD(a, b, c, d, m)
#define BS_MUL(d, a, b, c, m) TTI_SFPMUL(a, b, c, d, m)
#define BS_ADD(d, a, b, c, m) TTI_SFPADD(a, b, c, d, m)
#define BS_ADDI(d, imm, m) TTI_SFPADDI(imm, d, m)
#define BS_SWAP(d, c, m) TTI_SFPSWAP(0, c, d, m)
#define BS_EXEXP(d, c, m) TTI_SFPEXEXP(0, c, d, m)
#define BS_EXMAN(d, c, m) TTI_SFPEXMAN(0, c, d, m)
#define BS_SHFT(d, c, imm, m) TTI_SFPSHFT(imm, c, d, m)
#define BS_CAST(d, c, m) TTI_SFPCAST(c, d, m)
#define BS_SETEXP(d, c, imm, m) TTI_SFPSETEXP(imm, c, d, m)
#define BS_SETCC(c, imm, m) TTI_SFPSETCC(imm, c, 0, m)
#define BS_MOV(d, c, m) TTI_SFPMOV(0, c, d, m)
#define BS_ENCC(imm, m) TTI_SFPENCC(imm, 0, 0, m)
#define BS_NOP() TTI_SFPNOP

// Single body = x/y, front, middle, tail (schedule.py SCHED, FRONT, MIDDLE, TAIL).
template <uint32_t IX>
[[gnu::always_inline]] inline void bs_xy() {
    BS_LD(2, BS_V(DR_X, IX), 0, 7);
    BS_LD(1, BS_V(DR_Y, IX), 0, 7);
}
// power = A dx^2 + B dx dy + C dy^2 (x in L2, y in L1).
[[gnu::always_inline]] inline void bs_front() {
    BS_LD(6, BS_S(S_MX), 0, 7);
    BS_LD(5, BS_S(S_MY), 0, 7);
    BS_ADD(2, 10, 2, 6, 2);
    BS_ADD(1, 10, 1, 5, 2);
    BS_MUL(5, 2, 2, 9, 0);
    BS_LD(4, BS_S(S_A), 0, 7);
    BS_MUL(4, 4, 5, 9, 0);
    BS_MUL(2, 2, 1, 9, 0);
    BS_LD(3, BS_S(S_B), 0, 7);
    BS_MAD(2, 3, 2, 4, 0);
    BS_MUL(1, 1, 1, 9, 0);
    BS_LD(0, BS_S(S_C), 0, 7);
    BS_MAD(0, 0, 1, 2, 0);
}
// weight = exp21f(min(power, 0)), alpha = min(op * weight, 0.99), colours.
[[gnu::always_inline]] inline void bs_middle() {
    BS_NOP();
    BS_SWAP(0, 9, 1);
    BS_NOP();
    BS_MUL(0, 0, 12, 9, 0);
    BS_LD(4, BS_S(S_C0), 0, 7);
    BS_ADDI(0, 17150, 0);
    BS_NOP();
    BS_SWAP(9, 0, 1);
    BS_NOP();
    BS_EXEXP(1, 0, 0);
    BS_EXMAN(0, 0, 0);
    BS_SHFT(0, 1, 0x000, 0);
    BS_EXEXP(1, 0, 1);
    BS_EXMAN(0, 0, 1);
    BS_CAST(0, 0, 0);
    BS_MAD(2, 0, 13, 14, 0);
    BS_LD(5, BS_S(S_OP), 0, 7);
    BS_MAD(0, 0, 2, 4, 0);
    BS_LD(6, BS_S(S_K99), 0, 7);
    BS_SETEXP(1, 0, 0x000, 0);
    BS_MUL(0, 5, 1, 9, 0);
    BS_LD(7, BS_S(S_FL), 0, 7);
    BS_SWAP(0, 6, 1);
    BS_NOP();
    BS_LD(3, BS_S(S_CR), 0, 7);
    BS_LD(4, BS_S(S_CG), 0, 7);
    BS_LD(5, BS_S(S_CB), 0, 7);
}
// Pixel floor, R/G/B += alpha T colour, T *= 1 - alpha. AT/AR/AG/AB: addresses
// of T/R/G/B (absolute in F, relative to RWC D in the A2 replay).
template <uint32_t AT, uint32_t AR, uint32_t AG, uint32_t AB>
[[gnu::always_inline]] inline void bs_tail() {
    BS_MAD(1, 7, 11, 0, 0);
    BS_LD(2, AT, 0, 7);
    BS_SETCC(1, 0x000, 0);
    BS_MOV(0, 9, 0);
    BS_ENCC(0x003, 10);
    BS_MUL(1, 0, 2, 9, 0);
    BS_ADD(0, 10, 10, 0, 2);
    BS_LD(6, AR, 0, 7);
    BS_MAD(6, 1, 3, 6, 0);
    BS_MUL(0, 2, 0, 9, 0);
    BS_ST(6, AR, 0, 7);
    BS_LD(7, AG, 0, 7);
    BS_MAD(7, 1, 4, 7, 0);
    BS_LD(6, AB, 0, 7);
    BS_ST(7, AG, 0, 7);
    BS_MAD(6, 1, 5, 6, 0);
    BS_ST(0, AT, 0, 7);
    BS_ST(6, AB, 0, 7);
}
template <uint32_t IX>
[[gnu::always_inline]] inline void sched_single() {
    bs_xy<IX>();
    bs_front();
    bs_middle();
    bs_tail<BS_V(DR_T, IX), BS_V(DR_R, IX), BS_V(DR_G, IX), BS_V(DR_B, IX)>();
}
// Pair body (schedule.py PAIR): the single's steps for microblocks IXA (chain a)
// and IXB (chain b) interleaved, b a few ops behind a. They share the MX MY A B C
// OP CR CG CB loads; C0, K99 and FL are loaded per chain, as compiled.
template <uint32_t IXA, uint32_t IXB>
[[gnu::always_inline]] inline void sched_pair() {
    BS_LD(0, BS_S(S_MX), 0, 7);
    BS_LD(1, BS_S(S_MY), 0, 7);
    BS_LD(2, BS_V(DR_X, IXA), 0, 7);
    BS_LD(3, BS_V(DR_Y, IXA), 0, 7);
    BS_ADD(2, 10, 2, 0, 2);
    BS_LD(4, BS_V(DR_X, IXB), 0, 7);
    BS_ADD(3, 10, 3, 1, 2);
    BS_ADD(4, 10, 4, 0, 2);
    BS_LD(0, BS_V(DR_Y, IXB), 0, 7);
    BS_MUL(5, 2, 2, 9, 0);
    BS_ADD(0, 10, 0, 1, 2);
    BS_LD(1, BS_S(S_A), 0, 7);
    BS_MUL(5, 1, 5, 9, 0);
    BS_MUL(6, 4, 4, 9, 0);
    BS_MUL(2, 2, 3, 9, 0);
    BS_MUL(6, 1, 6, 9, 0);
    BS_LD(1, BS_S(S_B), 0, 7);
    BS_MAD(2, 1, 2, 5, 0);
    BS_MUL(4, 4, 0, 9, 0);
    BS_MUL(3, 3, 3, 9, 0);
    BS_MAD(4, 1, 4, 6, 0);
    BS_LD(1, BS_S(S_C), 0, 7);
    BS_MAD(2, 1, 3, 2, 0);
    BS_MUL(0, 0, 0, 9, 0);
    BS_SWAP(2, 9, 1);
    BS_NOP();
    BS_MAD(0, 1, 0, 4, 0);
    BS_MUL(2, 2, 12, 9, 0);
    BS_SWAP(0, 9, 1);
    BS_NOP();
    BS_ADDI(2, 17150, 0);
    BS_MUL(0, 0, 12, 9, 0);
    BS_NOP();
    BS_SWAP(9, 2, 1);
    BS_NOP();
    BS_ADDI(0, 17150, 0);
    BS_EXEXP(1, 2, 0);
    BS_EXMAN(2, 2, 0);
    BS_NOP();
    BS_SWAP(9, 0, 1);
    BS_NOP();
    BS_SHFT(2, 1, 0x000, 0);
    BS_EXEXP(3, 0, 0);
    BS_EXEXP(1, 2, 1);
    BS_EXMAN(0, 0, 0);
    BS_EXMAN(2, 2, 1);
    BS_SHFT(0, 3, 0x000, 0);
    BS_CAST(2, 2, 0);
    BS_EXEXP(3, 0, 1);
    BS_MAD(4, 2, 13, 14, 0);
    BS_EXMAN(0, 0, 1);
    BS_LD(5, BS_S(S_C0), 0, 7);
    BS_MAD(2, 2, 4, 5, 0);
    BS_CAST(0, 0, 0);
    BS_SETEXP(1, 2, 0x000, 0);
    BS_MAD(4, 0, 13, 14, 0);
    BS_LD(5, BS_S(S_C0), 0, 7);
    BS_MAD(0, 0, 4, 5, 0);
    BS_LD(4, BS_S(S_OP), 0, 7);
    BS_SETEXP(3, 0, 0x000, 0);
    BS_MUL(2, 4, 1, 9, 0);
    BS_MUL(0, 4, 3, 9, 0);
    BS_LD(1, BS_S(S_K99), 0, 7);
    BS_SWAP(2, 1, 1);
    BS_NOP();
    BS_LD(3, BS_S(S_K99), 0, 7);
    BS_SWAP(0, 3, 1);
    BS_NOP();
    BS_LD(1, BS_S(S_FL), 0, 7);
    BS_MAD(1, 1, 11, 2, 0);
    BS_LD(3, BS_S(S_FL), 0, 7);
    BS_SETCC(1, 0x000, 0);
    BS_MOV(2, 9, 0);
    BS_ENCC(0x003, 10);
    BS_MAD(3, 3, 11, 0, 0);
    BS_LD(4, BS_V(DR_T, IXA), 0, 7);
    BS_SETCC(3, 0x000, 0);
    BS_MOV(0, 9, 0);
    BS_ENCC(0x003, 10);
    BS_MUL(1, 2, 4, 9, 0);
    BS_LD(5, BS_V(DR_T, IXB), 0, 7);
    BS_ADD(2, 10, 10, 2, 2);
    BS_MUL(3, 0, 5, 9, 0);
    BS_LD(6, BS_S(S_CR), 0, 7);
    BS_ADD(0, 10, 10, 0, 2);
    BS_LD(7, BS_V(DR_R, IXA), 0, 7);
    BS_MAD(7, 1, 6, 7, 0);
    BS_MUL(4, 4, 2, 9, 0);
    BS_ST(7, BS_V(DR_R, IXA), 0, 7);
    BS_LD(7, BS_V(DR_R, IXB), 0, 7);
    BS_MAD(7, 3, 6, 7, 0);
    BS_ST(4, BS_V(DR_T, IXA), 0, 7);
    BS_ST(7, BS_V(DR_R, IXB), 0, 7);
    BS_MUL(5, 5, 0, 9, 0);
    BS_LD(6, BS_S(S_CG), 0, 7);
    BS_ST(5, BS_V(DR_T, IXB), 0, 7);
    BS_LD(7, BS_V(DR_G, IXA), 0, 7);
    BS_MAD(7, 1, 6, 7, 0);
    BS_LD(5, BS_V(DR_G, IXB), 0, 7);
    BS_ST(7, BS_V(DR_G, IXA), 0, 7);
    BS_MAD(5, 3, 6, 5, 0);
    BS_LD(6, BS_S(S_CB), 0, 7);
    BS_ST(5, BS_V(DR_G, IXB), 0, 7);
    BS_LD(7, BS_V(DR_B, IXA), 0, 7);
    BS_MAD(7, 1, 6, 7, 0);
    BS_LD(5, BS_V(DR_B, IXB), 0, 7);
    BS_MAD(5, 3, 6, 5, 0);
    BS_ST(7, BS_V(DR_B, IXA), 0, 7);
    BS_ST(5, BS_V(DR_B, IXB), 0, 7);
}
#if BLEND_SCHED == 2
// D += N in INCRWC steps (the field is 4 bits; even steps of <= 14).
template <uint32_t N>
[[gnu::always_inline]] inline void bs_incr_d() {
    if constexpr (N > 0u) {
        TTI_INCRWC(0, (N > 14u ? 14u : N), 0, 0);
        bs_incr_d<(N > 14u ? N - 14u : 0u)>();
    }
}
template <uint32_t IX>
[[gnu::always_inline]] inline void sched_single_a2() {
    bs_xy<IX>();
    lltt::replay(0, 13);  // bs_front
    bs_middle();
    bs_incr_d<BS_V(0u, IX)>();
    lltt::replay(13, 19);  // bs_tail relative to D, then D = 0
}
// Replay slots 0-12: front; 13-31: tail at D + (T, R, G, B of microblock 0),
// then SETRWC D=0. Called before each records loop: other TRISC1 code (the
// fill_tile zero fill at tile start) records its own ops into slots 0-3.
inline void blend_sched_record() {
    lltt::record<lltt::NoExec>(0, 13);
    bs_front();
    lltt::record<lltt::NoExec>(13, 19);
    bs_tail<BS_V(DR_T, 0u), BS_V(DR_R, 0u), BS_V(DR_G, 0u), BS_V(DR_B, 0u)>();
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}
#endif
#endif  // BLEND_SCHED
template <uint32_t J, uint32_t PM>
__attribute__((noinline)) void blend_pair_body() {
#if BLEND_SCHED == 1
    if constexpr (PM == 3u) {
        sched_pair<2u * J, 2u * J + 1u>();
    } else {
        sched_single<2u * J + (PM == 2u ? 1u : 0u)>();
    }
#elif BLEND_SCHED == 2
    if constexpr (PM & 1u) {
        sched_single_a2<2u * J>();
    }
    if constexpr (PM & 2u) {
        sched_single_a2<2u * J + 1u>();
    }
#else
    if constexpr (PM == 3u) {
        blend_pair_gaussian_math<2u * J, 2u * J + 1u>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    } else if constexpr (PM == 1u) {
        blend_one_gaussian_math<2u * J>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    } else {
        blend_one_gaussian_math<2u * J + 1u>(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
#endif
}
using BlendBodyFn = void (*)();
// Index 4 * pair + (2-bit pair mask); entry 0 of each pair is never called.
// Filled at runtime: a const initializer would need dynamic relocations, which
// the kernel loader rejects. Globals live in the TRISC's local memory.
BlendBodyFn g_blend_bodies[64];
template <uint32_t I>
inline void blend_bodies_init() {
    if constexpr (I < 64u) {
        if constexpr ((I & 3u) != 0u) {
            g_blend_bodies[I] = &blend_pair_body<I / 4u, I & 3u>;
        }
        blend_bodies_init<I + 1u>();
    }
}

inline void dispatch_blend_jump(uint32_t mask) {
    while (mask != 0u) {
        const uint32_t b = static_cast<uint32_t>(__builtin_ctz(mask)) & ~1u;  // pair's low bit
        g_blend_bodies[2u * b + ((mask >> b) & 3u)]();
        mask &= ~(3u << b);
    }
}
#else
#define BLEND_USE_JUMP_WALK 0
#endif

// PACK2 (iter 50): two 32B splats per 64B page in CB_BUCKET_BULK; splat g at
// page g/2, half g&1. Tile-local mean in words [4,5]; UNORM16 op/color [6,7].
constexpr uint32_t L1_SPLAT_BYTES = 32u;
constexpr uint32_t L1_PACK_PAGE_BYTES = 64u;

// M1b: FIXED-SIZE bulk CB slots. CB_BUCKET_BULK is circular and accessed with
// LINEAR pointer arithmetic (buck + q*page) over a whole tile's multi-page span.
// A reservation that wraps the ring corrupts the tail (linear reads run past the
// physical end). Fat full subchunks are exactly BULK_REC_SLOT pages so they
// never straddle; variable single-subchunk tiles do. Reserve/wait/pop a FIXED
// slot per tile so every tile is slot-aligned (the CB is sized as exactly 2
// slots in blend_device.cpp) — no ring straddle. Actual num_g records live at
// the slot head; the rest of the slot is unread.
constexpr uint32_t BULK_REC_SLOT = (MB_BUCKET_FIT + 1u) >> 1;          // 4096

inline const uint32_t* l1_splat_words(const uint32_t buck, uint32_t g) {
    return reinterpret_cast<const uint32_t*>(
        buck + (g >> 1) * L1_PACK_PAGE_BYTES + (g & 1u) * L1_SPLAT_BYTES);
}

// ---- Blend transmittance saturation early-out (iter 107) -------------------
// Per microblock, stop blending once its MAX transmittance T drops below eps
// (front-to-back). T is read back to the scalar MATH thread via the STANDARD
// emit handshake (tile_regs_commit/wait + pack_tile of the T slot) but the dest
// is released WITHOUT zeroing it — open-coded TTI_STALLWAIT(STALL_MATH,PACK) +
// _llk_packer_set_math_semaphore_(), dropping the TTI_ZEROACC(CLR_ALL) the
// normal tile_regs_release performs — so the R/G/B/T accumulator survives the
// readback. The live mask gates ONLY the MATH blend dispatch (it is MATH-only
// state), so the three TRISC threads never diverge on control flow. Two runtime
// knobs (compile-defines fed from env; sweeping needs NO .so rebuild):
// BLEND_T_EPS and BLEND_T_PERIOD (period 0 => feature OFF / clean baseline).
#ifndef BLEND_FAST_TRED
#define BLEND_FAST_TRED 0
#endif
#ifndef BLEND_DECODE_AHEAD
#define BLEND_DECODE_AHEAD 0  // task #231, see the decode-ahead block below
#endif
// Decode-ahead stages the default blend path only; other knob mixes (and
// MB_STATS) run without it, so the host default can stay on.
#if BLEND_DECODE_AHEAD && (!(BLEND_COEF_DEST && BLEND_SFPU_UNORM == 1 && BLEND_JUMP_WALK && BLEND_ABL == 0 && \
                             !BLEND_FPU_QF_ABL) || defined(GSPLAT_TT_MB_STATS))
#undef BLEND_DECODE_AHEAD
#define BLEND_DECODE_AHEAD 0
#endif
#ifndef BLEND_T_EPS
#define BLEND_T_EPS 0.00390625f
#endif
#ifndef BLEND_T_PERIOD
// Default 512: period 64 is overhead-dominated (net SLOWER); 512 is the measured
// green+faster operating point (the readback handshake+scan cost amortizes while
// deep tiles still catch saturation early). Override via env BLEND_T_PERIOD.
#define BLEND_T_PERIOD 512u
#endif
constexpr float kBlendTEps = (BLEND_T_EPS);
constexpr uint32_t kBlendTPeriod = (BLEND_T_PERIOD);

// Runtime override of the saturation epsilon (driven by the viewer's
// "Transmittance threshold" slider via compute runtime-arg 0). Defaults to the
// compile-time kBlendTEps; a runtime-arg of 0 bits keeps that default, so every
// caller that does not forward a threshold reproduces the iter-107 baseline
// exactly. Each TRISC thread holds its own copy; set once in kernel_main.
static float g_blend_t_eps = kBlendTEps;

#ifdef TRISC_PACK
// Non-zeroing dest release: wait for the pack to drain and hand the dest back to
// MATH, but do NOT TTI_ZEROACC — preserve the running R/G/B/T accumulator.
inline void non_zeroing_pack_release() {
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    _llk_packer_set_math_semaphore_();
}
#endif

// MATH-only: reduce per-microblock MAX T from the packed bf16 T tile in L1 and
// rebuild the live-microblock mask (bit m set <=> microblock m still has a pixel
// with T >= eps). T decreases monotonically, so a cleared bit stays cleared.
// Task #146 (BLEND_FAST_TRED, host default 1): the reduce lives in
// blend_t_live.h; 1 = the range-compare form with a per-row-pair early exit,
// 0 = the original per-value max loop (bit-identical, tests/unit/test_blend_t_live.cpp).
inline void blend_t_reduce(uint32_t& live_mb_mask, uint32_t t_rb_addr) {
    mb_cb_consume_fence();
    const volatile uint32_t* w = reinterpret_cast<const volatile uint32_t*>(t_rb_addr);
    uint32_t eps_bits;
    __builtin_memcpy(&eps_bits, &g_blend_t_eps, 4);
#if BLEND_FAST_TRED
    live_mb_mask = blend_t_live_fast(w, eps_bits);
#else
    live_mb_mask = blend_t_live_ref(w, eps_bits);
#endif
}

// Mid-accumulation T readback. ALL threads call this; each performs its thread
// part. Uses the STANDARD CB producer/consumer flow (reserve/pack/push then
// wait_front/read/pop) so the framework guarantees the read address matches
// where the packer wrote (pack and read pointer conventions differ otherwise).
// cb_push_back resets the sequential pack counter, so each readback packs tile 0
// of a fresh slot. Only the NON-zeroing release differs from a normal emit.
inline void blend_t_readback(uint32_t& live_mb_mask) {
    MATH((_llk_math_eltwise_unary_sfpu_done_()));    // drain SFPU writes into dest
    tile_regs_commit();                               // MATH: dest section done (no zero)
    tile_regs_wait();                                 // PACK: wait for math done
    cb_reserve_back(CB_T_RB, 1);                       // PACK: reserve scratch slot
    pack_tile(3, CB_T_RB);                             // PACK: pack T (dest tile 3) -> slot tile 0
    cb_push_back(CB_T_RB, 1);                          // PACK: publish to UNPACK/MATH
    PACK((non_zeroing_pack_release()));               // release WITHOUT zeroing the acc
    tile_regs_acquire();                              // MATH: re-acquire dest (acc intact)
    cb_wait_front(CB_T_RB, 1);                         // UNPACK: wait for the packed T
    const uint32_t t_rb_addr = get_tile_address(CB_T_RB, 0);  // all threads (mailbox sync)
#if BLEND_DECODE_AHEAD != 2
    MATH((blend_t_reduce(live_mb_mask, t_rb_addr)));  // MATH-only: rebuild live mask
#endif
#if BLEND_DECODE_AHEAD
    UNPACK((blend_t_reduce(live_mb_mask, t_rb_addr)));  // decode-ahead: TRISC0 masks the records
#endif
    cb_pop_front(CB_T_RB, 1);                          // UNPACK: free the scratch slot
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));   // resume the SFPU section
}

// ---- Decode-ahead staging (task #231, host env GSPLAT_TT_BLEND_DECODE_AHEAD) --
// TRISC1 issue time, not the SFPU, limits the blend (t230). TRISC0 (UNPACK) is
// idle in this loop, so it walks the records instead: per live record it writes
// the 14 SFPLOADI words of blend_stage_coeffs_raw, already encoded, and the
// record's mask into an L1 ring (CB_DA_RING, set up by the host only when the
// knob is on). TRISC1 loads the words and pushes them with the same SFPU ops in
// the same order as BLEND_RAW_STAGE (bit-identical, t146), then the bodies.
// TRISC0 runs its own copy of the T reduce at each readback (same code, same T
// tile), so it masks the records exactly as TRISC1 would.
// 1 = S2a: TRISC1 still scans the records and does the readbacks; it takes a
//     ring slot for each record with a live mask.
// 2 = S2: the ring holds only live records plus markers: a readback marker at
//     the same record position as today's g_seen % 512 trigger, and an end of
//     subchunk marker. TRISC1 reads only the ring and skips its T reduce.
// TRISC2 only takes part in the readbacks, so it steps from one to the next.
// Ring: header (produced count at word 0, written by TRISC0; consumed count at
// word 8, written by TRISC1), then DA_SLOTS slots of 16 words: [0, 14) the
// SFPLOADI words, [14] the mask (0 = marker), [15] the marker kind. Counts run
// over the whole launch; TRISC0 zeroes them and mailboxes the ring address to
// TRISC1 before the first get_tile_address, so the mailbox order is unchanged.
#if BLEND_DECODE_AHEAD
#if BLEND_DECODE_AHEAD > 2
#error "BLEND_DECODE_AHEAD must be 0, 1 or 2"
#endif
constexpr uint32_t CB_DA_RING = 10;
constexpr uint32_t DA_SLOTS = 64;  // power of two; mirrors blend_device.cpp
constexpr uint32_t DA_SLOT_BYTES = 64;
constexpr uint32_t DA_HDR_BYTES = 64;
constexpr uint32_t DA_PROD = 0, DA_CONS = 8;  // header words
constexpr uint32_t DA_MARK_RB = 1, DA_MARK_END = 2;
uint32_t g_da_ring = 0;  // ring byte address (TRISC0, TRISC1)
uint32_t g_da_idx = 0;   // TRISC0: slots produced; TRISC1: slots consumed
uint32_t g_da_seen = 0;  // TRISC0: last consumed count read; TRISC1: last produced count read

inline volatile uint32_t* da_hdr() {
    return reinterpret_cast<volatile uint32_t*>(g_da_ring);
}
inline volatile uint32_t* da_slot(uint32_t i) {
    return reinterpret_cast<volatile uint32_t*>(
        g_da_ring + DA_HDR_BYTES + (i & (DA_SLOTS - 1u)) * DA_SLOT_BYTES);
}
// Spin between polls of the other thread's count, so the poll loads do not
// compete with the other thread's L1 loads.
inline void da_backoff() {
    for (uint32_t k = 0; k < 8u; ++k) {
        asm volatile("nop; nop; nop; nop");
    }
}

inline void da_init() {
#ifdef TRISC_UNPACK
    g_da_ring = get_local_cb_interface(get_operand_id(CB_DA_RING)).fifo_rd_ptr << 4;
    da_hdr()[DA_PROD] = 0u;
    da_hdr()[DA_CONS] = 0u;
    asm volatile("fence" ::: "memory");  // zeroes land before TRISC1 reads the counts
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, g_da_ring);
#endif
#ifdef TRISC_MATH
    g_da_ring = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
#endif
}

#ifdef TRISC_UNPACK
// Wait for a free slot. Once full, wait for 8 free slots before polling stops.
inline void da_wait_space() {
    if (g_da_idx - g_da_seen < DA_SLOTS) {
        return;
    }
    do {
        da_backoff();
        mb_cb_consume_fence();
        g_da_seen = da_hdr()[DA_CONS];
    } while (g_da_idx - g_da_seen > DA_SLOTS - 8u);
}
inline void da_publish() {
    ++g_da_idx;
    asm volatile("fence" ::: "memory");  // the slot's stores land before the count
    da_hdr()[DA_PROD] = g_da_idx;
}
// The SFPLOADI words of blend_stage_coeffs_raw: L0 USHORT lo + L0 HI16_ONLY hi
// for d, e, a, b, c; L0 / L2 USHORT for the UNORM16 halves of w6, w7.
inline void da_put_rec(const uint32_t* rec, uint32_t mask) {
    constexpr uint32_t klo = 0x71020000u, khi = 0x71080000u, klo2 = 0x71220000u;
    const uint32_t a = rec[0], b = rec[1], c = rec[2], d = rec[4], e = rec[5];
    const uint32_t w6 = rec[6], w7 = rec[7];
    da_wait_space();
    volatile uint32_t* s = da_slot(g_da_idx);
    s[0] = (d & 0xffffu) + klo;
    s[1] = (d >> 16) + khi;
    s[2] = (e & 0xffffu) + klo;
    s[3] = (e >> 16) + khi;
    s[4] = (a & 0xffffu) + klo;
    s[5] = (a >> 16) + khi;
    s[6] = (b & 0xffffu) + klo;
    s[7] = (b >> 16) + khi;
    s[8] = (c & 0xffffu) + klo;
    s[9] = (c >> 16) + khi;
    s[10] = (w6 & 0xffffu) + klo;
    s[11] = (w6 >> 16) + klo2;
    s[12] = (w7 & 0xffffu) + klo;
    s[13] = (w7 >> 16) + klo2;
    s[14] = mask;
    da_publish();
}
#if BLEND_DECODE_AHEAD == 2
inline void da_put_mark(uint32_t kind) {
    da_wait_space();
    volatile uint32_t* s = da_slot(g_da_idx);
    s[14] = 0u;
    s[15] = kind;
    da_publish();
}
#endif
#endif  // TRISC_UNPACK

#ifdef TRISC_MATH
inline const volatile uint32_t* da_wait_slot() {
    if (g_da_idx == g_da_seen) {
        for (;;) {
            mb_cb_consume_fence();
            g_da_seen = da_hdr()[DA_PROD];
            if (g_da_idx != g_da_seen) {
                break;
            }
            da_backoff();
        }
    }
    return da_slot(g_da_idx);
}
// After the slot's last load is used, so TRISC0 may refill it.
inline void da_consumed() {
    ++g_da_idx;
    da_hdr()[DA_CONS] = g_da_idx;
}
// blend_stage_coeffs_raw with the SFPLOADI words loaded from the slot. Loads run
// four words ahead of the pushes that use them (volatile: kept in this order).
inline void da_stage(const volatile uint32_t* s) {
    volatile uint32_t* ib = ckernel::instrn_buffer;
    constexpr uint32_t A0 = (DR_S) * 2u;
    const uint32_t v0 = s[0], v1 = s[1], v2 = s[2], v3 = s[3];
    ib[0] = v0;
    const uint32_t v4 = s[4];
    ib[0] = v1;
    const uint32_t v5 = s[5];
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_MX);
    ib[0] = v2;
    const uint32_t v6 = s[6];
    ib[0] = v3;
    const uint32_t v7 = s[7];
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_MY);
    ib[0] = v4;
    const uint32_t v8 = s[8];
    ib[0] = v5;
    const uint32_t v9 = s[9];
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_A);
    ib[0] = v6;
    const uint32_t v10 = s[10];
    ib[0] = v7;
    const uint32_t v11 = s[11];
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_B);
    ib[0] = v8;
    const uint32_t v12 = s[12];
    ib[0] = v9;
    const uint32_t v13 = s[13];
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_C);
    TTI_SFPLOAD(1, 0, 7, A0 + 2u * S_INV);
    ib[0] = v10; TTI_SFPCAST(0, 0, 0);
    ib[0] = v11; TTI_SFPMUL(0, 1, 9, 0, 0); TTI_SFPCAST(2, 2, 0);
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_OP);
    ib[0] = v12; TTI_SFPMUL(2, 1, 9, 2, 0); TTI_SFPCAST(0, 0, 0);
    TTI_SFPSTORE(2, 0, 7, A0 + 2u * S_CR);
    ib[0] = v13; TTI_SFPMUL(0, 1, 9, 0, 0); TTI_SFPCAST(2, 2, 0);
    TTI_SFPSTORE(0, 0, 7, A0 + 2u * S_CG);
    TTI_SFPMUL(2, 1, 9, 2, 0);
    TTI_SFPSTORE(2, 0, 7, A0 + 2u * S_CB);
}
#endif  // TRISC_MATH
#endif  // BLEND_DECODE_AHEAD

// ---- Sub-tile waste instrumentation (GSPLAT_TT_MB_STATS, default OFF) -------
// Scalar MATH-thread counters over every slab record the blend consumes; one
// DPRINT line per core per launch at kernel end. Compiled out entirely unless
// the host sets GSPLAT_TT_MB_STATS=1 (blend_device.cpp), so the perf path is
// untouched. Pixel test mirrors blend_one_gaussian_math: pixel (c+0.5, r+0.5)
// of microblock m (box origin ((m&3)*8, (m>>2)*4), 8 wide x 4 tall) is "live"
// iff op*exp(min(power,0)) >= floor  <=>  min(power,0) >= -ln(op/floor);
// floor = 1/GSPLAT_TT_MB_STATS_INV_FLOOR (default 255, the GPU 3DGS skip).
#if defined(GSPLAT_TT_MB_STATS) && defined(TRISC_MATH)
#ifndef GSPLAT_TT_MB_STATS_INV_FLOOR
#define GSPLAT_TT_MB_STATS_INV_FLOOR 255.0f
#endif
constexpr float kStInvFloor = GSPLAT_TT_MB_STATS_INV_FLOOR;
struct MbStats {
    uint32_t rec;        // records consumed (splat-tile pairs reaching blend)
    uint32_t rec_live;   // records with a non-zero cull mask
    uint32_t mb_kept;    // popcount(cull mask)
    uint32_t mb_sat;     // kept microblocks skipped by T-saturation early-out
    uint32_t mb_disp;    // microblocks actually blended (SFPU vector dispatches)
    uint32_t pairops;    // dispatch_blend_pairs calls (pair or single)
    uint32_t mb_useful;  // dispatched microblocks with >= 1 live pixel
    uint32_t px_live;    // live pixels inside dispatched microblocks
    // Task #148 (true saturation: T read back before EVERY record, eps = BLEND_T_EPS):
    uint32_t rec_disp;   // records that dispatch >= 1 microblock (mask != 0)
    uint32_t mb_dsat;    // dispatched microblocks already saturated (max T < eps)
    uint32_t mb_waste;   // dispatched microblocks with no live pixel OR saturated
    uint32_t pair_waste; // pair-ops whose every dispatched microblock is waste
    uint32_t rec_waste;  // dispatching records whose every microblock is waste
};
MbStats g_mb_st{};

inline uint32_t st_popc(uint32_t v) {
    uint32_t n = 0;
    while (v != 0u) {
        v &= v - 1u;
        ++n;
    }
    return n;
}

inline float st_bits_f(uint32_t u) {
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

// ln(x), x > 0 normal: exponent split + atanh series (|err| ~1e-5).
inline float st_ln(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, 4);
    const int32_t ex = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
    const float m = st_bits_f((bits & 0x7fffffu) | 0x3f800000u);
    const float s = (m - 1.0f) / (m + 1.0f);
    const float s2 = s * s;
    const float lnm = 2.0f * s * (1.0f + s2 * (1.0f / 3.0f + s2 * (1.0f / 5.0f + s2 * (1.0f / 7.0f))));
    return lnm + static_cast<float>(ex) * 0.69314718f;
}

// live_true: bit m set <=> microblock m still has a pixel with T >= eps,
// from a T readback taken right before this record (task #148).
inline void st_count_waste(uint32_t mask, uint32_t useful) {
    const uint32_t waste = mask & ~useful;
    g_mb_st.mb_waste += st_popc(waste);
    for (uint32_t j = 0; j < NUM_MB / 2; ++j) {
        const uint32_t pm = (mask >> (2u * j)) & 3u;
        if (pm != 0u && ((waste >> (2u * j)) & 3u) == pm) {
            ++g_mb_st.pair_waste;
        }
    }
    if (waste == mask) {
        ++g_mb_st.rec_waste;
    }
}

inline void st_record(const uint32_t* rec, uint32_t mask, uint32_t live_true) {
    const uint32_t raw = rec[3];
    ++g_mb_st.rec;
    if (raw == 0u) {
        return;
    }
    ++g_mb_st.rec_live;
    g_mb_st.mb_kept += st_popc(raw);
    g_mb_st.mb_sat += st_popc(raw & ~mask);
    if (mask == 0u) {
        return;
    }
    ++g_mb_st.rec_disp;
    g_mb_st.mb_dsat += st_popc(mask & ~live_true);
    g_mb_st.mb_disp += st_popc(mask);
    for (uint32_t j = 0; j < NUM_MB / 2; ++j) {
        if (((mask >> (2u * j)) & 3u) != 0u) {
            ++g_mb_st.pairops;
        }
    }
    const float op = static_cast<float>(rec[6] & 0xffffu) * (1.0f / 65535.0f);
    if (op * kStInvFloor < 1.0f) {
        st_count_waste(mask, 0u);
        return;  // no pixel can reach floor
    }
    uint32_t useful = 0u;  // dispatched, >= 1 live pixel, not saturated
    const float thr = -st_ln(op * kStInvFloor);
    const float A = st_bits_f(rec[0]), B = st_bits_f(rec[1]), C = st_bits_f(rec[2]);
    const float mx = st_bits_f(rec[4]), my = st_bits_f(rec[5]);
    for (uint32_t m = 0; m < NUM_MB; ++m) {
        if (((mask >> m) & 1u) == 0u) {
            continue;
        }
        const float ox = static_cast<float>((m & 3u) * 8u) + 0.5f - mx;
        const float oy = static_cast<float>((m >> 2) * 4u) + 0.5f - my;
        uint32_t n = 0;
        for (uint32_t r = 0; r < 4u; ++r) {
            const float dy = oy + static_cast<float>(r);
            for (uint32_t c = 0; c < 8u; ++c) {
                const float dx = ox + static_cast<float>(c);
                float p = A * (dx * dx) + B * (dx * dy) + C * (dy * dy);
                if (p > 0.0f) {
                    p = 0.0f;
                }
                if (p >= thr) {
                    ++n;
                }
            }
        }
        g_mb_st.px_live += n;
        if (n != 0u) {
            ++g_mb_st.mb_useful;
            useful |= 1u << m;
        }
    }
    st_count_waste(mask, useful & live_true);
}

inline void st_emit() {
    DPRINT << "MBSTATS " << g_mb_st.rec << " " << g_mb_st.rec_live << " " << g_mb_st.mb_kept << " "
           << g_mb_st.mb_sat << " " << g_mb_st.mb_disp << " " << g_mb_st.pairops << " "
           << g_mb_st.mb_useful << " " << g_mb_st.px_live << " " << g_mb_st.rec_disp << " "
           << g_mb_st.mb_dsat << " " << g_mb_st.mb_waste << " " << g_mb_st.pair_waste << " "
           << g_mb_st.rec_waste << ENDL();
}
#define MB_STATS_RECORD(rec, mask, live_true) MATH((st_record((rec), (mask), (live_true))))
#define MB_STATS_EMIT() MATH((st_emit()))
#else
#define MB_STATS_RECORD(rec, mask, live_true)
#define MB_STATS_EMIT()
#endif

// ---- Per-tile MATH cost (GSPLAT_TT_MB_TILECYC=1, task #147; default OFF) ----
// Records, live records (mask != 0) and wall-clock cycles (1350 MHz) spent in
// process_tile_l1_blend per output tile, summed over its subchunks; DPRINTed
// ("TC rec live cyc ...") at kernel end. Feeds the fused materialize+blend model.
// Task #172 appends the per-tile fixed-cost split (all MATH-thread wall cycles):
//   disp  microblock dispatches (popcount of every dispatched mask)
//   ntrb  T readbacks, trb their cycles (inside cyc)
//   stage per-subchunk SFPU start + floor/inv/hoist staging (inside cyc)
//   tail  per-subchunk SFPU done + MATH->UNPACK ack + slab pop
//   init  DEST acquire + fill R/G/B/T + ramp copies (acquire waits on the
//         previous tile's pack)
//   emit  commit + R/G/B pack issue
//   wall  tile start (first counts wait) -> emit end
// Line: "TC rec live cyc disp ntrb trb stage tail init emit wall".
#if defined(GSPLAT_TT_MB_TILECYC) && defined(TRISC_MATH)
struct TileCyc {
    uint32_t rec, live, cyc, disp, ntrb, trb, stage, tail, init, emit, wall;
};
constexpr uint32_t kTcMax = 48;
TileCyc g_tc[kTcMax];
uint32_t g_tc_n = 0;
uint32_t g_tc_drop = 0;
TileCyc g_tc_cur{};
uint32_t g_tc_t0 = 0;
uint32_t g_tc_tp = 0;  // start of the current part
uint32_t g_tc_tw = 0;  // start of the current tile
inline uint32_t tc_now() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0];
}
inline uint32_t tc_popc(uint32_t v) {
    uint32_t n = 0;
    while (v != 0u) {
        v &= v - 1u;
        ++n;
    }
    return n;
}
inline void tc_dump() {
    for (uint32_t i = 0; i < g_tc_n; ++i) {
        const TileCyc& c = g_tc[i];
        DPRINT << "TC " << c.rec << " " << c.live << " " << c.cyc << " " << c.disp << " " << c.ntrb
               << " " << c.trb << " " << c.stage << " " << c.tail << " " << c.init << " " << c.emit
               << " " << c.wall << ENDL();
    }
    DPRINT << "TCEND " << g_tc_n << " " << g_tc_drop << ENDL();
}
#define TC_BEGIN() MATH((g_tc_t0 = tc_now()))
#define TC_LIVE(mask) MATH((++g_tc_cur.live, g_tc_cur.disp += tc_popc(mask)))
#define TC_END(n) MATH((g_tc_cur.cyc += tc_now() - g_tc_t0, g_tc_cur.rec += (n)))
#define TC_PART0() MATH((g_tc_tp = tc_now()))
#define TC_PART1(field) MATH((g_tc_cur.field += tc_now() - g_tc_tp))
#define TC_TRB() MATH((++g_tc_cur.ntrb))
#define TC_TILE0() MATH((g_tc_tw = tc_now()))
#define TC_EMIT() \
    MATH((g_tc_cur.wall = tc_now() - g_tc_tw, \
          g_tc_n < kTcMax ? (void)(g_tc[g_tc_n++] = g_tc_cur) : (void)++g_tc_drop, g_tc_cur = TileCyc{}))
#define TC_DUMP() MATH((tc_dump()))
#else
#define TC_BEGIN()
#define TC_LIVE(mask)
#define TC_END(n)
#define TC_PART0()
#define TC_PART1(field)
#define TC_TRB()
#define TC_TILE0()
#define TC_EMIT()
#define TC_DUMP()
#endif

#if BLEND_DECODE_AHEAD
// The records loop of process_tile_l1_blend in decode-ahead mode, per thread.
inline void blend_da_walk(uint32_t num_g, uint32_t buck, uint32_t& live_mb_mask, uint32_t& g_seen) {
#if defined(TRISC_UNPACK)
    mb_cb_consume_fence();  // first read of this slab on this thread
    for (uint32_t g = 0; g < num_g; g++) {
        if (kBlendTPeriod != 0u && g_seen != 0u && (g_seen % kBlendTPeriod) == 0u) {
            BLEND_PZ_SUB("cmp_trb");
#if BLEND_DECODE_AHEAD == 2
            da_put_mark(DA_MARK_RB);  // before blocking in the readback
#endif
            blend_t_readback(live_mb_mask);
        }
        ++g_seen;
        const uint32_t* rec = l1_splat_words(buck, g);
        const uint32_t mask = rec[3] & live_mb_mask;
        if (mask != 0u) {
            da_put_rec(rec, mask);
        }
    }
#if BLEND_DECODE_AHEAD == 2
    da_put_mark(DA_MARK_END);
#endif
#elif defined(TRISC_MATH) && BLEND_DECODE_AHEAD == 1
    for (uint32_t g = 0; g < num_g; g++) {
        if (kBlendTPeriod != 0u && g_seen != 0u && (g_seen % kBlendTPeriod) == 0u) {
            BLEND_PZ_SUB("cmp_trb");
            TC_PART0();
            blend_t_readback(live_mb_mask);
            TC_PART1(trb);
            TC_TRB();
        }
        ++g_seen;
        const uint32_t mask = l1_splat_words(buck, g)[3] & live_mb_mask;
        if (mask != 0u) {
            TC_LIVE(mask);
            const volatile uint32_t* s = da_wait_slot();
            da_stage(s);
            da_consumed();
            dispatch_blend_jump(mask);
        }
    }
#elif defined(TRISC_MATH)
    (void)buck;
    g_seen += num_g;  // unused here: the markers carry the readback positions
    for (;;) {
        const volatile uint32_t* s = da_wait_slot();
        const uint32_t mask = s[14];
        if (mask == 0u) {
            const uint32_t kind = s[15];
            da_consumed();
            if (kind == DA_MARK_END) {
                break;
            }
            BLEND_PZ_SUB("cmp_trb");
            TC_PART0();
            blend_t_readback(live_mb_mask);
            TC_PART1(trb);
            TC_TRB();
            continue;
        }
        TC_LIVE(mask);
        da_stage(s);
        da_consumed();
        dispatch_blend_jump(mask);
    }
#else  // TRISC_PACK: only the readbacks, at the same record positions
    (void)buck;
    for (uint32_t g = 0; g < num_g;) {
        if (kBlendTPeriod != 0u && g_seen != 0u && (g_seen % kBlendTPeriod) == 0u) {
            BLEND_PZ_SUB("cmp_trb");
            blend_t_readback(live_mb_mask);
        }
        uint32_t step = num_g - g;
        if (kBlendTPeriod != 0u) {
            const uint32_t to_next = kBlendTPeriod - (g_seen % kBlendTPeriod);
            step = to_next < step ? to_next : step;
        }
        g += step;
        g_seen += step;
    }
#endif
}
#endif  // BLEND_DECODE_AHEAD

// Blend one subchunk whose PACK2 records + masks sit in CB_BUCKET_BULK /
// CB_BMASK_BULK (iter 49/50). Separate from in-budget CB_BUCKET/CB_BMASK so
// bulk reserve does not deadlock against coeff-stream scratch.
inline void process_tile_l1_blend(
    uint32_t num_g, uint32_t& live_mb_mask, uint32_t& g_seen) {
    if (num_g == 0) {
        return;
    }
    {
        BLEND_PZ_WAIT("cmp_bulk_wait");
        cb_wait_front(CB_BUCKET_BULK, BULK_REC_SLOT);
    }
    const uint32_t buck = get_tile_address(CB_BUCKET_BULK, 0);
    TC_BEGIN();
    TC_PART0();
    {
        BLEND_PZ_SUB("cmp_stage");
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
#if BLEND_COEF_DEST && defined(BLEND_PIXEL_FLOOR)
        MATH((blend_stage_floor()));
#endif
#if BLEND_COEF_DEST && BLEND_SFPU_UNORM
        MATH((blend_stage_inv()));
#endif
#if BLEND_CONST_HOIST && BLEND_COEF_DEST
        MATH((blend_stage_hoist()));
#endif
    }
    TC_PART1(stage);
#if BLEND_USE_JUMP_WALK && BLEND_SCHED == 2
    MATH((blend_sched_record()));  // after all other TRISC1 replay use (tile init)
#endif
#if BLEND_COEF_DEST && BLEND_SFPU_UNORM == 1 && BLEND_USE_RAW_STAGE
    // SFPLOADI L0 USHORT, L0 HI16_ONLY, L2 USHORT opcode words, kept in registers.
    uint32_t raw_klo = 0x71020000u, raw_khi = 0x71080000u, raw_klo2 = 0x71220000u;
    asm volatile("" : "+r"(raw_klo), "+r"(raw_khi), "+r"(raw_klo2));
#endif
#if BLEND_DECODE_AHEAD
    blend_da_walk(num_g, buck, live_mb_mask, g_seen);
#else
#if BLEND_ABL == 4
    for (uint32_t g = num_g; g < num_g; g++) {
#else
    for (uint32_t g = 0; g < num_g; g++) {
#endif
        // Periodic transmittance readback (per-tile gaussian count, across
        // subchunks). period 0 => disabled (compiles out to the baseline path).
        if (kBlendTPeriod != 0u && g_seen != 0u && (g_seen % kBlendTPeriod) == 0u) {
            BLEND_PZ_SUB("cmp_trb");
            TC_PART0();
            blend_t_readback(live_mb_mask);
            TC_PART1(trb);
            TC_TRB();
        }
        ++g_seen;
        const uint32_t* rec = l1_splat_words(buck, g);
        // M3: the cull writer stored the 32-bit microblock mask into word3 of the
        // slab record (the dead depth key). Read it straight from rec[3] — no
        // separate CB_BMASK_BULK / DRAM cull_masks round-trip.
        // Mask out microblocks whose transmittance already saturated (MATH-only:
        // live_mb_mask stays all-ones on UNPACK/PACK, whose dispatch is a no-op).
        const uint32_t mask = rec[3] & live_mb_mask;
#if defined(GSPLAT_TT_MB_STATS)
        // Task #148: true per-microblock saturation before this record (all
        // threads take part in the readback; the kernel's own live mask and
        // its period-driven early-out are left untouched).
        uint32_t st_live_true = 0xFFFFFFFFu;
        if (g_seen > 1u) {
            blend_t_readback(st_live_true);
        }
        MB_STATS_RECORD(rec, mask, st_live_true);
#endif
#if BLEND_ABL == 3
        if (mask == 0xFFFFFFFFu && rec[6] == 0xFFFFFFFFu) {
            asm volatile("nop");  // keep the mask read; never true in practice
        }
        if (false) {
#else
        if (mask != 0u) {
#endif
            TC_LIVE(mask);
            // UNORM16 op/color -> fp32 bits, integer bit-exact (TRISC scalar code
            // has no FPU; the float form was 8 libgcc calls per record, task #39).
            const uint32_t w6 = rec[6], w7 = rec[7];
#if BLEND_COEF_DEST && BLEND_SFPU_UNORM == 1
            // Decoded on the SFPU; the dispatch reads the staged values, not these.
            const uint32_t op = 0u, cr = 0u, cg = 0u, cbv = 0u;
#if BLEND_USE_RAW_STAGE
            (void)w6; (void)w7;
            MATH((blend_stage_coeffs_raw(rec, raw_klo, raw_khi, raw_klo2)));
#else
            MATH((blend_stage_coeffs_q(rec[0], rec[1], rec[2], rec[4], rec[5], w6, w7)));
#endif
#else
            const uint32_t op = dm_fp32::unorm16_to_f(w6 & 0xffffu);
            const uint32_t cr = dm_fp32::unorm16_to_f(w6 >> 16);
            const uint32_t cg = dm_fp32::unorm16_to_f(w7 & 0xffffu);
            const uint32_t cbv = dm_fp32::unorm16_to_f(w7 >> 16);
#if BLEND_COEF_DEST
            MATH((blend_stage_coeffs(rec[0], rec[1], rec[2], rec[4], rec[5], op, cr, cg, cbv)));
#if BLEND_SFPU_UNORM == 2
            MATH((unorm16_sfpu_check(w6 & 0xffffu, op)));
            MATH((unorm16_sfpu_check(w6 >> 16, cr)));
            MATH((unorm16_sfpu_check(w7 & 0xffffu, cg)));
            MATH((unorm16_sfpu_check(w7 >> 16, cbv)));
#endif
#endif
#endif
#if BLEND_USE_JUMP_WALK
            dispatch_blend_jump(mask);
#else
            dispatch_blend_pairs<0>(mask, rec[0], rec[1], rec[2], rec[4], rec[5], 0u,
                                    op, cr, cg, cbv);
#endif
        }
    }
#endif  // BLEND_DECODE_AHEAD
    TC_PART0();
    BLEND_PZ_SUB("cmp_sc_tail");
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    TC_END(num_g);
    // MATH->UNPACK back-pressure ack (mirrors process_tile_gaussians): UNPACK runs
    // cb_pop_front and would otherwise free this CB_BUCKET_BULK slot the instant it
    // mailboxed MATH the address — letting a FAST producer (the bulk payload DMA)
    // recycle the slot to the next subchunk before MATH finished reading => torn
    // records on a few tiles (non-deterministic ~35 dB). The slow gather producer
    // hid this. Block UNPACK on MATH completion before the pop.
    MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, num_g + 1u)));
    UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId));
    cb_pop_front(CB_BUCKET_BULK, BULK_REC_SLOT);
    TC_PART1(tail);
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_blend_sfpu");
    // Runtime-arg 0: saturation epsilon bits (viewer slider). 0 => keep the
    // compile-time default (kBlendTEps) so non-forwarding callers are unchanged.
    {
        const uint32_t eps_bits = get_arg_val<uint32_t>(0);
        if (eps_bits != 0u) {
            __builtin_memcpy(&g_blend_t_eps, &eps_bits, 4);
        }
    }
#if defined(BLEND_PIXEL_FLOOR)
    g_pixel_floor_bits = get_arg_val<uint32_t>(1);
#endif
#if BLEND_DECODE_AHEAD
    da_init();  // before the first get_tile_address (mailbox order)
#endif
    cb_wait_front(CB_CORE_TILES, 1);
    const uint32_t num_tiles =
        reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
    cb_pop_front(CB_CORE_TILES, 1);

    init_sfpu(CB_XRAMP, CB_COLOR_OUT);
    fill_tile_init();
#if BLEND_USE_JUMP_WALK
    blend_bodies_init<0>();
#endif

    if (num_tiles == 0) {
        MB_STATS_EMIT();
        return;
    }

    // Ramps are constant across tiles; the reader streams them once per core.
    cb_wait_front(CB_XRAMP, 1);
    cb_wait_front(CB_YRAMP, 1);

    // Task #60: num_tiles == 0xFFFFFFFF => the reader claims tiles dynamically
    // and ends the stream with an MB_FLAG_DONE counts page.
    const bool dynamic = num_tiles == 0xFFFFFFFFu;
    for (uint32_t t = 0; dynamic || t < num_tiles; t++) {
        if (dynamic) {
            cb_wait_front(CB_MB_COUNTS, 1);
            const uint32_t f0 =
                reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0))[1];
            if ((f0 & MB_FLAG_DONE) != 0u) {
                cb_pop_front(CB_MB_COUNTS, 1);
                break;
            }
        }
        bool tile_regs_held = false;
        bool tile_done = false;
        // Per-tile early-out state (persists across this tile's subchunks).
        uint32_t live_mb_mask = 0xFFFFFFFFu;
        uint32_t g_seen = 0u;
        TC_TILE0();
        while (!tile_done) {
            {
                BLEND_PZ_WAIT("cmp_wait_cnt");
                cb_wait_front(CB_MB_COUNTS, 1);
            }
            uint32_t num_g;
            uint32_t flags = 1u;
            {
                auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0));
                num_g = cptr[0];
                flags = cptr[1];
            }
            const bool continue_blend = (flags & MB_FLAG_CONTINUE) != 0;
            const bool emit_tile = (flags & MB_FLAG_EMIT) != 0;
            const bool l1_bulk = (flags & MB_FLAG_L1_BULK) != 0;

            if (!continue_blend) {
                BLEND_PZ("cmp_init");
                TC_PART0();
                tile_regs_acquire();
                tile_regs_held = true;

                fill_tile(0, 0.0f);
                fill_tile(1, 0.0f);
                fill_tile(2, 0.0f);
                fill_tile(3, 1.0f);

                copy_tile_to_dst_init_short(CB_XRAMP);
                copy_tile(CB_XRAMP, 0, 4);
                copy_tile_to_dst_init_short(CB_YRAMP);
                copy_tile(CB_YRAMP, 0, 5);
                TC_PART1(init);
            }

            // M1: ALL tiles (single + fat) consume the materialized L1 slab.
            // Empty tiles (num_g==0) early-return inside process_tile_l1_blend.
            (void)l1_bulk;
            process_tile_l1_blend(num_g, live_mb_mask, g_seen);

            if (emit_tile) {
                BLEND_PZ("cmp_emit");
                TC_PART0();
                tile_regs_commit();
                tile_regs_wait();
                cb_reserve_back(CB_COLOR_OUT, 3);
                pack_tile(0, CB_COLOR_OUT);
                pack_tile(1, CB_COLOR_OUT);
                pack_tile(2, CB_COLOR_OUT);
                cb_push_back(CB_COLOR_OUT, 3);
                tile_regs_release();
                tile_regs_held = false;
                tile_done = true;
                TC_PART1(emit);
                TC_EMIT();
            }

            cb_pop_front(CB_MB_COUNTS, 1);
        }
        (void)tile_regs_held;
    }

    cb_pop_front(CB_XRAMP, 1);
    cb_pop_front(CB_YRAMP, 1);
    MB_STATS_EMIT();
    TC_DUMP();
}
