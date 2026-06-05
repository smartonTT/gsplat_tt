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

#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
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

    // DEVCONIC: a_bits/b_bits/c_bits carry raw cov {cov_a,cov_b,cov_c}; derive A,B,C on SFPU.
    vFloat cov_a = ckernel::sfpu::Converter::as_float(a_bits);
    vFloat cov_b = ckernel::sfpu::Converter::as_float(b_bits);
    vFloat cov_c = ckernel::sfpu::Converter::as_float(c_bits);
    vFloat det = cov_a * cov_c - cov_b * cov_b;
    vFloat det_floor = 1e-6f;
    vec_min_max(det_floor, det);
    vFloat inv = approx_recip(det);
    inv = inv * (vFloat(2.0f) - det * inv);
    inv = inv * (vFloat(2.0f) - det * inv);
    vFloat A = vFloat(-0.5f) * (cov_c * inv);
    vFloat B = cov_b * inv;
    vFloat C = vFloat(-0.5f) * (cov_a * inv);
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
inline void blend_t_reduce(uint32_t& live_mb_mask, uint32_t t_rb_addr) {
    mb_cb_consume_fence();
    const volatile uint32_t* w = reinterpret_cast<const volatile uint32_t*>(t_rb_addr);
    float mbmax[NUM_MB];
    for (uint32_t m = 0; m < NUM_MB; ++m) {
        mbmax[m] = 0.0f;
    }
    // The packed bf16 tile is 1024 values in ROW-MAJOR device-raster order (this
    // kernel's pack/unpack is set up so device raster index == memory index; cf.
    // make_ramp + mb_perm in blend_device.cpp), two bf16 per uint32 word (low
    // half = even index). Memory index t = r*32 + c -> microblock vector
    // V = 2*(r/2)+(c&1) (identity to the mask bit / dispatch vector).
    for (uint32_t t = 0; t < 1024u; ++t) {
        const uint32_t word = w[t >> 1];
        const uint32_t half = (t & 1u) ? (word >> 16) : (word & 0xffffu);
        const uint32_t fb = half << 16;
        float tf;
        __builtin_memcpy(&tf, &fb, 4);
        const uint32_t r = t >> 5;       // t = r*32 + c (row-major)
        const uint32_t c = t & 31u;
        const uint32_t V = (r & ~1u) | (c & 1u);
        if (tf > mbmax[V]) {
            mbmax[V] = tf;
        }
    }
    uint32_t live = 0u;
    for (uint32_t m = 0; m < NUM_MB; ++m) {
        if (mbmax[m] >= g_blend_t_eps) {
            live |= (1u << m);
        }
    }
    live_mb_mask = live;
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
    MATH((blend_t_reduce(live_mb_mask, t_rb_addr)));  // MATH-only: rebuild live mask
    cb_pop_front(CB_T_RB, 1);                          // UNPACK: free the scratch slot
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));   // resume the SFPU section
}

// Blend one subchunk whose PACK2 records + masks sit in CB_BUCKET_BULK /
// CB_BMASK_BULK (iter 49/50). Separate from in-budget CB_BUCKET/CB_BMASK so
// bulk reserve does not deadlock against coeff-stream scratch.
inline void process_tile_l1_blend(
    uint32_t num_g, uint32_t& live_mb_mask, uint32_t& g_seen) {
    if (num_g == 0) {
        return;
    }
    cb_wait_front(CB_BUCKET_BULK, BULK_REC_SLOT);
    const uint32_t buck = get_tile_address(CB_BUCKET_BULK, 0);

    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    for (uint32_t g = 0; g < num_g; g++) {
        // Periodic transmittance readback (per-tile gaussian count, across
        // subchunks). period 0 => disabled (compiles out to the baseline path).
        if (kBlendTPeriod != 0u && g_seen != 0u && (g_seen % kBlendTPeriod) == 0u) {
            blend_t_readback(live_mb_mask);
        }
        ++g_seen;
        const uint32_t* rec = l1_splat_words(buck, g);
        const uint32_t a = rec[0], b = rec[1], c = rec[2];
        const uint32_t d = rec[4], e = rec[5];
        const uint32_t w6 = rec[6], w7 = rec[7];
        constexpr float kUnormInv = 1.0f / 65535.0f;
        auto unorm_bits = [](uint32_t u16) -> uint32_t {
            const float f = static_cast<float>(u16) * kUnormInv;
            uint32_t bits;
            __builtin_memcpy(&bits, &f, 4);
            return bits;
        };
        const uint32_t op = unorm_bits(w6 & 0xffffu);
        const uint32_t cr = unorm_bits(w6 >> 16);
        const uint32_t cg = unorm_bits(w7 & 0xffffu);
        const uint32_t cbv = unorm_bits(w7 >> 16);
        // M3: the cull writer stored the 32-bit microblock mask into word3 of the
        // slab record (the dead depth key). Read it straight from rec[3] — no
        // separate CB_BMASK_BULK / DRAM cull_masks round-trip.
        // Mask out microblocks whose transmittance already saturated (MATH-only:
        // live_mb_mask stays all-ones on UNPACK/PACK, whose dispatch is a no-op).
        const uint32_t mask = rec[3] & live_mb_mask;
        if (mask != 0u) {
            dispatch_blend_guarded<0>(mask, a, b, c, d, e, 0u, op, cr, cg, cbv);
        }
    }
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    // MATH->UNPACK back-pressure ack (mirrors process_tile_gaussians): UNPACK runs
    // cb_pop_front and would otherwise free this CB_BUCKET_BULK slot the instant it
    // mailboxed MATH the address — letting a FAST producer (the bulk payload DMA)
    // recycle the slot to the next subchunk before MATH finished reading => torn
    // records on a few tiles (non-deterministic ~35 dB). The slow gather producer
    // hid this. Block UNPACK on MATH completion before the pop.
    MATH((ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, num_g + 1u)));
    UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId));
    cb_pop_front(CB_BUCKET_BULK, BULK_REC_SLOT);
}

#if defined(MB_FUSE_TILE_L1_CULL)
#include "tile_l1_cull_sfpu.hpp"

// Minimal fuse compile hook (iter 71): pulls tile_l1_cull SFPU into the blend
// compute TU for LRA margin measurement; not called on the default path.
inline void fuse_l1_cull_compile_hook(
    uint32_t keep_base, uint32_t nb, uint32_t pos_base,
    const uint32_t* a, const uint32_t* b, const uint32_t* c,
    const uint32_t* mx, const uint32_t* my, const uint32_t* thr,
    uint32_t txf_bits, uint32_t tyf_bits) {
    if (nb == 0u) {
        return;
    }
    tile_l1_cull_sfpu::cull_dispatch(
        keep_base, nb, pos_base, a, b, c, mx, my, thr, txf_bits, tyf_bits, false);
}
#endif

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
    cb_wait_front(CB_CORE_TILES, 1);
    const uint32_t num_tiles =
        reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_CORE_TILES, 0))[0];
    cb_pop_front(CB_CORE_TILES, 1);

    init_sfpu(CB_XRAMP, CB_COLOR_OUT);
    fill_tile_init();

    if (num_tiles == 0) {
        return;
    }

    // Ramps are constant across tiles; the reader streams them once per core.
    cb_wait_front(CB_XRAMP, 1);
    cb_wait_front(CB_YRAMP, 1);

    for (uint32_t t = 0; t < num_tiles; t++) {
        bool tile_regs_held = false;
        bool tile_done = false;
        // Per-tile early-out state (persists across this tile's subchunks).
        uint32_t live_mb_mask = 0xFFFFFFFFu;
        uint32_t g_seen = 0u;
        while (!tile_done) {
            cb_wait_front(CB_MB_COUNTS, 1);
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
            }

            // M1: ALL tiles (single + fat) consume the materialized L1 slab.
            // Empty tiles (num_g==0) early-return inside process_tile_l1_blend.
            (void)l1_bulk;
            process_tile_l1_blend(num_g, live_mb_mask, g_seen);

            if (emit_tile) {
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
            }

            cb_pop_front(CB_MB_COUNTS, 1);
        }
        (void)tile_regs_held;
    }

    cb_pop_front(CB_XRAMP, 1);
    cb_pop_front(CB_YRAMP, 1);
}
