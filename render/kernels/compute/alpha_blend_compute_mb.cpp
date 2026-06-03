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
constexpr uint32_t CB_MB_COEFF  = 2;   // one 48B coeff row per gaussian (mb-major)
constexpr uint32_t CB_MB_COUNTS = 3;   // 32 uint32 per tile (per-microblock count)
constexpr uint32_t CB_CORE_TILES = 7;  // MB_RESIDENT: tile count from reader (no host arg)
constexpr uint32_t CB_COLOR_OUT = 16;

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
    //
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

// Stream all of a tile's gaussian-major rows. Each row's 10 coeffs + mask word
// are read ONCE (not once per microblock), then dispatched to the masked
// microblocks. One start_/done_ for the whole tile (proven safe by VECMAP).
inline void process_tile_gaussians(uint32_t num_g) {
    if (num_g == 0) {
        return;
    }
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    for (uint32_t g = 0; g < num_g; g++) {
        cb_wait_front(CB_MB_COEFF, 1);
        const uint32_t* row = reinterpret_cast<const uint32_t*>(get_tile_address(CB_MB_COEFF, 0));
        // BUCKET_CB_FENCE. The blend math reads each coeff row from L1 DIRECTLY on
        // the MATH thread, but CB flow control (cb_wait_front/cb_pop_front) runs on
        // UNPACK, which frees the slot the instant it has mailboxed MATH the
        // address — without waiting for the (slow, SFPU-bound) MATH read. On the
        // fast bucket feed the producer then recycles the slot to a LATER record
        // before MATH reads it => torn/stale rows. Two parts:
        //  (1) invalidate MATH's write-through L1 cache so it re-reads this slot
        //      fresh (the slot was last cached 8 records ago, depth=8); and
        //  (2) a bounded MATH->UNPACK back-pressure ack (below, around the row
        //      loads) so UNPACK cannot pop/free the slot until MATH has loaded it.
        MATH((mb_cb_consume_fence()));  // == invalidate_l1_cache() on Blackhole
        const uint32_t a = row[0], b = row[1], c = row[2], d = row[3], e = row[4];
        const uint32_t fc = row[5], op = row[6], cr = row[7], cg = row[8], cbv = row[9];
        uint32_t mask = row[10];
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
        dispatch_blend_guarded<0>(mask, a, b, c, d, e, fc, op, cr, cg, cbv);
        UNPACK((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId));
        cb_pop_front(CB_MB_COEFF, 1);
    }
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
}

}  // namespace

void kernel_main() {
    DeviceZoneScopedN("tile_blend_sfpu");
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
        while (!tile_done) {
            cb_wait_front(CB_MB_COUNTS, 1);
            uint32_t num_g;
            uint32_t flags = 1u;
            {
                auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0));
                num_g = cptr[0];
                flags = cptr[1];
            }
            const bool continue_blend = (flags & 2u) != 0;
            const bool emit_tile = (flags & 1u) != 0;

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

            if (num_g <= MB_BUCKET_FIT) {
                DeviceZoneScopedN("cp_inb");
                process_tile_gaussians(num_g);
            } else {
                DeviceZoneScopedN("cp_ovf");
                process_tile_gaussians(num_g);
            }

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
