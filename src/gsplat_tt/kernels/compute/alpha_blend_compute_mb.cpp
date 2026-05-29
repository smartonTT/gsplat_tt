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
constexpr uint32_t CB_COLOR_OUT = 16;

constexpr uint32_t NUM_MB = 32;

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
    vFloat A  = ckernel::sfpu::Converter::as_float(a_bits);
    vFloat B  = ckernel::sfpu::Converter::as_float(b_bits);
    vFloat C  = ckernel::sfpu::Converter::as_float(c_bits);
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

// Process all gaussians of microblock m (vector index IX). Reads coeff rows
// from CB_MB_COEFF on the compute thread; runs the SFPU body on the math thread.
template <uint32_t IX>
inline void process_microblock(uint32_t cnt) {
    if (cnt == 0) {
        return;
    }
    // Enter the SFPU dest-addressing context ONCE per microblock (proven by the
    // ramp diagnostic). Calling start_/done_ per gaussian drifts the dest RWC and
    // corrupts the persistent R/G/B/T read-modify-write across the loop.
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    for (uint32_t i = 0; i < cnt; i++) {
#if defined(MB_DEBUG_WEIGHT) || defined(MB_DEBUG_SHOWA)
        // Only the frontmost gaussian matters for this diagnostic; still drain
        // the rest of the CB so the reader doesn't block.
        const bool do_math = (i == 0);
#else
        const bool do_math = true;
#endif
        cb_wait_front(CB_MB_COEFF, 1);
        // get_tile_address reads the CB rd_ptr on the UNPACK thread and broadcasts
        // the byte address to all threads via mailbox; L1 is shared so MATH can
        // then deref it directly. (Compute kernels can't call get_read_ptr.)
        auto row = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COEFF, 0));
        const uint32_t a   = row[0];
        const uint32_t b   = row[1];
        const uint32_t c   = row[2];
        const uint32_t d   = row[3];
        const uint32_t e   = row[4];
        const uint32_t fc  = row[5];
        const uint32_t op  = row[6];
        const uint32_t cr  = row[7];
        const uint32_t cg  = row[8];
        const uint32_t cbv = row[9];
        if (do_math) {
            MATH((blend_one_gaussian_math<IX>(a, b, c, d, e, fc, op, cr, cg, cbv)));
        }
        cb_pop_front(CB_MB_COEFF, 1);
    }
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
}

// Compile-time unroll over the 32 microblocks.
template <uint32_t M>
inline void process_all_microblocks(const uint32_t* counts) {
    if constexpr (M < NUM_MB) {
        process_microblock<MB_IX[M]>(counts[M]);
        process_all_microblocks<M + 1>(counts);
    }
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
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    init_sfpu(CB_XRAMP, CB_COLOR_OUT);
    fill_tile_init();

    if (num_tiles == 0) {
        return;
    }

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(CB_MB_COUNTS, 1);
        uint32_t counts[NUM_MB];
        {
            auto cptr = reinterpret_cast<volatile uint32_t*>(get_tile_address(CB_MB_COUNTS, 0));
            for (uint32_t m = 0; m < NUM_MB; m++) {
                counts[m] = cptr[m];
            }
        }

        cb_wait_front(CB_XRAMP, 1);
        cb_wait_front(CB_YRAMP, 1);

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
        for (uint32_t m = 0; m < NUM_MB; m++) {
            for (uint32_t i = 0; i < counts[m]; i++) {
                cb_wait_front(CB_MB_COEFF, 1);
                cb_pop_front(CB_MB_COEFF, 1);
            }
        }
#elif defined(MB_DEBUG_RAMP)
        debug_all_microblocks<0>();
        // Still drain CB_MB_COEFF so the reader doesn't deadlock.
        for (uint32_t m = 0; m < NUM_MB; m++) {
            for (uint32_t i = 0; i < counts[m]; i++) {
                cb_wait_front(CB_MB_COEFF, 1);
                cb_pop_front(CB_MB_COEFF, 1);
            }
        }
#else
        process_all_microblocks<0>(counts);
#endif

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_COLOR_OUT, 3);
        pack_tile(0, CB_COLOR_OUT);
        pack_tile(1, CB_COLOR_OUT);
        pack_tile(2, CB_COLOR_OUT);
        cb_push_back(CB_COLOR_OUT, 3);
        tile_regs_release();

        cb_pop_front(CB_XRAMP, 1);
        cb_pop_front(CB_YRAMP, 1);
        cb_pop_front(CB_MB_COUNTS, 1);
    }
}
