// T1 blend backward compute (#693, design docs/t685/DESIGN.md, numpy model
// gsplat/blend_backward_dev.py:emulate). Per 32x32 tile: 5 fp32 planes (T, BH, dL/dimage rgb)
// and the x/y ramps sit in Dst; per pair (forward order) the SFPU visits only the vectors in
// the pair's mask and accumulates 9 per-lane term sums. v1: the ACC tile (dst 7) is packed per
// pair and the host does the lane reduction and the finish (A Su + B Sv, ...).
// v2 (#704): the finish is applied per lane, the 32 lanes are reduced on the SFPU (SFPTRANSP
// + add tree + row rotate-add) and each pair's 9 floats land in column q of 3 out vectors;
// the ACC tile is packed once per NB pairs and the writer scatters 9 floats per pair.
// Dst (fp32, full sync, 8 tiles, dst_reg index = tile * 32 + vector):
//   0 T, 1 BH, 2-4 g, 5 x ramp, 6 y ramp,
//   7 ACC: 0-8 term sums, 9-17 pair scalars, 18-23 scratch (u, v, w, ga, 1 - alpha, alpha),
//          24-29 out: pair i of a pack -> column i % 8 of vectors 24 + 3 (i / 8) + {0, 1, 2}
//          (group 0 rows: dop dr dg db; group 1: dA dB dC dmx; group 2 row 0: dmy).
// Compile defines: BWD_AMAX_BITS, BWD_AMIN_BITS, BWD_TMIN_BITS (fp32 bits of the blend cfg).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_PLANE = 1, CB_RAMP = 2, CB_REC = 3, CB_OUT = 16;
constexpr uint32_t N_PLANES = 5;
constexpr uint32_t DR_T = 0, DR_BH = 32, DR_G0 = 64, DR_G1 = 96, DR_G2 = 128, DR_X = 160, DR_Y = 192;
constexpr uint32_t ACC_TILE = 7, DR_ACC = 224;
constexpr uint32_t DR_S = DR_ACC + 9;  // Af Bf Cf mx my op cr cg cb
constexpr uint32_t S_AF = 0, S_BF = 1, S_CF = 2, S_MX = 3, S_MY = 4, S_OP = 5, S_CR = 6, S_CG = 7, S_CB = 8;
constexpr uint32_t DR_U = DR_ACC + 18, DR_V = DR_ACC + 19, DR_W = DR_ACC + 20, DR_GA = DR_ACC + 21,
                   DR_OM = DR_ACC + 22, DR_AL = DR_ACC + 23;
constexpr uint32_t DR_O = DR_ACC + 24, NB = 16;  // out vectors, pairs per ACC pack
constexpr float kAmax = __builtin_bit_cast(float, static_cast<uint32_t>(BWD_AMAX_BITS));
constexpr float kAmin = __builtin_bit_cast(float, static_cast<uint32_t>(BWD_AMIN_BITS));
constexpr float kTmin = __builtin_bit_cast(float, static_cast<uint32_t>(BWD_TMIN_BITS));
}  // namespace

#ifdef TRISC_PACK
// As alpha_blend_compute_mb.cpp: hand Dst back to MATH without TTI_ZEROACC (state survives).
inline void non_zeroing_pack_release() {
    TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::PACK);
    _llk_packer_set_math_semaphore_();
}
#endif

#ifdef TRISC_MATH
// fp32-accurate exp for x in [-80, 0] (Juffa / tt-metal _sfpu_exp_fp32_accurate_<true>, inlined
// here so the kernel does not depend on which ckernel_sfpu_exp.h the include path finds).
// _sfpu_exp_21f_bf16_ (the forward's exp) has a degree-2 2^f poly, ~1.7e-3 relative error: the
// first device run failed the T0 oracle on small slots with it (max abs err 5e-4 of scale).
sfpi_inline sfpi::vFloat bwd_exp(sfpi::vFloat a) {
    using namespace sfpi;
    vFloat j = 1.442695f * a;
    vInt i = float_to_int16(j, RoundMode::NearestEven);
    j = int32_to_float(i, RoundMode::NearestEven);
    vFloat f = j * -6.93145752e-1f + a;
    f = j * -1.42860677e-6f + f;
    vFloat r = 1.37805939e-3f;
    r = r * f + 8.37312452e-3f;
    r = r * f + 4.16695364e-2f;
    r = r * f + 1.66664720e-1f;
    r = r * f + 4.99999851e-1f;
    i = abs(i);
    vFloat y = r * f + 1.0f;
    i = reinterpret<vInt>(copysgn(reinterpret<vFloat>(i), j));
    r = y * f + 1.0f;
    vInt e = exexp(r, ExponentMode::NoDebias);
    e += i;
    return setexp(r, e);
}

// One visited vector V of the current tile for the staged pair (emulate() body, same order).
// Kept to <= 8 live vFloats: intermediates go through the ACC tile's scratch vectors.
template <uint32_t V>
__attribute__((noinline)) void bwd_body() {
    using namespace sfpi;
    {
        vFloat u = dst_reg[DR_X + V];
        u = u - dst_reg[DR_S + S_MX];
        vFloat v = dst_reg[DR_Y + V];
        v = v - dst_reg[DR_S + S_MY];
        dst_reg[DR_U] = u;
        dst_reg[DR_V] = v;
        vFloat pw = dst_reg[DR_S + S_AF] * (u * u) + dst_reg[DR_S + S_BF] * (u * v);
        pw = pw + dst_reg[DR_S + S_CF] * (v * v);
        vFloat ex = pw;
        v_if (pw > 0.0f) { ex = 0.0f; }
        v_endif;
        v_if (ex < -80.0f) { ex = -80.0f; }  // exp(-80) is still a normal fp32; alpha < amin there
        v_endif;
        vFloat w = bwd_exp(ex);
        vFloat al = dst_reg[DR_S + S_OP] * w;
        vFloat ga = al;
        v_if (pw > 0.0f) { ga = 0.0f; }
        v_endif;
        v_if (al > kAmax) {
            al = kAmax;
            w = 0.0f;
            ga = 0.0f;
        }
        v_endif;
        dst_reg[DR_W] = w;
        dst_reg[DR_GA] = ga;
        dst_reg[DR_AL] = al;
        dst_reg[DR_OM] = vConst1 - al;
    }
    vFloat al = dst_reg[DR_AL];
    vFloat T = dst_reg[DR_T + V];
    v_if (al >= kAmin) {
        vFloat om = dst_reg[DR_OM];
        vFloat Tn = T * om;
        v_if (Tn >= kTmin) {
            dst_reg[DR_T + V] = Tn;
            vFloat at = al * T;
            vFloat g = dst_reg[DR_G0 + V];
            dst_reg[DR_ACC + 1] = dst_reg[DR_ACC + 1] + at * g;
            vFloat cg = g * dst_reg[DR_S + S_CR];
            g = dst_reg[DR_G1 + V];
            dst_reg[DR_ACC + 2] = dst_reg[DR_ACC + 2] + at * g;
            cg = cg + g * dst_reg[DR_S + S_CG];
            g = dst_reg[DR_G2 + V];
            dst_reg[DR_ACC + 3] = dst_reg[DR_ACC + 3] + at * g;
            cg = cg + g * dst_reg[DR_S + S_CB];
            vFloat bh = dst_reg[DR_BH + V] - at * cg;
            dst_reg[DR_BH + V] = bh;
            vFloat r = approx_recip(om);
            r = r * (vFloat(2.0f) - om * r);
            r = r * (vFloat(2.0f) - om * r);
            vFloat da = T * cg - bh * r;
            dst_reg[DR_ACC + 0] = dst_reg[DR_ACC + 0] + da * dst_reg[DR_W];
            vFloat G = da * dst_reg[DR_GA];
            vFloat u = dst_reg[DR_U];
            vFloat Gu = G * u;
            dst_reg[DR_ACC + 7] = dst_reg[DR_ACC + 7] + Gu;
            dst_reg[DR_ACC + 4] = dst_reg[DR_ACC + 4] + Gu * u;
            vFloat v = dst_reg[DR_V];
            dst_reg[DR_ACC + 5] = dst_reg[DR_ACC + 5] + Gu * v;
            vFloat Gv = G * v;
            dst_reg[DR_ACC + 8] = dst_reg[DR_ACC + 8] + Gv;
            dst_reg[DR_ACC + 6] = dst_reg[DR_ACC + 6] + Gv * v;
        }
        v_else { dst_reg[DR_T + V] = 0.0f; }
        v_endif;
    }
    v_endif;
}

// Sum of a's 32 lanes ... d's into row j = lane sum of the j-th argument, in all 8 columns.
sfpi_inline sfpi::vFloat bwd_lane_sum(sfpi::vFloat a, sfpi::vFloat b, sfpi::vFloat c, sfpi::vFloat d) {
    using namespace sfpi;
    subvec_transp(a, b, c, d);  // reg i row j = old reg j row i
    a = (a + b) + (c + d);      // row j: 8 column partials of argument j
    vFloat r = subvec_shflror1(a);
    a = a + r;
    r = subvec_shflror1(a);
    r = subvec_shflror1(r);
    a = a + r;
    r = subvec_shflror1(a);
    r = subvec_shflror1(r);
    r = subvec_shflror1(r);
    r = subvec_shflror1(r);
    return a + r;
}

// Column select: the pack's column value 2q + 0.5 is staged in scratch vector DR_U.
template <uint32_t O>
sfpi_inline void bwd_put(sfpi::vFloat s) {
    using namespace sfpi;
    vFloat qx = dst_reg[DR_U];
    vFloat o = dst_reg[O];
    vFloat x = dst_reg[DR_X];  // vector 0 of the x ramp: column k holds 2k + 0.5
    v_if (x == qx) { o = s; }
    v_endif;
    dst_reg[O] = o;
}

// Pair finish (linear, so applied per lane before the reduce): A = -2 Af, B = -Bf, C = -2 Cf
// exactly. dmx = A Su + B Sv, dmy = B Su + C Sv, dA = -Suu/2, dB = -Suv, dC = -Svv/2.
template <uint32_t H>
__attribute__((noinline)) void bwd_finish() {
    using namespace sfpi;
    constexpr uint32_t O = DR_O + 3 * H;
    {
        vFloat a0 = dst_reg[DR_ACC + 0], a1 = dst_reg[DR_ACC + 1], a2 = dst_reg[DR_ACC + 2],
               a3 = dst_reg[DR_ACC + 3];
        bwd_put<O>(bwd_lane_sum(a0, a1, a2, a3));
    }
    {
        vFloat su = dst_reg[DR_ACC + 7], sv = dst_reg[DR_ACC + 8];
        vFloat af = dst_reg[DR_S + S_AF];
        vFloat t = af * su;
        t = t + t;
        vFloat bf = dst_reg[DR_S + S_BF];
        vFloat dmx = -(t + bf * sv);
        dst_reg[DR_ACC + 7] = dmx;
        vFloat cf = dst_reg[DR_S + S_CF];
        t = cf * sv;
        t = t + t;
        vFloat dmy = -(t + bf * su);
        dst_reg[DR_ACC + 8] = dmy;
    }
    {
        vFloat a4 = dst_reg[DR_ACC + 4], a5 = dst_reg[DR_ACC + 5], a6 = dst_reg[DR_ACC + 6],
               a7 = dst_reg[DR_ACC + 7];
        a4 = a4 * -0.5f;
        a5 = -a5;
        a6 = a6 * -0.5f;
        bwd_put<O + 1>(bwd_lane_sum(a4, a5, a6, a7));
    }
    {
        vFloat a8 = dst_reg[DR_ACC + 8];
        bwd_put<O + 2>(bwd_lane_sum(a8, vConst0, vConst0, vConst0));
    }
}

// x ramp value of column q (2q + 0.5), as fp32 bits.
constexpr uint32_t kColX[8] = {0x3F000000u, 0x40200000u, 0x40900000u, 0x40D00000u,
                               0x41080000u, 0x41280000u, 0x41480000u, 0x41680000u};

using BwdBodyFn = void (*)();
// Filled at runtime: a const initializer would need dynamic relocations (kernel loader).
BwdBodyFn g_bwd_bodies[32];
template <uint32_t I>
inline void bwd_bodies_init() {
    if constexpr (I < 32u) {
        g_bwd_bodies[I] = &bwd_body<I>;
        bwd_bodies_init<I + 1u>();
    }
}

// Read the record page (Af Bf Cf mx my op r g b A B C mask slot), then free it for UNPACK.
inline uint32_t bwd_read_record(uint32_t ra, uint32_t* s) {
    asm volatile("fence" ::: "memory");
    const volatile uint32_t* rec = reinterpret_cast<const volatile uint32_t*>(ra);
    for (uint32_t k = 0; k < 9; k++) s[k] = rec[k];
    const uint32_t mask = rec[12];
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1u);
    return mask;
}

inline void bwd_pair(const uint32_t* s, uint32_t mask) {
    using namespace sfpi;
    using ckernel::sfpu::Converter;
    dst_reg[DR_S + 0] = Converter::as_float(s[0]);
    dst_reg[DR_S + 1] = Converter::as_float(s[1]);
    dst_reg[DR_S + 2] = Converter::as_float(s[2]);
    dst_reg[DR_S + 3] = Converter::as_float(s[3]);
    dst_reg[DR_S + 4] = Converter::as_float(s[4]);
    dst_reg[DR_S + 5] = Converter::as_float(s[5]);
    dst_reg[DR_S + 6] = Converter::as_float(s[6]);
    dst_reg[DR_S + 7] = Converter::as_float(s[7]);
    dst_reg[DR_S + 8] = Converter::as_float(s[8]);
    dst_reg[DR_ACC + 0] = vConst0;
    dst_reg[DR_ACC + 1] = vConst0;
    dst_reg[DR_ACC + 2] = vConst0;
    dst_reg[DR_ACC + 3] = vConst0;
    dst_reg[DR_ACC + 4] = vConst0;
    dst_reg[DR_ACC + 5] = vConst0;
    dst_reg[DR_ACC + 6] = vConst0;
    dst_reg[DR_ACC + 7] = vConst0;
    dst_reg[DR_ACC + 8] = vConst0;
    while (mask != 0u) {
        g_bwd_bodies[__builtin_ctz(mask)]();
        mask &= mask - 1u;
    }
}
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    init_sfpu(CB_PLANE, CB_OUT);
    MATH((bwd_bodies_init<0>()));
    if (n_tiles == 0) return;
    cb_wait_front(CB_RAMP, 2);
    for (uint32_t t = 0; t < n_tiles; t++) {
        cb_wait_front(CB_HDR, 1);
        const uint32_t ha = get_tile_address(CB_HDR, 0);  // all threads (mailbox sync)
        asm volatile("fence" ::: "memory");
        const uint32_t cnt = reinterpret_cast<const volatile uint32_t*>(ha)[2];
        cb_pop_front(CB_HDR, 1);

        tile_regs_acquire();
        cb_wait_front(CB_PLANE, N_PLANES);
        copy_tile_to_dst_init_short(CB_PLANE);
        for (uint32_t k = 0; k < N_PLANES; k++) copy_tile(CB_PLANE, k, k);
        cb_pop_front(CB_PLANE, N_PLANES);
        copy_tile_to_dst_init_short(CB_RAMP);
        copy_tile(CB_RAMP, 0, 5);
        copy_tile(CB_RAMP, 1, 6);
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));

        for (uint32_t p = 0; p < cnt; p++) {
            cb_wait_front(CB_REC, 1);
            const uint32_t ra = get_tile_address(CB_REC, 0);  // all threads (mailbox sync)
#ifdef TRISC_MATH
            uint32_t s[9];
            const uint32_t mask = bwd_read_record(ra, s);
            bwd_pair(s, mask);
            sfpi::dst_reg[DR_U] = ckernel::sfpu::Converter::as_float(kColX[p & 7u]);
            if ((p & 8u) == 0u) {
                bwd_finish<0>();
            } else {
                bwd_finish<1>();
            }
#else
            (void)ra;
#endif
            UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId)));
            cb_pop_front(CB_REC, 1);
            if ((p % NB) != NB - 1u && p + 1 != cnt) continue;

            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_OUT, 1);
            pack_tile(ACC_TILE, CB_OUT);
            cb_push_back(CB_OUT, 1);
            if (p + 1 == cnt) {
                tile_regs_release();
            } else {
                PACK((non_zeroing_pack_release()));
                tile_regs_acquire();
                MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            }
        }
    }
    cb_pop_front(CB_RAMP, 2);
}
