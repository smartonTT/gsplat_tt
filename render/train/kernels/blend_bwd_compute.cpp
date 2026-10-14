// T1 blend backward compute (#693, design docs/t685/DESIGN.md, numpy model
// gsplat/blend_backward_dev.py:emulate). Per 32x32 tile: 5 fp32 planes (T, BH, dL/dimage rgb)
// and the x/y ramps sit in Dst; per pair (forward order) the SFPU visits only the vectors in
// the pair's mask and accumulates 9 per-lane term sums. v1: the ACC tile (dst 7) is packed per
// pair and the host does the lane reduction and the finish (A Su + B Sv, ...).
// v2 (#704): the finish is applied per lane, the 32 lanes are reduced on the SFPU (SFPTRANSP
// + add tree + row rotate-add) and each pair's 9 floats land in column q of 3 out vectors;
// the ACC tile is packed once per NB pairs and the writer scatters 9 floats per pair.
// Dst (fp32, full sync, 8 tiles, dst_reg index = tile * 32 + vector):
//   0 T, 1 BH, 2-4 g, 5 x ramp, 6 y ramp (#747: x of vector V equals x of V & 9 and y of V equals
//     y of V & ~9 (vector_lanes), so the bodies read only those; x ramp vectors 16-31 hold the
//     body's fp32 constants, DR_K, rewritten per tile after the ramp copy),
//   7 ACC: 0-8 term sums, 9-17 pair scalars, 18-21 scratch (u, v, w, ga), 22 P0, 23 alpha,
//          30 Lx, 31 Ly (#728 tile-centred power, see bwd_body), 24-29 out: pair i of a pack -> column i % 8 of vectors 24 + 3 (i / 8) + {0, 1, 2}
//          (group 0 rows: dop dr dg db; group 1: dA dB dC dmx; group 2 row 0: dmy).
// Compile defines: BWD_AMAX_BITS, BWD_AMIN_BITS, BWD_TMIN_BITS (fp32 bits of the blend cfg).
// BWD_STATS (#707): MATH-thread wall-clock counters per stage (tile setup, record wait, pair
// setup, bodies, finish, pack sync) plus counts; after the last tile they are written as
// exact floats (0x4B000000 | 16-bit half, two vectors per counter) into ACC and packed as
// one extra CB_OUT tile, which the writer sends to the stats buffer (bwd_writer.cpp).
// BWD_PRESTAGE (#767): the reader hands over each record float as its two SFPLOADI instruction
// words (bwd_reader.cpp), so pair setup is copy word, copy word, SFPSTORE per value instead of
// building both words on the MATH RISC-V; same instructions, bit-identical results.
// BWD_SKIPDEAD (#782): skip vectors whose 32 pixels are all dead (T == 0). With T == 0 a body
// changes nothing but scratch rows 18-23 (Tn = 0 < t_min), and a dead pixel never comes back,
// so any stale dead mask is exact. At each mid-tile ACC pack the T plane is also packed to
// CB_TL; over the next 16 pairs PACK ORs one 64-word block per pair (vectors 2b, 2b+1) and
// publishes the tile's dead mask in CB_LIVE (word 1, then word 0 = tile tag); MATH drops the
// dead vectors from each pair's mask. Bit-identical results.
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
constexpr uint32_t CB_TL = 18, CB_LIVE = 19;  // BWD_SKIPDEAD: packed T plane, dead-mask mailbox
constexpr uint32_t N_PLANES = 5;
constexpr uint32_t DR_T = 0, DR_BH = 32, DR_G0 = 64, DR_G1 = 96, DR_G2 = 128, DR_X = 160, DR_Y = 192;
constexpr uint32_t ACC_TILE = 7, DR_ACC = 224;
constexpr uint32_t DR_S = DR_ACC + 9;  // Af Bf Cf mx my op cr cg cb
constexpr uint32_t S_AF = 0, S_BF = 1, S_CF = 2, S_MX = 3, S_MY = 4, S_OP = 5, S_CR = 6, S_CG = 7, S_CB = 8;
constexpr uint32_t DR_U = DR_ACC + 18, DR_V = DR_ACC + 19, DR_W = DR_ACC + 20, DR_GA = DR_ACC + 21,
                   DR_P0 = DR_ACC + 22, DR_AL = DR_ACC + 23, DR_LX = DR_ACC + 30, DR_LY = DR_ACC + 31;
constexpr uint32_t DR_O = DR_ACC + 24, NB = 16;  // out vectors, pairs per ACC pack
// #747: body constants staged in Dst (one SFPLOAD instead of two SFPLOADI) and three in the
// programmable constant registers (no load at all); same values, so bit-identical results.
constexpr uint32_t DR_K = DR_X + 16;
constexpr uint32_t K_LN2H = 0, K_LN2L = 1, K_C3 = 2, K_C2 = 3, K_C1 = 4, K_AMAX = 5, K_AMIN = 6, K_TMIN = 7;
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
    vFloat j = a * vConstFloatPrgm0;  // 1.442695f (bwd_consts_init)
    vInt i = float_to_int16(j, RoundMode::NearestEven);
    j = int32_to_float(i, RoundMode::NearestEven);
    vFloat f = j * dst_reg[DR_K + K_LN2H] + a;
    f = j * dst_reg[DR_K + K_LN2L] + f;
    vFloat r = f * vConstFloatPrgm2 + vConstFloatPrgm1;  // 1.37805939e-3 f + 8.37312452e-3
    r = r * f + dst_reg[DR_K + K_C3];
    r = r * f + dst_reg[DR_K + K_C2];
    r = r * f + dst_reg[DR_K + K_C1];
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
        // Tile-centred power (#728): A' u^2 + B' u v + C' v^2 cancels catastrophically in fp32
        // when the mean is 1e4+ px off the tile. With du = X - 16, dv = Y - 16 (exact) and P0,
        // Lx, Ly formed in float64 on the host (blend_backward_ref.centred_power_coeffs):
        // power = P0 + du (Lx + A' du + B' dv) + dv (Ly + C' dv).
        vFloat x = dst_reg[DR_X + (V & 9u)];
        vFloat y = dst_reg[DR_Y + (V & ~9u)];
        dst_reg[DR_U] = x - dst_reg[DR_S + S_MX];
        dst_reg[DR_V] = y - dst_reg[DR_S + S_MY];
        vFloat du = x - 16.0f;
        vFloat dv = y - 16.0f;
        vFloat pw = dst_reg[DR_LX] + dst_reg[DR_S + S_AF] * du + dst_reg[DR_S + S_BF] * dv;
        vFloat q = dst_reg[DR_LY] + dst_reg[DR_S + S_CF] * dv;
        pw = dst_reg[DR_P0] + du * pw + dv * q;
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
        vFloat amax = dst_reg[DR_K + K_AMAX];
        v_if (al > amax) {
            al = amax;
            w = 0.0f;
            ga = 0.0f;
        }
        v_endif;
        dst_reg[DR_W] = w;
        dst_reg[DR_GA] = ga;
        dst_reg[DR_AL] = al;
    }
    vFloat al = dst_reg[DR_AL];
    vFloat T = dst_reg[DR_T + V];
    vFloat amin = dst_reg[DR_K + K_AMIN];
    v_if (al >= amin) {
        vFloat om = vConst1 - al;
        vFloat Tn = T * om;
        vFloat tmin = dst_reg[DR_K + K_TMIN];
        v_if (Tn >= tmin) {
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
    for (uint32_t k = 0; k < 12; k++) s[k] = rec[k];
    const uint32_t mask = rec[12];
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1u);
    return mask;
}

#ifdef BWD_PRESTAGE
// Record page from the reader: words 2k, 2k+1 load value k (Af Bf Cf mx my op r g b P0 Lx Ly)
// into LREG0, word 24 is the vector mask.
constexpr uint32_t K_LOAD_LO = 0x71020000u, K_LOAD_HI = 0x71080000u, REC_MASK_W = 24;
static_assert(TT_OP_SFPLOADI(0, 2, 0) == K_LOAD_LO, "SFPLOADI USHORT word (bwd_reader.cpp)");
static_assert(TT_OP_SFPLOADI(0, 8, 0) == K_LOAD_HI, "SFPLOADI UPPER word (bwd_reader.cpp)");
// SFPLOAD/SFPSTORE address of dst_reg index i (fp32 Dst), as sfpi emits it.
constexpr uint32_t dr_addr(uint32_t i) { return 2u * i; }
#define BWD_STAGE(k, dr)                                         \
    do {                                                         \
        const uint32_t lo_ = a[2 * (k)], hi_ = a[2 * (k) + 1];   \
        ib[0] = lo_;                                             \
        ib[0] = hi_;                                             \
        TTI_SFPSTORE(0, 0, 7, dr_addr(dr));                      \
    } while (0)

// Stage the pair scalars from the record page, free the page for UNPACK, zero the term sums.
inline uint32_t bwd_stage_record(uint32_t ra) {
    asm volatile("fence" ::: "memory");
    volatile uint32_t* ib = ckernel::instrn_buffer;
#if BWD_PRESTAGE == 2
    // All 25 L1 loads back to back before the first instruction-buffer store, as
    // bwd_read_record does for its 13 (mode 1 interleaves them and measured slower).
    const volatile uint32_t* vw = reinterpret_cast<const volatile uint32_t*>(ra);
    uint32_t a[REC_MASK_W + 1];
#pragma GCC unroll 25
    for (uint32_t k = 0; k <= REC_MASK_W; k++) a[k] = vw[k];
#else
    // Plain loads: the compiler may issue them ahead of the instruction-buffer stores.
    const uint32_t* a = reinterpret_cast<const uint32_t*>(ra);
#endif
    BWD_STAGE(0, DR_S + 0);
    BWD_STAGE(1, DR_S + 1);
    BWD_STAGE(2, DR_S + 2);
    BWD_STAGE(3, DR_S + 3);
    BWD_STAGE(4, DR_S + 4);
    BWD_STAGE(5, DR_S + 5);
    BWD_STAGE(6, DR_S + 6);
    BWD_STAGE(7, DR_S + 7);
    BWD_STAGE(8, DR_S + 8);
    BWD_STAGE(9, DR_P0);
    BWD_STAGE(10, DR_LX);
    BWD_STAGE(11, DR_LY);
    const uint32_t mask = a[REC_MASK_W];
    asm volatile("" ::: "memory");  // every page read before the page goes back to UNPACK
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, 1u);
    using namespace sfpi;
    dst_reg[DR_ACC + 0] = vConst0;
    dst_reg[DR_ACC + 1] = vConst0;
    dst_reg[DR_ACC + 2] = vConst0;
    dst_reg[DR_ACC + 3] = vConst0;
    dst_reg[DR_ACC + 4] = vConst0;
    dst_reg[DR_ACC + 5] = vConst0;
    dst_reg[DR_ACC + 6] = vConst0;
    dst_reg[DR_ACC + 7] = vConst0;
    dst_reg[DR_ACC + 8] = vConst0;
    return mask;
}
#undef BWD_STAGE
#endif

inline void bwd_pair_setup(const uint32_t* s) {
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
    dst_reg[DR_P0] = Converter::as_float(s[9]);
    dst_reg[DR_LX] = Converter::as_float(s[10]);
    dst_reg[DR_LY] = Converter::as_float(s[11]);
    dst_reg[DR_ACC + 0] = vConst0;
    dst_reg[DR_ACC + 1] = vConst0;
    dst_reg[DR_ACC + 2] = vConst0;
    dst_reg[DR_ACC + 3] = vConst0;
    dst_reg[DR_ACC + 4] = vConst0;
    dst_reg[DR_ACC + 5] = vConst0;
    dst_reg[DR_ACC + 6] = vConst0;
    dst_reg[DR_ACC + 7] = vConst0;
    dst_reg[DR_ACC + 8] = vConst0;
}

// Per tile, after the ramp copy (it overwrites DR_K): the body constants (see DR_K).
inline void bwd_consts_init() {
    using namespace sfpi;
    vConstFloatPrgm0 = 1.442695f;
    vConstFloatPrgm1 = 8.37312452e-3f;
    vConstFloatPrgm2 = 1.37805939e-3f;
    dst_reg[DR_K + K_LN2H] = -6.93145752e-1f;
    dst_reg[DR_K + K_LN2L] = -1.42860677e-6f;
    dst_reg[DR_K + K_C3] = 4.16695364e-2f;
    dst_reg[DR_K + K_C2] = 1.66664720e-1f;
    dst_reg[DR_K + K_C1] = 4.99999851e-1f;
    dst_reg[DR_K + K_AMAX] = kAmax;
    dst_reg[DR_K + K_AMIN] = kAmin;
    dst_reg[DR_K + K_TMIN] = kTmin;
}

inline void bwd_bodies(uint32_t mask) {
    while (mask != 0u) {
        g_bwd_bodies[__builtin_ctz(mask)]();
        mask &= mask - 1u;
    }
}
#endif

#ifdef BWD_SKIPDEAD
static_assert(NB == 16, "BWD_SKIPDEAD scans one 64-word T block per pair, 16 per ACC pack");
// MATH has no CB interfaces (cb_interface is undefined on TRISC1): UNPACK resets the mailbox
// and hands its address over at kernel start.
uint32_t g_live_mb;
inline volatile uint32_t* bwd_live_mb() { return reinterpret_cast<volatile uint32_t*>(g_live_mb); }
inline void bwd_live_init() {
#ifdef TRISC_UNPACK
    g_live_mb = get_local_cb_interface(CB_LIVE).fifo_rd_ptr << 4;
    bwd_live_mb()[0] = ~0u;  // no tile tag yet (L1 keeps the last run's); PACK writes only after MATH commits
    asm volatile("fence" ::: "memory");
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, g_live_mb);
#endif
#ifdef TRISC_MATH
    g_live_mb = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
#endif
#ifdef TRISC_PACK
    g_live_mb = get_local_cb_interface(CB_LIVE).fifo_rd_ptr << 4;
#endif
}
#ifdef TRISC_MATH
// Dead-vector mask of tile t as last published by PACK (0 if none yet for this tile).
inline uint32_t bwd_dead(uint32_t t) {
    volatile uint32_t* mb = bwd_live_mb();
    if (mb[0] != t) return 0u;
    asm volatile("fence" ::: "memory");
    return mb[1];
}
#endif
#ifdef TRISC_PACK
uint32_t g_dead, g_tl_ok;
// After the T pack's push: wait until the packer has written CB_TL (its received count).
inline void bwd_tl_wait() {
    const uint16_t want = get_local_cb_interface(CB_TL).tiles_received;
    volatile tt_l1_ptr uint32_t* rp = get_cb_tiles_received_ptr(CB_TL);
    while (static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(rp))) != want) {
    }
    asm volatile("fence" ::: "memory");
    g_tl_ok = 1u;
}
// Block b of the packed T tile (tilized words 64b..64b+63): even words are vector 2b, odd 2b+1.
// T is +0.0 or positive, so a vector is dead iff the OR of its words is 0.
inline void bwd_scan_block(uint32_t b, uint32_t t) {
    if (g_tl_ok == 0u) return;
    const volatile uint32_t* w =
        reinterpret_cast<const volatile uint32_t*>((get_local_cb_interface(CB_TL).fifo_rd_ptr << 4) + 256u * b);
    uint32_t e = 0, o = 0;
#pragma GCC unroll 8
    for (uint32_t i = 0; i < 64; i += 2) {
        e |= w[i];
        o |= w[i + 1];
    }
    const uint32_t nd = ((e == 0u ? 1u : 0u) | (o == 0u ? 2u : 0u)) << (2u * b);
    if ((nd & ~g_dead) == 0u) return;
    g_dead |= nd;
    volatile uint32_t* mb = bwd_live_mb();
    mb[1] = g_dead;
    asm volatile("fence" ::: "memory");
    mb[0] = t;
}
#endif
#endif

#if defined(BWD_STATS) && defined(TRISC_MATH)
enum : uint32_t { ST_TOT, ST_TILE, ST_REC, ST_SETUP, ST_BODY, ST_FIN, ST_PACK, ST_NT, ST_NP, ST_NV, ST_MARK, ST_N };
uint32_t g_st[ST_N];
inline uint32_t st_now() { return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0]; }
#define ST_T0(v) uint32_t v = st_now()
#define ST_LAP(i, v) do { const uint32_t n_ = st_now(); g_st[i] += n_ - (v); (v) = n_; } while (0)
#define ST_CNT(i, n) g_st[i] += (n)
inline uint32_t bwd_popcount(uint32_t m) {
    uint32_t n = 0;
    for (; m != 0u; m &= m - 1u) n++;
    return n;
}
inline void bwd_emit_stats() {
    g_st[ST_MARK] = 707u;
    for (uint32_t k = 0; k < ST_N; k++) {
        sfpi::dst_reg[DR_ACC + 2 * k] = ckernel::sfpu::Converter::as_float(0x4B000000u | (g_st[k] & 0xFFFFu));
        sfpi::dst_reg[DR_ACC + 2 * k + 1] = ckernel::sfpu::Converter::as_float(0x4B000000u | (g_st[k] >> 16));
    }
}
#else
#define ST_T0(v)
#define ST_LAP(i, v)
#define ST_CNT(i, n)
#endif

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    init_sfpu(CB_PLANE, CB_OUT);
    MATH((bwd_bodies_init<0>()));
    if (n_tiles == 0) return;
    cb_wait_front(CB_RAMP, 2);
#ifdef BWD_SKIPDEAD
    bwd_live_init();
#endif
#if defined(BWD_STATS) && defined(TRISC_MATH)
    for (uint32_t k = 0; k < ST_N; k++) g_st[k] = 0;
#endif
    ST_T0(st_k0);
    for (uint32_t t = 0; t < n_tiles; t++) {
        ST_T0(st_c);
        cb_wait_front(CB_HDR, 1);
        const uint32_t ha = get_tile_address(CB_HDR, 0);  // all threads (mailbox sync)
        asm volatile("fence" ::: "memory");
        const uint32_t cnt = reinterpret_cast<const volatile uint32_t*>(ha)[2];
        cb_pop_front(CB_HDR, 1);
#ifdef BWD_SKIPDEAD
        PACK((g_dead = 0u, g_tl_ok = 0u));
#endif

        tile_regs_acquire();
        cb_wait_front(CB_PLANE, N_PLANES);
        copy_tile_to_dst_init_short(CB_PLANE);
        for (uint32_t k = 0; k < N_PLANES; k++) copy_tile(CB_PLANE, k, k);
        cb_pop_front(CB_PLANE, N_PLANES);
        copy_tile_to_dst_init_short(CB_RAMP);
        copy_tile(CB_RAMP, 0, 5);
        copy_tile(CB_RAMP, 1, 6);
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
        MATH((bwd_consts_init()));
        ST_LAP(ST_TILE, st_c);
        ST_CNT(ST_NT, 1);

        for (uint32_t p = 0; p < cnt; p++) {
            cb_wait_front(CB_REC, 1);
            const uint32_t ra = get_tile_address(CB_REC, 0);  // all threads (mailbox sync)
#ifdef TRISC_MATH
            ST_LAP(ST_REC, st_c);
#ifdef BWD_PRESTAGE
            uint32_t mask = bwd_stage_record(ra);
#else
            uint32_t s[12];
            uint32_t mask = bwd_read_record(ra, s);
            bwd_pair_setup(s);
#endif
#ifdef BWD_SKIPDEAD
            mask &= ~bwd_dead(t);
#endif
            ST_LAP(ST_SETUP, st_c);
            ST_CNT(ST_NP, 1);
            ST_CNT(ST_NV, bwd_popcount(mask));
            bwd_bodies(mask);
            ST_LAP(ST_BODY, st_c);
            sfpi::dst_reg[DR_U] = ckernel::sfpu::Converter::as_float(kColX[p & 7u]);
            if ((p & 8u) == 0u) {
                bwd_finish<0>();
            } else {
                bwd_finish<1>();
            }
            ST_LAP(ST_FIN, st_c);
#else
            (void)ra;
#endif
            UNPACK(((void)ckernel::mailbox_read(ckernel::ThreadId::MathThreadId)));
            cb_pop_front(CB_REC, 1);
#ifdef BWD_SKIPDEAD
            PACK((bwd_scan_block(p & 15u, t)));
#endif
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
#ifdef BWD_SKIPDEAD
                pack_tile(DR_T / 32u, CB_TL);
                cb_push_back(CB_TL, 1);
#endif
                PACK((non_zeroing_pack_release()));
                tile_regs_acquire();
                MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
#ifdef BWD_SKIPDEAD
                PACK((bwd_tl_wait()));
#endif
            }
            ST_LAP(ST_PACK, st_c);
        }
    }
    cb_pop_front(CB_RAMP, 2);
#ifdef BWD_STATS
    ST_LAP(ST_TOT, st_k0);
    tile_regs_acquire();
    MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
    MATH((bwd_emit_stats()));
    MATH((_llk_math_eltwise_unary_sfpu_done_()));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_OUT, 1);
    pack_tile(ACC_TILE, CB_OUT);
    cb_push_back(CB_OUT, 1);
    tile_regs_release();
#endif
}
