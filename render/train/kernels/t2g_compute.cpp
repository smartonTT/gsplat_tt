// #748: the t697 gather-reduce T2 kernel (origin/ttp/t697 head), kept for the on-device training
// step chain (render/train/train_step.cpp): T1 cannot emit #726's block-float pieces (docs/t740).
// T2 param pass compute (#697): segment reduce of the rank planes, projection forward recompute
// + backward to means / log_scales / quats / opacity logit / SH0, sparse Adam (fp32 math, bf16
// RNE moments rounded on the SFPU), pfwc epilogue. Model: gsplat/t2_param_backward.py; layout:
// gsplat/t2_dev.py (one tile = 64 Gaussians x 16 streams, stream s = SFPU vectors 2s, 2s+1).
// Per group: Dst tile 0 params, 1 m, 2 v, 3 g2d (then pfwc out), 4 rank plane (then grads),
// 5-7 scratch. The body uses even vectors only; it runs once, dst_reg++, once more (lanes 32-63).
// Runtime args: group count, then NCONST fp32 constants (bits, t2_dev.consts order).
// #784: T2_PHASE_A sums a heavy group's planes only and packs the sums (tile 3) to CB_FO
// (t2h_writer). T2_G2D_ONLY (check runs) skips the body, so the pfwc pages carry g2d.
// T2_SH (#751, sh_degree > 0): after the reduce, tile 4 holds the t2sh pre pass aux page
// (t2_dev.py: d raw_c / d mean_k, keep, dir); dL/dmeans gains the view-direction term
// sum_c dcolor_c J_c, dL/dsh0 uses the full-colour clamp mask, grads stream 14 = visible flag
// (the t2sh post pass gates its sparse Adam on it). The pfwc colour stays band 0.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_AUX = 7;
constexpr uint32_t CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_FO = 19, CB_GO = 20;
constexpr uint32_t NCONST = 31;
}  // namespace

#ifdef TRISC_MATH
namespace {
using namespace sfpi;
// Dst vector index of stream s of tile T; scratch slot n (tiles 5-7, 48 slots).
constexpr uint32_t TP = 0, TM = 32, TV = 64, TG = 96, TK = 128;
constexpr uint32_t S(uint32_t n) { return 160 + 2 * n; }
// Extra free slots: params/m/v streams 14, 15 and g2d streams 9..15.
constexpr uint32_t E0 = TP + 28, E1 = TP + 30, E2 = TG + 18;
constexpr uint32_t CT0 = TG + 20, CT1 = TG + 26;  // 3 each, stride 2 (g2d streams 10..15)

// Constants (t2_dev.consts order).
enum : uint32_t {
    CR0 = 0, T0C = 9, FX = 12, FY, CX, CY, WW, HH, NEAR, LOWPASS, SS0, B1 = 25, OMB1, B2, OMB2,
    ISQ, CEPS
};
float g_c[NCONST];
sfpi_inline vFloat K(uint32_t i) { return vFloat(g_c[i]); }

sfpi_inline vFloat t2_recip(vFloat x) {
    vFloat y = approx_recip(x);
    y = y * (2.0f - x * y);
    return y * (2.0f - x * y);
}

// 1/sqrt(x), x > 0: bit-trick seed + 3 Newton steps (fp32-limited).
sfpi_inline vFloat t2_rsqrt(vFloat x) {
    vUInt i = reinterpret<vUInt>(x);
    vUInt magic = 0x5f3759df;
    vFloat y = reinterpret<vFloat>(magic - (i >> 1));
    vFloat h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y * (1.5f - h * y * y);
}

sfpi_inline vFloat t2_sqrt(vFloat x) {
    vFloat r = 0.0f;
    v_if(x > 0.0f) { r = x * t2_rsqrt(x); }
    v_endif;
    return r;
}

// fp32-accurate exp (blend_bwd_compute.cpp bwd_exp, tt-metal _sfpu_exp_fp32_accurate_).
sfpi_inline vFloat t2_exp(vFloat a) {
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

// ceil(3 sqrt(max(a, 0))), exact at any size: #750's hero view has visible Gaussians whose
// centre is 40k px off screen, so a radius clamp (was 30000) wrongly culled them from Adam.
// Up to 30000 via int16; up to 2^23 via the 2^23 add/subtract (floor or nearest under any
// rounding, then the +1 fix-up); above 2^23 every fp32 is already an integer.
sfpi_inline vFloat t2_radius(vFloat a) {
    vFloat r = 3.0f * t2_sqrt(a);
    vFloat f = r;
    v_if(r <= 30000.0f) {
        vInt i = float_to_int16(r, RoundMode::NearestEven);
        f = int32_to_float(i, RoundMode::NearestEven);
    }
    v_elseif(r < 8388608.0f) { f = (r + 8388608.0f) - 8388608.0f; }
    v_endif;
    v_if(f < r) { f = f + 1.0f; }
    v_endif;
    return f;
}

sfpi_inline vFloat t2_bf16(vFloat x) {
    vUInt u = reinterpret<vUInt>(x);
    vUInt one = 1;
    vUInt lsb = (u >> 16) & one;
    vUInt bias = 0x7FFF;
    u = u + bias + lsb;
    vUInt mask = 0xFFFF0000;
    return reinterpret<vFloat>(u & mask);
}

// Scratch slots of the forward.
enum : uint32_t {
    X = 0, Y, GA, IZ, VAL, VIS, A_, B_, C_, J00, J02, J11, J12, QW, QX, QY, QZ, QINV,
    RQ = 18, SC = 27, MM = 30, C3 = 39, TT0 = 45
};
constexpr uint32_t TT1[3] = {E0, E1, E2};

// Quaternion normalise, rotation, scales, M = Rq diag(s), C3 = M M^T (6 unique, stride 2 at C3OUT).
template <uint32_t C3OUT>
sfpi_inline void t2_cov() {
    {
        vFloat qw = dst_reg[TP + 12], qx = dst_reg[TP + 14], qy = dst_reg[TP + 16], qz = dst_reg[TP + 18];
        vFloat qi = t2_rsqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        dst_reg[S(QINV)] = qi;
        dst_reg[S(QW)] = qw * qi;
        dst_reg[S(QX)] = qx * qi;
        dst_reg[S(QY)] = qy * qi;
        dst_reg[S(QZ)] = qz * qi;
    }
    {
        vFloat w = dst_reg[S(QW)], x = dst_reg[S(QX)], y = dst_reg[S(QY)], z = dst_reg[S(QZ)];
        dst_reg[S(RQ + 0)] = 1.0f - 2.0f * (y * y + z * z);
        dst_reg[S(RQ + 1)] = 2.0f * (x * y - w * z);
        dst_reg[S(RQ + 2)] = 2.0f * (x * z + w * y);
        dst_reg[S(RQ + 3)] = 2.0f * (x * y + w * z);
        dst_reg[S(RQ + 4)] = 1.0f - 2.0f * (x * x + z * z);
        dst_reg[S(RQ + 5)] = 2.0f * (y * z - w * x);
        dst_reg[S(RQ + 6)] = 2.0f * (x * z - w * y);
        dst_reg[S(RQ + 7)] = 2.0f * (y * z + w * x);
        dst_reg[S(RQ + 8)] = 1.0f - 2.0f * (x * x + y * y);
    }
#pragma GCC unroll 3
    for (uint32_t j = 0; j < 3; j++) {
        vFloat s = t2_exp(dst_reg[TP + 6 + 2 * j]);
        dst_reg[S(SC + j)] = s;
        dst_reg[S(MM + j)] = dst_reg[S(RQ + j)] * s;
        dst_reg[S(MM + 3 + j)] = dst_reg[S(RQ + 3 + j)] * s;
        dst_reg[S(MM + 6 + j)] = dst_reg[S(RQ + 6 + j)] * s;
    }
    constexpr uint32_t ii[6] = {0, 0, 0, 1, 1, 2}, kk[6] = {0, 1, 2, 1, 2, 2};
#pragma GCC unroll 6
    for (uint32_t u = 0; u < 6; u++) {
        dst_reg[C3OUT + 2 * u] = dst_reg[S(MM + 3 * ii[u])] * dst_reg[S(MM + 3 * kk[u])] +
                                 dst_reg[S(MM + 3 * ii[u] + 1)] * dst_reg[S(MM + 3 * kk[u] + 1)] +
                                 dst_reg[S(MM + 3 * ii[u] + 2)] * dst_reg[S(MM + 3 * kk[u] + 2)];
    }
}

constexpr uint32_t c3i(uint32_t i, uint32_t k) {
    // (i, k) -> unique index 0..5 of (00 01 02 11 12 22)
    return i <= k ? (i == 0 ? k : (i == 1 ? 2 + k : 5)) : c3i(k, i);
}

__attribute__((noinline)) void t2_body() {
    // ---- forward: camera, J, cov3d, cov2d, conic, visibility
    {
        vFloat m0 = dst_reg[TP + 0], m1 = dst_reg[TP + 2], m2 = dst_reg[TP + 4];
        dst_reg[S(X)] = K(CR0 + 0) * m0 + K(CR0 + 1) * m1 + K(CR0 + 2) * m2 + K(T0C + 0);
        dst_reg[S(Y)] = K(CR0 + 3) * m0 + K(CR0 + 4) * m1 + K(CR0 + 5) * m2 + K(T0C + 1);
        vFloat z = K(CR0 + 6) * m0 + K(CR0 + 7) * m1 + K(CR0 + 8) * m2 + K(T0C + 2);
        vFloat val = 0.0f;
        vFloat zz = 1.0f;
        v_if(z > K(NEAR)) {
            val = 1.0f;
            zz = z;
        }
        v_endif;
        dst_reg[S(VAL)] = val;
        dst_reg[S(IZ)] = t2_recip(zz);
    }
    {
        vFloat x = dst_reg[S(X)], y = dst_reg[S(Y)], iz = dst_reg[S(IZ)];
        vFloat fx = K(FX), fy = K(FY);
        dst_reg[S(J00)] = fx * iz;
        dst_reg[S(J02)] = -fx * x * iz * iz;
        dst_reg[S(J11)] = fy * iz;
        dst_reg[S(J12)] = -fy * y * iz * iz;
    }
    t2_cov<S(C3)>();
#pragma GCC unroll 3
    for (uint32_t k = 0; k < 3; k++) {
        dst_reg[S(TT0 + k)] = dst_reg[S(J00)] * K(CR0 + k) + dst_reg[S(J02)] * K(CR0 + 6 + k);
        dst_reg[TT1[k]] = dst_reg[S(J11)] * K(CR0 + 3 + k) + dst_reg[S(J12)] * K(CR0 + 6 + k);
    }
#pragma GCC unroll 3
    for (uint32_t i = 0; i < 3; i++) {
        vFloat c0 = dst_reg[S(C3 + c3i(i, 0))], c1 = dst_reg[S(C3 + c3i(i, 1))], c2 = dst_reg[S(C3 + c3i(i, 2))];
        dst_reg[CT0 + 2 * i] = c0 * dst_reg[S(TT0 + 0)] + c1 * dst_reg[S(TT0 + 1)] + c2 * dst_reg[S(TT0 + 2)];
        dst_reg[CT1 + 2 * i] = c0 * dst_reg[TT1[0]] + c1 * dst_reg[TT1[1]] + c2 * dst_reg[TT1[2]];
    }
    {
        vFloat a = dst_reg[S(TT0 + 0)] * dst_reg[CT0 + 0] + dst_reg[S(TT0 + 1)] * dst_reg[CT0 + 2] +
                   dst_reg[S(TT0 + 2)] * dst_reg[CT0 + 4] + K(LOWPASS);
        vFloat b = dst_reg[TT1[0]] * dst_reg[CT0 + 0] + dst_reg[TT1[1]] * dst_reg[CT0 + 2] +
                   dst_reg[TT1[2]] * dst_reg[CT0 + 4];
        vFloat c = dst_reg[TT1[0]] * dst_reg[CT1 + 0] + dst_reg[TT1[1]] * dst_reg[CT1 + 2] +
                   dst_reg[TT1[2]] * dst_reg[CT1 + 4] + K(LOWPASS);
        vFloat idet = t2_recip(a * c - b * b);
        dst_reg[S(A_)] = c * idet;
        dst_reg[S(B_)] = -b * idet;
        dst_reg[S(C_)] = a * idet;
        dst_reg[S(GA)] = a;  // a, c parked until the radii are done
        dst_reg[S(VIS)] = c;
    }
    {
        vFloat rx = t2_radius(dst_reg[S(GA)]);
        vFloat ry = t2_radius(dst_reg[S(VIS)]);
        vFloat x = dst_reg[S(X)], y = dst_reg[S(Y)], iz = dst_reg[S(IZ)];
        vFloat mx = K(FX) * x * iz + K(CX), my = K(FY) * y * iz + K(CY);
        vFloat vis = dst_reg[S(VAL)];
        v_if(mx + rx <= 0.0f) { vis = 0.0f; }
        v_endif;
        v_if(mx - rx >= K(WW)) { vis = 0.0f; }
        v_endif;
        v_if(my + ry <= 0.0f) { vis = 0.0f; }
        v_endif;
        v_if(my - ry >= K(HH)) { vis = 0.0f; }
        v_endif;
        dst_reg[S(VIS)] = vis;
    }
    // ---- backward: conic -> cov2d
    {
        vFloat A = dst_reg[S(A_)], B = dst_reg[S(B_)], C = dst_reg[S(C_)];
        vFloat gA = dst_reg[TG + 4], gB = dst_reg[TG + 6], gC = dst_reg[TG + 8];
        dst_reg[S(GA)] = -(A * A * gA + A * B * gB + B * B * gC);
        dst_reg[S(C3 + 0)] = -(B * B * gA + B * C * gB + C * C * gC);                       // gc
        dst_reg[S(C3 + 1)] = 0.5f * -(2.0f * A * B * gA + (A * C + B * B) * gB + 2.0f * B * C * gC);  // hb
    }
    // dT = 2 Gs T C3 (in place over CT0 / CT1), then gJ (C3 + 2..5 = gj00 gj02 gj11 gj12).
#pragma GCC unroll 3
    for (uint32_t k = 0; k < 3; k++) {
        vFloat ga = dst_reg[S(GA)], gc = dst_reg[S(C3 + 0)], hb = dst_reg[S(C3 + 1)];
        vFloat t0 = dst_reg[CT0 + 2 * k], t1 = dst_reg[CT1 + 2 * k];
        dst_reg[CT0 + 2 * k] = 2.0f * (ga * t0 + hb * t1);
        dst_reg[CT1 + 2 * k] = 2.0f * (hb * t0 + gc * t1);
    }
    {
        vFloat d0 = dst_reg[CT0 + 0], d1 = dst_reg[CT0 + 2], d2 = dst_reg[CT0 + 4];
        dst_reg[S(C3 + 2)] = d0 * K(CR0 + 0) + d1 * K(CR0 + 1) + d2 * K(CR0 + 2);
        dst_reg[S(C3 + 3)] = d0 * K(CR0 + 6) + d1 * K(CR0 + 7) + d2 * K(CR0 + 8);
        vFloat e0 = dst_reg[CT1 + 0], e1 = dst_reg[CT1 + 2], e2 = dst_reg[CT1 + 4];
        dst_reg[S(C3 + 4)] = e0 * K(CR0 + 3) + e1 * K(CR0 + 4) + e2 * K(CR0 + 5);
        dst_reg[S(C3 + 5)] = e0 * K(CR0 + 6) + e1 * K(CR0 + 7) + e2 * K(CR0 + 8);
    }
    // means: gx gy gz parked in A_ B_ C_ (free now)
    {
        vFloat iz = dst_reg[S(IZ)];
        vFloat iz2 = iz * iz;
        dst_reg[S(A_)] = dst_reg[TG + 0] * K(FX) * iz - dst_reg[S(C3 + 3)] * K(FX) * iz2;
        dst_reg[S(B_)] = dst_reg[TG + 2] * K(FY) * iz - dst_reg[S(C3 + 5)] * K(FY) * iz2;
    }
    {
        vFloat iz = dst_reg[S(IZ)], x = dst_reg[S(X)], y = dst_reg[S(Y)];
        vFloat iz2 = iz * iz;
        vFloat t = dst_reg[TG + 0] * K(FX) * x + dst_reg[TG + 2] * K(FY) * y + dst_reg[S(C3 + 2)] * K(FX) +
                   dst_reg[S(C3 + 4)] * K(FY);
        vFloat u = dst_reg[S(C3 + 3)] * K(FX) * x + dst_reg[S(C3 + 5)] * K(FY) * y;
        dst_reg[S(C_)] = -t * iz2 + 2.0f * u * iz2 * iz;
    }
    {
        vFloat val = dst_reg[S(VAL)];
        v_if(val < 0.5f) {
            dst_reg[S(A_)] = 0.0f;
            dst_reg[S(B_)] = 0.0f;
            dst_reg[S(C_)] = 0.0f;
        }
        v_endif;
    }
#pragma GCC unroll 3
    for (uint32_t k = 0; k < 3; k++) {
#ifdef T2_SH
        // aux stream 3 c + k is read before stream k is overwritten (k ascending).
        vFloat dir = dst_reg[TG + 12] * dst_reg[TK + 2 * k];
        dir = dir + dst_reg[TG + 14] * dst_reg[TK + 6 + 2 * k];
        dir = dir + dst_reg[TG + 16] * dst_reg[TK + 12 + 2 * k];
        dst_reg[TK + 2 * k] = K(CR0 + k) * dst_reg[S(A_)] + K(CR0 + 3 + k) * dst_reg[S(B_)] + K(CR0 + 6 + k) * dst_reg[S(C_)] + dir;
#else
        dst_reg[TK + 2 * k] = K(CR0 + k) * dst_reg[S(A_)] + K(CR0 + 3 + k) * dst_reg[S(B_)] + K(CR0 + 6 + k) * dst_reg[S(C_)];
#endif
    }
    // G3 (6 unique) into X Y IZ J00 J02 J11 (free now): G3[i,k] = ga T0i T0k + hb (T0i T1k + T1i T0k) + gc T1i T1k
    constexpr uint32_t G3S[6] = {S(X), S(Y), S(IZ), S(J00), S(J02), S(J11)};
    constexpr uint32_t ii[6] = {0, 0, 0, 1, 1, 2}, kk[6] = {0, 1, 2, 1, 2, 2};
#pragma GCC unroll 6
    for (uint32_t u = 0; u < 6; u++) {
        vFloat a0 = dst_reg[S(TT0 + ii[u])], a1 = dst_reg[S(TT0 + kk[u])];
        vFloat b0 = dst_reg[TT1[ii[u]]], b1 = dst_reg[TT1[kk[u]]];
        vFloat r = dst_reg[S(GA)] * a0 * a1;
        r = r + dst_reg[S(C3 + 1)] * (a0 * b1 + b0 * a1);
        dst_reg[G3S[u]] = r + dst_reg[S(C3 + 0)] * b0 * b1;
    }
    // dM = 2 G3 M per column j (in place over M), gls_j, gR = dM s_j (in place).
#pragma GCC unroll 3
    for (uint32_t j = 0; j < 3; j++) {
        vFloat m0 = dst_reg[S(MM + j)], m1 = dst_reg[S(MM + 3 + j)], m2 = dst_reg[S(MM + 6 + j)];
#pragma GCC unroll 3
        for (uint32_t i = 0; i < 3; i++) {
            dst_reg[S(MM + 3 * i + j)] = 2.0f * (dst_reg[G3S[c3i(i, 0)]] * m0 + dst_reg[G3S[c3i(i, 1)]] * m1 +
                                                 dst_reg[G3S[c3i(i, 2)]] * m2);
        }
        vFloat s = dst_reg[S(SC + j)];
        vFloat d0 = dst_reg[S(MM + j)], d1 = dst_reg[S(MM + 3 + j)], d2 = dst_reg[S(MM + 6 + j)];
        dst_reg[TK + 6 + 2 * j] = (d0 * dst_reg[S(RQ + j)] + d1 * dst_reg[S(RQ + 3 + j)] + d2 * dst_reg[S(RQ + 6 + j)]) * s;
        dst_reg[S(MM + j)] = d0 * s;
        dst_reg[S(MM + 3 + j)] = d1 * s;
        dst_reg[S(MM + 6 + j)] = d2 * s;
    }
    // quaternion: gqn into C3 + 0..3, then projection off qn and 1/|q|.
    {
        vFloat w = dst_reg[S(QW)], qx = dst_reg[S(QX)], qy = dst_reg[S(QY)], qz = dst_reg[S(QZ)];
#define GR(n) dst_reg[S(MM + n)]
        dst_reg[S(C3 + 0)] = 2.0f * (-qz * GR(1) + qy * GR(2) + qz * GR(3) - qx * GR(5) - qy * GR(6) + qx * GR(7));
        dst_reg[S(C3 + 1)] = 2.0f * (qy * GR(1) + qz * GR(2) + qy * GR(3) - 2.0f * qx * GR(4) - w * GR(5) +
                                     qz * GR(6) + w * GR(7) - 2.0f * qx * GR(8));
        dst_reg[S(C3 + 2)] = 2.0f * (-2.0f * qy * GR(0) + qx * GR(1) + w * GR(2) + qx * GR(3) + qz * GR(5) -
                                     w * GR(6) + qz * GR(7) - 2.0f * qy * GR(8));
        dst_reg[S(C3 + 3)] = 2.0f * (-2.0f * qz * GR(0) - w * GR(1) + qx * GR(2) + w * GR(3) - 2.0f * qz * GR(4) +
                                     qy * GR(5) + qx * GR(6) + qy * GR(7));
#undef GR
    }
    {
        vFloat dot = dst_reg[S(QW)] * dst_reg[S(C3 + 0)] + dst_reg[S(QX)] * dst_reg[S(C3 + 1)] +
                     dst_reg[S(QY)] * dst_reg[S(C3 + 2)] + dst_reg[S(QZ)] * dst_reg[S(C3 + 3)];
        vFloat qi = dst_reg[S(QINV)];
#pragma GCC unroll 4
        for (uint32_t i = 0; i < 4; i++) {
            dst_reg[TK + 12 + 2 * i] = (dst_reg[S(C3 + i)] - dst_reg[S(QW + i)] * dot) * qi;
        }
    }
    // opacity, SH0
    {
        vFloat op = t2_recip(1.0f + t2_exp(-vFloat(dst_reg[TP + 20])));
        dst_reg[TK + 20] = dst_reg[TG + 10] * op * (1.0f - op);
    }
#pragma GCC unroll 3
    for (uint32_t c = 0; c < 3; c++) {
        constexpr float SH_C0 = 0.28209479177387814f;
        vFloat g = SH_C0 * dst_reg[TG + 12 + 2 * c];
#ifdef T2_SH
        // full-colour clamp mask from the pre pass (aux stream 11 + c, in place)
        v_if(dst_reg[TK + 22 + 2 * c] < 0.5f) { g = 0.0f; }
        v_endif;
#else
        // 3DGS/gsplat training colour max(SH + 0.5, 0): no upper clamp (#763 decision).
        vFloat raw = SH_C0 * dst_reg[TP + 22 + 2 * c] + 0.5f;
        v_if(raw < 0.0f) { g = 0.0f; }
        v_endif;
#endif
        dst_reg[TK + 22 + 2 * c] = g;
    }
#ifdef T2_SH
    dst_reg[TK + 28] = vFloat(dst_reg[S(VIS)]);
#else
    dst_reg[TK + 28] = 0.0f;
#endif
    dst_reg[TK + 30] = 0.0f;
    // ---- sparse Adam over the 14 param streams (visible lanes only)
    constexpr uint32_t grp[14] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 4, 4, 4};
#pragma GCC unroll 14
    for (uint32_t s = 0; s < 14; s++) {
        vFloat vis = dst_reg[S(VIS)];
        v_if(vis > 0.5f) {
            vFloat g = dst_reg[TK + 2 * s];
            dst_reg[TM + 2 * s] = K(B1) * dst_reg[TM + 2 * s] + K(OMB1) * g;
            dst_reg[TV + 2 * s] = K(B2) * dst_reg[TV + 2 * s] + K(OMB2) * g * g;
        }
        v_endif;
        v_if(vis > 0.5f) {
            vFloat den = t2_sqrt(dst_reg[TV + 2 * s]) * K(ISQ) + K(CEPS);
            dst_reg[TP + 2 * s] = dst_reg[TP + 2 * s] - K(SS0 + grp[s]) * (dst_reg[TM + 2 * s] * t2_recip(den));
            dst_reg[TM + 2 * s] = t2_bf16(dst_reg[TM + 2 * s]);
            dst_reg[TV + 2 * s] = t2_bf16(dst_reg[TV + 2 * s]);
        }
        v_endif;
    }
    // ---- pfwc epilogue from the updated params (into the g2d tile)
    dst_reg[TG + 0] = vFloat(dst_reg[TP + 0]);
    dst_reg[TG + 2] = vFloat(dst_reg[TP + 2]);
    dst_reg[TG + 4] = vFloat(dst_reg[TP + 4]);
    t2_cov<TG + 6>();
    dst_reg[TG + 18] = t2_recip(1.0f + t2_exp(-vFloat(dst_reg[TP + 20])));
#pragma GCC unroll 3
    for (uint32_t c = 0; c < 3; c++) {
        constexpr float SH_C0 = 0.28209479177387814f;
        vFloat raw = SH_C0 * dst_reg[TP + 22 + 2 * c] + 0.5f;
        v_if(raw < 0.0f) { raw = 0.0f; }
        v_endif;
        dst_reg[TG + 20 + 2 * c] = raw;
    }
}

// Rank plane (tile 4) into the g2d accumulator (tile 3): 9 streams, both parities.
inline void t2_add_plane() {
#pragma GCC unroll 18
    for (uint32_t v = 0; v < 18; v++) dst_reg[TG + v] = vFloat(dst_reg[TG + v]) + vFloat(dst_reg[TK + v]);
}
}  // namespace
#endif

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    MATH(({
        for (uint32_t i = 0; i < NCONST; i++) g_c[i] = ckernel::sfpu::Converter::as_float(get_arg_val<uint32_t>(1 + i));
    }));
    if (n_groups == 0) return;
#ifdef T2_PHASE_A
    init_sfpu(CB_K, CB_FO);
    for (uint32_t g = 0; g < n_groups; g++) {
        cb_wait_front(CB_HDR, 1);
        const uint32_t ha = get_tile_address(CB_HDR, 0);  // all threads (mailbox sync)
        asm volatile("fence" ::: "memory");
        const uint32_t nk = reinterpret_cast<const volatile uint32_t*>(ha)[0];
        cb_pop_front(CB_HDR, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_K);
        cb_wait_front(CB_K, 1);
        copy_tile(CB_K, 0, 3);
        cb_pop_front(CB_K, 1);
        for (uint32_t k = 1; k < nk; k++) {
            cb_wait_front(CB_K, 1);
            copy_tile(CB_K, 0, 4);
            cb_pop_front(CB_K, 1);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((t2_add_plane()));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_FO, 1);
        pack_tile(3, CB_FO);
        cb_push_back(CB_FO, 1);
        tile_regs_release();
    }
    return;
#endif
    init_sfpu(CB_P, CB_PO);
    for (uint32_t g = 0; g < n_groups; g++) {
        cb_wait_front(CB_HDR, 1);
        const uint32_t ha = get_tile_address(CB_HDR, 0);  // all threads (mailbox sync)
        asm volatile("fence" ::: "memory");
        const uint32_t nk = reinterpret_cast<const volatile uint32_t*>(ha)[0];
        cb_pop_front(CB_HDR, 1);

        tile_regs_acquire();
        cb_wait_front(CB_P, 1);
        copy_tile_to_dst_init_short_with_dt(CB_K, CB_P);
        copy_tile(CB_P, 0, 0);
        cb_pop_front(CB_P, 1);
        cb_wait_front(CB_M, 1);
        copy_tile_to_dst_init_short_with_dt(CB_P, CB_M);
        copy_tile(CB_M, 0, 1);
        cb_pop_front(CB_M, 1);
        cb_wait_front(CB_V, 1);
        copy_tile_to_dst_init_short_with_dt(CB_M, CB_V);
        copy_tile(CB_V, 0, 2);
        cb_pop_front(CB_V, 1);
        copy_tile_to_dst_init_short_with_dt(CB_V, CB_K);
        cb_wait_front(CB_K, 1);
        copy_tile(CB_K, 0, 3);
        cb_pop_front(CB_K, 1);
        for (uint32_t k = 1; k < nk; k++) {
            cb_wait_front(CB_K, 1);
            copy_tile(CB_K, 0, 4);
            cb_pop_front(CB_K, 1);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((t2_add_plane()));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
        }
#ifdef T2_SH
        copy_tile_to_dst_init_short_with_dt(CB_K, CB_AUX);
        cb_wait_front(CB_AUX, 1);
        copy_tile(CB_AUX, 0, 4);
        cb_pop_front(CB_AUX, 1);
#endif
#ifndef T2_G2D_ONLY
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
        MATH((t2_body()));
        MATH((sfpi::dst_reg++));
        MATH((t2_body()));
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
#endif
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(CB_PO, 1);
        pack_reconfig_data_format(CB_PO);
        pack_tile(0, CB_PO);
        cb_push_back(CB_PO, 1);
        cb_reserve_back(CB_MO, 1);
        pack_reconfig_data_format(CB_MO);
        pack_tile(1, CB_MO);
        cb_push_back(CB_MO, 1);
        cb_reserve_back(CB_VO, 1);
        pack_tile(2, CB_VO);
        cb_push_back(CB_VO, 1);
        cb_reserve_back(CB_FO, 1);
        pack_reconfig_data_format(CB_FO);
        pack_tile(3, CB_FO);
        cb_push_back(CB_FO, 1);
#ifdef T2_GRADS_OUT
        cb_reserve_back(CB_GO, 1);
        pack_tile(4, CB_GO);
        cb_push_back(CB_GO, 1);
#endif
        tile_regs_release();
    }
}
