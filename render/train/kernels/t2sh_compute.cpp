// #751: SH bands 1-3 around T2 (model: gsplat/t2_param_backward.py sh_lanes / projection
// backward; layout: gsplat/t2_dev.py rest and aux pages). Two modes, same groups as T2 (the
// readers skip groups with no visible lane):
//   pre  (default, before T2, needs no gradients): Dst 0 params, 1-3 rest pages, 4 aux out.
//        dir = (mean - campos) / |.|, raw_c = (SH_C0 sh0_c + 0.5) + sum_j B_j(dir) sh_jc,
//        keep_c = 0 <= raw_c <= 1, J_c = keep_c (grad_d - dir (dir . grad_d)) / |.| with
//        grad_d = sum_j dB_j/d dir sh_jc. T2 (T2_SH) adds sum_c dcolor_c J_c to dL/dmeans and
//        masks dL/dsh0 with keep.
//   post (T2SH_POST, after T2): per rest page r: Dst 0 aux, 1 T2 grads page, 2 rest params,
//        3 m, 4 v, 5 rest grads out. dL/dsh_jc = B_j(dir) dL/dsh0_c / SH_C0 (dL/dsh0 is already
//        SH_C0 dcolor keep), then sparse Adam (sh_rest step) on lanes with grads stream 14 set.
// Stream s of tile T is Dst vector 32 T + 2 s; the body runs on even vectors, dst_reg++, again.
// Runtime args: group count, then NSH fp32 constants (bits, t2_dev.sh_consts order).
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
constexpr uint32_t CB_A = 1, CB_B = 2, CB_M = 3, CB_V = 4, CB_G = 7;
constexpr uint32_t CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_GO = 20;
constexpr uint32_t NSH = 10;
}  // namespace

#ifdef TRISC_MATH
namespace {
using namespace sfpi;
enum : uint32_t { CAMX = 0, CAMY, CAMZ, SS, B1, OMB1, B2, OMB2, ISQ, CEPS };
float g_c[NSH];
sfpi_inline vFloat K(uint32_t i) { return vFloat(g_c[i]); }

constexpr float SH_C0 = 0.28209479177387814f, C1 = 0.4886025119029199f;
constexpr float C20 = 1.0925484305920792f, C21 = -1.0925484305920792f, C22 = 0.31539156525252005f,
                C23 = -1.0925484305920792f, C24 = 0.5462742152960396f;
constexpr float C30 = -0.5900435899266435f, C31 = 2.890611442640554f, C32 = -0.4570457994644658f,
                C33 = 0.3731763325901154f, C34 = -0.4570457994644658f, C35 = 1.445305721320277f,
                C36 = -0.5900435899266435f;

sfpi_inline vFloat sh_recip(vFloat x) {
    vFloat y = approx_recip(x);
    y = y * (2.0f - x * y);
    return y * (2.0f - x * y);
}

sfpi_inline vFloat sh_rsqrt(vFloat x) {
    vUInt i = reinterpret<vUInt>(x);
    vUInt magic = 0x5f3759df;
    vFloat y = reinterpret<vFloat>(magic - (i >> 1));
    vFloat h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y * (1.5f - h * y * y);
}

sfpi_inline vFloat sh_sqrt(vFloat x) {
    vFloat r = 0.0f;
    v_if(x > 0.0f) { r = x * sh_rsqrt(x); }
    v_endif;
    return r;
}

sfpi_inline vFloat sh_bf16(vFloat x) {
    vUInt u = reinterpret<vUInt>(x);
    vUInt one = 1;
    vUInt lsb = (u >> 16) & one;
    vUInt bias = 0x7FFF;
    u = u + bias + lsb;
    vUInt mask = 0xFFFF0000;
    return reinterpret<vFloat>(u & mask);
}

// Real SH basis j = 1..15 at unit dir (x, y, z) (train_oracle.sh_eval).
sfpi_inline vFloat sh_basis(uint32_t j, vFloat x, vFloat y, vFloat z) {
    switch (j) {
        case 1: return -C1 * y;
        case 2: return C1 * z;
        case 3: return -C1 * x;
        case 4: return C20 * x * y;
        case 5: return C21 * y * z;
        case 6: return C22 * (2.0f * z * z - x * x - y * y);
        case 7: return C23 * x * z;
        case 8: return C24 * (x * x - y * y);
        case 9: return C30 * y * (3.0f * x * x - y * y);
        case 10: return C31 * x * y * z;
        case 11: return C32 * y * (4.0f * z * z - x * x - y * y);
        case 12: return C33 * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y);
        case 13: return C34 * x * (4.0f * z * z - x * x - y * y);
        case 14: return C35 * z * (x * x - y * y);
        default: return C36 * x * (x * x - 3.0f * y * y);
    }
}

#ifndef T2SH_POST
constexpr uint32_t TP = 0, TA = 128;
constexpr uint32_t S(uint32_t n) { return 160 + 2 * n; }
// Scratch: unit dir, products, basis 1..15 at BS + j - 1, colour gradient wrt dir.
enum : uint32_t { X = 0, Y, Z, XX, YY, ZZ, XY, YZ, XZ, BS, GX = BS + 15, GY, GZ };
// SH band j colour c in the rest tiles 1-3.
constexpr uint32_t SHV(uint32_t j, uint32_t c) { return 32 * (1 + (j - 1) / 5) + 2 * (3 * ((j - 1) % 5) + c); }

__attribute__((noinline)) void sh_pre_body() {
    {
        vFloat vx = dst_reg[TP + 0] - K(CAMX), vy = dst_reg[TP + 2] - K(CAMY), vz = dst_reg[TP + 4] - K(CAMZ);
        vFloat inv = sh_rsqrt(vx * vx + vy * vy + vz * vz);
        dst_reg[TA + 30] = inv;
        vx = vx * inv;
        vy = vy * inv;
        vz = vz * inv;
        dst_reg[S(X)] = vx;
        dst_reg[S(Y)] = vy;
        dst_reg[S(Z)] = vz;
        dst_reg[TA + 18] = vx;
        dst_reg[TA + 20] = vy;
        dst_reg[TA + 28] = vz;
        dst_reg[S(XX)] = vx * vx;
        dst_reg[S(YY)] = vy * vy;
        dst_reg[S(ZZ)] = vz * vz;
        dst_reg[S(XY)] = vx * vy;
        dst_reg[S(YZ)] = vy * vz;
        dst_reg[S(XZ)] = vx * vz;
    }
#pragma GCC unroll 15
    for (uint32_t j = 1; j <= 15; j++) {
        dst_reg[S(BS + j - 1)] = sh_basis(j, dst_reg[S(X)], dst_reg[S(Y)], dst_reg[S(Z)]);
    }
#pragma GCC unroll 3
    for (uint32_t c = 0; c < 3; c++) {
        {
            vFloat rest = dst_reg[S(BS)] * dst_reg[SHV(1, c)];
#pragma GCC unroll 14
            for (uint32_t j = 2; j <= 15; j++) rest = rest + dst_reg[S(BS + j - 1)] * dst_reg[SHV(j, c)];
            vFloat raw = (SH_C0 * dst_reg[TP + 22 + 2 * c] + 0.5f) + rest;
            vFloat keep = 1.0f;
            v_if(raw < 0.0f) { keep = 0.0f; }
            v_endif;
            v_if(raw > 1.0f) { keep = 0.0f; }
            v_endif;
            dst_reg[TA + 22 + 2 * c] = keep;
        }
        // d (sum_j B_j sh_jc) / d dir, unconstrained (t2_param_backward.sh_dbasis).
        {
            vFloat g = -C1 * dst_reg[SHV(3, c)];
            g = g + C20 * dst_reg[S(Y)] * dst_reg[SHV(4, c)];
            g = g - 2.0f * C22 * dst_reg[S(X)] * dst_reg[SHV(6, c)];
            g = g + C23 * dst_reg[S(Z)] * dst_reg[SHV(7, c)];
            g = g + 2.0f * C24 * dst_reg[S(X)] * dst_reg[SHV(8, c)];
            g = g + 6.0f * C30 * dst_reg[S(XY)] * dst_reg[SHV(9, c)];
            g = g + C31 * dst_reg[S(YZ)] * dst_reg[SHV(10, c)];
            g = g - 2.0f * C32 * dst_reg[S(XY)] * dst_reg[SHV(11, c)];
            g = g - 6.0f * C33 * dst_reg[S(XZ)] * dst_reg[SHV(12, c)];
            g = g + C34 * (4.0f * dst_reg[S(ZZ)] - 3.0f * dst_reg[S(XX)] - dst_reg[S(YY)]) * dst_reg[SHV(13, c)];
            g = g + 2.0f * C35 * dst_reg[S(XZ)] * dst_reg[SHV(14, c)];
            g = g + 3.0f * C36 * (vFloat(dst_reg[S(XX)]) - dst_reg[S(YY)]) * dst_reg[SHV(15, c)];
            dst_reg[S(GX)] = g;
        }
        {
            vFloat g = -C1 * dst_reg[SHV(1, c)];
            g = g + C20 * dst_reg[S(X)] * dst_reg[SHV(4, c)];
            g = g + C21 * dst_reg[S(Z)] * dst_reg[SHV(5, c)];
            g = g - 2.0f * C22 * dst_reg[S(Y)] * dst_reg[SHV(6, c)];
            g = g - 2.0f * C24 * dst_reg[S(Y)] * dst_reg[SHV(8, c)];
            g = g + 3.0f * C30 * (vFloat(dst_reg[S(XX)]) - dst_reg[S(YY)]) * dst_reg[SHV(9, c)];
            g = g + C31 * dst_reg[S(XZ)] * dst_reg[SHV(10, c)];
            g = g + C32 * (4.0f * dst_reg[S(ZZ)] - dst_reg[S(XX)] - 3.0f * dst_reg[S(YY)]) * dst_reg[SHV(11, c)];
            g = g - 6.0f * C33 * dst_reg[S(YZ)] * dst_reg[SHV(12, c)];
            g = g - 2.0f * C34 * dst_reg[S(XY)] * dst_reg[SHV(13, c)];
            g = g - 2.0f * C35 * dst_reg[S(YZ)] * dst_reg[SHV(14, c)];
            g = g - 6.0f * C36 * dst_reg[S(XY)] * dst_reg[SHV(15, c)];
            dst_reg[S(GY)] = g;
        }
        {
            vFloat g = C1 * dst_reg[SHV(2, c)];
            g = g + C21 * dst_reg[S(Y)] * dst_reg[SHV(5, c)];
            g = g + 4.0f * C22 * dst_reg[S(Z)] * dst_reg[SHV(6, c)];
            g = g + C23 * dst_reg[S(X)] * dst_reg[SHV(7, c)];
            g = g + C31 * dst_reg[S(XY)] * dst_reg[SHV(10, c)];
            g = g + 8.0f * C32 * dst_reg[S(YZ)] * dst_reg[SHV(11, c)];
            g = g + C33 * (6.0f * dst_reg[S(ZZ)] - 3.0f * dst_reg[S(XX)] - 3.0f * dst_reg[S(YY)]) * dst_reg[SHV(12, c)];
            g = g + 8.0f * C34 * dst_reg[S(XZ)] * dst_reg[SHV(13, c)];
            g = g + C35 * (vFloat(dst_reg[S(XX)]) - dst_reg[S(YY)]) * dst_reg[SHV(14, c)];
            dst_reg[S(GZ)] = g;
        }
        {
            vFloat dot = dst_reg[S(X)] * dst_reg[S(GX)] + dst_reg[S(Y)] * dst_reg[S(GY)] + dst_reg[S(Z)] * dst_reg[S(GZ)];
            vFloat inv = dst_reg[TA + 30];
            v_if(dst_reg[TA + 22 + 2 * c] < 0.5f) { inv = 0.0f; }
            v_endif;
#pragma GCC unroll 3
            for (uint32_t k = 0; k < 3; k++) {
                dst_reg[TA + 2 * (3 * c + k)] = (dst_reg[S(GX + k)] - dst_reg[S(X + k)] * dot) * inv;
            }
        }
    }
}
#else
constexpr uint32_t TA = 0, TG = 32, TP = 64, TM = 96, TV = 128, TO = 160;
constexpr uint32_t S(uint32_t n) { return 192 + 2 * n; }

__attribute__((noinline)) void sh_post_body(uint32_t r) {
#pragma GCC unroll 5
    for (uint32_t jj = 0; jj < 5; jj++) {
        dst_reg[S(jj)] = sh_basis(5 * r + jj + 1, dst_reg[TA + 18], dst_reg[TA + 20], dst_reg[TA + 28]);
    }
    constexpr float INV_C0 = 1.0f / SH_C0;
#pragma GCC unroll 5
    for (uint32_t jj = 0; jj < 5; jj++) {
#pragma GCC unroll 3
        for (uint32_t c = 0; c < 3; c++) {
            const uint32_t s = 3 * jj + c;
            vFloat g = dst_reg[S(jj)] * (INV_C0 * dst_reg[TG + 22 + 2 * c]);
            dst_reg[TO + 2 * s] = g;
            vFloat vis = dst_reg[TG + 28];
            v_if(vis > 0.5f) {
                dst_reg[TM + 2 * s] = K(B1) * dst_reg[TM + 2 * s] + K(OMB1) * g;
                dst_reg[TV + 2 * s] = K(B2) * dst_reg[TV + 2 * s] + K(OMB2) * g * g;
            }
            v_endif;
            v_if(vis > 0.5f) {
                vFloat den = sh_sqrt(dst_reg[TV + 2 * s]) * K(ISQ) + K(CEPS);
                dst_reg[TP + 2 * s] = dst_reg[TP + 2 * s] - K(SS) * (dst_reg[TM + 2 * s] * sh_recip(den));
                dst_reg[TM + 2 * s] = sh_bf16(dst_reg[TM + 2 * s]);
                dst_reg[TV + 2 * s] = sh_bf16(dst_reg[TV + 2 * s]);
            }
            v_endif;
        }
    }
    dst_reg[TO + 30] = 0.0f;
}
#endif
}  // namespace
#endif

void kernel_main() {
    const uint32_t n_groups = get_arg_val<uint32_t>(0);
    MATH(({
        for (uint32_t i = 0; i < NSH; i++) g_c[i] = ckernel::sfpu::Converter::as_float(get_arg_val<uint32_t>(1 + i));
    }));
    if (n_groups == 0) return;
    init_sfpu(CB_A, CB_PO);
#ifndef T2SH_POST
    for (uint32_t g = 0; g < n_groups; g++) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(CB_B, CB_A);
        cb_wait_front(CB_A, 1);
        copy_tile(CB_A, 0, 0);
        cb_pop_front(CB_A, 1);
        copy_tile_to_dst_init_short_with_dt(CB_A, CB_B);
        cb_wait_front(CB_B, 3);
        for (uint32_t r = 0; r < 3; r++) copy_tile(CB_B, r, 1 + r);
        cb_pop_front(CB_B, 3);
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
        MATH((sh_pre_body()));
        MATH((sfpi::dst_reg++));
        MATH((sh_pre_body()));
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_PO, 1);
        pack_tile(4, CB_PO);
        cb_push_back(CB_PO, 1);
        tile_regs_release();
    }
#else
    for (uint32_t g = 0; g < n_groups; g++) {
        cb_wait_front(CB_A, 1);
        cb_wait_front(CB_G, 1);
        for (uint32_t r = 0; r < 3; r++) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short_with_dt(CB_V, CB_A);
            copy_tile(CB_A, 0, 0);
            copy_tile_to_dst_init_short_with_dt(CB_A, CB_G);
            copy_tile(CB_G, 0, 1);
            copy_tile_to_dst_init_short_with_dt(CB_G, CB_B);
            cb_wait_front(CB_B, 1);
            copy_tile(CB_B, 0, 2);
            cb_pop_front(CB_B, 1);
            copy_tile_to_dst_init_short_with_dt(CB_B, CB_M);
            cb_wait_front(CB_M, 1);
            copy_tile(CB_M, 0, 3);
            cb_pop_front(CB_M, 1);
            copy_tile_to_dst_init_short_with_dt(CB_M, CB_V);
            cb_wait_front(CB_V, 1);
            copy_tile(CB_V, 0, 4);
            cb_pop_front(CB_V, 1);
            MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
            MATH((sh_post_body(r)));
            MATH((sfpi::dst_reg++));
            MATH((sh_post_body(r)));
            MATH((_llk_math_eltwise_unary_sfpu_done_()));
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(CB_PO, 1);
            pack_reconfig_data_format(CB_PO);
            pack_tile(2, CB_PO);
            cb_push_back(CB_PO, 1);
            cb_reserve_back(CB_MO, 1);
            pack_reconfig_data_format(CB_MO);
            pack_tile(3, CB_MO);
            cb_push_back(CB_MO, 1);
            cb_reserve_back(CB_VO, 1);
            pack_tile(4, CB_VO);
            cb_push_back(CB_VO, 1);
            cb_reserve_back(CB_GO, 1);
            pack_reconfig_data_format(CB_GO);
            pack_tile(5, CB_GO);
            cb_push_back(CB_GO, 1);
            tile_regs_release();
        }
        cb_pop_front(CB_A, 1);
        cb_pop_front(CB_G, 1);
    }
#endif
}
