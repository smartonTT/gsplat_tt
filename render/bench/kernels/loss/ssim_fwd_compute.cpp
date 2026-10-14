// Task #736 (T7): fused L1 + D-SSIM, forward half. Per output tile (channel, tile row I, tile
// column J): the three per-step filtered maps mu_x = G*x, E_xx = G*x^2, E_xy = G*(x y) on the
// SFPU in fp32 (the #720 filter, ssim_sfpu.h), the SSIM map, the loss partial and the three
// backward inputs d_mu, d_xx, d_xy (gsplat/train_loss.py loss_and_grad_device). Only the d maps
// leave the core (9 planes); the 15 filtered planes and pointwise terms stay in L1 / dest.
// Per-view constants (cached per GT image): mu_y, vy = G*(y y) - mu_y^2, wm = -lambda/(3HW) * mask.
// The loss partial per core: sum(S wm + wl1 |x - y|); loss = lambda + sum over cores.
// Same strip walk as sfilt_compute.cpp (H pass per tile row into 3 rotating slots, V pass).
#include "ssim_sfpu.h"

// u2d: every CB the compute copies from.
constexpr uint32_t CB_X = 0, CB_Y = 1, CB_C2 = 2, CB_C3 = 3, CB_HX = 6, CB_HXX = 9, CB_HXY = 12,
                   CB_D = 16, CB_M = 17, CB_F = 18, CB_ACC = 19, CB_SUM = 20;

#ifdef TRISC_MATH
// H pass: 0..2 = x^T, 4..6 = y^T -> 4 = G x, 3 = G x^2, 7 = G (x y) (transposed layout).
inline void h_sfpu() {
    using namespace sfpi;
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32 * !(SSIM_ABL & 4); v++) {
        dst_reg[128] = vFloat(dst_reg[0]) * dst_reg[128];
        dst_reg[160] = vFloat(dst_reg[32]) * dst_reg[160];
        dst_reg[192] = vFloat(dst_reg[64]) * dst_reg[192];
        dst_reg++;
    }
    sfpu_end();
    sfpu_begin();
    vfilt_body<4, 7>();
    sfpu_end();
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32 * !(SSIM_ABL & 4); v++) {
        vFloat a = dst_reg[0];
        dst_reg[128] = a * a;
        a = dst_reg[32];
        dst_reg[160] = a * a;
        a = dst_reg[64];
        dst_reg[192] = a * a;
        dst_reg++;
    }
    sfpu_end();
    sfpu_begin();
    vfilt_body<4, 3>();
    vfilt_body<0, 4>();
    sfpu_end();
}

// XY = E_xy, 4 = mu_x, XX = E_xx, 6 = mu_y, 7 = vy ({XY, XX} = {3, 5}) -> 0 = S, 1 = F_mu,
// 2 = F_xx, 7 = F_xy with d_mu = Sw F_mu, d_xx = Sw F_xx, d_xy = Sw F_xy (Sw = S wm, next block).
template <uint32_t XY = 3, uint32_t XX = 5>
inline void ssim_pt() {
    using namespace sfpi;
    if constexpr (SSIM_ABL & 4) return;
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        {
            const vFloat mux = dst_reg[128], muy = dst_reg[192];
            const vFloat mxy = mux * muy;
            dst_reg[0] = mxy * 2.0f + SS_C1;                                     // A
            dst_reg[32] = (vFloat(dst_reg[32 * XY]) - mxy) * 2.0f + SS_C2;       // B
            dst_reg[64] = mux * mux + muy * muy + SS_C1;                         // C
            dst_reg[32 * XX] = (vFloat(dst_reg[32 * XX]) - mux * mux) + dst_reg[224] + SS_C2;  // D
        }
        const vFloat rB = rcp(dst_reg[32]);
        dst_reg[224] = rB * 2.0f;  // F_xy
        {
            const vFloat rA = rcp(dst_reg[0]);
            dst_reg[32 * XY] = vFloat(dst_reg[192]) * (rA - rB);
        }
        const vFloat rC = rcp(dst_reg[64]), rD = rcp(dst_reg[32 * XX]);
        dst_reg[0] = (vFloat(dst_reg[0]) * dst_reg[32]) * (rC * rD);           // S
        dst_reg[32] = (vFloat(dst_reg[32 * XY]) + vFloat(dst_reg[128]) * (rD - rC)) * 2.0f;  // F_mu
        dst_reg[64] = -rD;                                                     // F_xx
        dst_reg++;
    }
    sfpu_end();
}

// 0 = S, 1, 2, FXY = F_mu, F_xx, F_xy, 4 = wm, 5 = x, 6 = y, AC = acc ({FXY, AC} = {3, 7})
// -> 1, 2, FXY = d maps, AC += loss.
template <uint32_t FXY = 3, uint32_t AC = 7>
inline void loss_pt() {
    using namespace sfpi;
    if constexpr (SSIM_ABL & 4) return;
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        const vFloat sw = vFloat(dst_reg[0]) * dst_reg[128];
        dst_reg[32] = sw * dst_reg[32];
        dst_reg[64] = sw * dst_reg[64];
        dst_reg[32 * FXY] = sw * dst_reg[32 * FXY];
        const vFloat d = vFloat(dst_reg[160]) - dst_reg[192];
        dst_reg[32 * AC] = vFloat(dst_reg[32 * AC]) + (sw + sfpi::abs(d) * SS_WL1);
        dst_reg++;
    }
    sfpu_end();
}
#endif

inline void wait3(uint32_t cb0, uint32_t a, uint32_t b, uint32_t c) {
    cb_wait_front(cb0 + a, SSIM_K);
    cb_wait_front(cb0 + b, SSIM_K);
    cb_wait_front(cb0 + c, SSIM_K);
}
inline void copy3(uint32_t cb0, uint32_t a, uint32_t b, uint32_t c, uint32_t k, uint32_t d0) {
    copy_tile(cb0 + a, k, d0);
    copy_tile(cb0 + b, k, d0 + 1);
    copy_tile(cb0 + c, k, d0 + 2);
}

// H pass of one tile row of a kw-column strip: column k filters row pages k .. k+2 of CB_X / CB_Y
// (SSIM_K + 2 per row) into page k of the slot q CBs (SSIM_K pages, one reserve / push per row).
inline void h_row(uint32_t q, uint32_t kw) {
    cb_wait_front(CB_X, SSIM_K + 2);
    cb_wait_front(CB_Y, SSIM_K + 2);
    cb_reserve_back(CB_HX + q, SSIM_K);
    cb_reserve_back(CB_HXX + q, SSIM_K);
    cb_reserve_back(CB_HXY + q, SSIM_K);
    for (uint32_t k = 0; k < kw; k++) {
        tile_regs_acquire();
        copy_at(CB_X, k, 3, 0);
        copy_at(CB_Y, k, 3, 4);
        transpose_tiles(0x77);
        MATH((h_sfpu()));
        transpose_tiles(0x98);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(4, CB_HX + q, k);
        pack_tile<true>(3, CB_HXX + q, k);
        pack_tile<true>(7, CB_HXY + q, k);
        release();
    }
    cb_pop_front(CB_X, SSIM_K + 2);
    cb_pop_front(CB_Y, SSIM_K + 2);
    cb_push_back(CB_HX + q, SSIM_K);
    cb_push_back(CB_HXX + q, SSIM_K);
    cb_push_back(CB_HXY + q, SSIM_K);
}

inline void push_zero(uint32_t q, uint32_t kw) {
    tile_regs_acquire();
    MATH((zero_tile<3>()));
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t cb : {CB_HX, CB_HXX, CB_HXY}) {
        cb_reserve_back(cb + q, SSIM_K);
        for (uint32_t k = 0; k < kw; k++) pack_tile<true>(3, cb + q, k);
        cb_push_back(cb + q, SSIM_K);
    }
    release();
}

// V pass of output column k (page k of the slots a, b, c = rows I-1, I, I+1).
#if SSIM_VF
// #753 SSIM_VF: the whole V pass in one dest acquire (no CB_M / CB_F round trips): 0..2 / 5..7 =
// mu_x / E_xx rows -> 4 = mu_x, 3 = E_xx; 0..2 = E_xy rows, 6, 7 = mu_y, vy -> 5 = E_xy;
// ssim_pt -> 0, 1, 2, 7; 4..6 = wm, x, y, 3 = acc -> loss_pt. Same fp32 ops (all CBs fp32).
inline void v_pass(uint32_t a, uint32_t b, uint32_t c, uint32_t k) {
    wait3(CB_HX, a, b, c);
    wait3(CB_HXX, a, b, c);
    wait3(CB_HXY, a, b, c);
    cb_wait_front(CB_C2, 2);
    cb_wait_front(CB_C3, 3);
    cb_wait_front(CB_ACC, 1);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HX + a);
    copy3(CB_HX, a, b, c, k, 0);
    copy3(CB_HXX, a, b, c, k, 5);
    MATH((sfpu_begin(), vfilt_body<0, 4>(), vfilt_body<5, 3>(), sfpu_end()));
    copy_tile_to_dst_init_short(CB_HXY + a);
    copy3(CB_HXY, a, b, c, k, 0);
    copy_n(CB_C2, 2, 6);
    MATH((sfpu_begin(), vfilt_body<0, 5>(), sfpu_end(), ssim_pt<5, 3>()));
    copy_n(CB_C3, 3, 4);
    copy_tile(CB_ACC, 0, 3);
    MATH((loss_pt<7, 3>()));
    cb_pop_front(CB_C2, 2);
    cb_pop_front(CB_C3, 3);
    cb_pop_front(CB_ACC, 1);
    const uint32_t dd[3] = {1, 2, 7}, acc = 3;
    commit_pack(CB_D, 3, dd);
    pack_more(CB_ACC, 1, &acc);
    release();
}
#else
inline void v_pass(uint32_t a, uint32_t b, uint32_t c, uint32_t k) {
    wait3(CB_HX, a, b, c);
    wait3(CB_HXX, a, b, c);
    wait3(CB_HXY, a, b, c);
    // mu_x, E_xx
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HX + a);
    copy3(CB_HX, a, b, c, k, 0);
    copy3(CB_HXX, a, b, c, k, 4);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
    const uint32_t m[2] = {3, 7};
    commit_pack(CB_M, 2, m);
    release();
    // E_xy, SSIM terms
    cb_wait_front(CB_M, 2);
    cb_wait_front(CB_C2, 2);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HXY + a);
    copy3(CB_HXY, a, b, c, k, 0);
    copy_n(CB_M, 2, 4);
    copy_n(CB_C2, 2, 6);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), sfpu_end(), ssim_pt()));
    cb_pop_front(CB_M, 2);
    cb_pop_front(CB_C2, 2);
    const uint32_t f[4] = {0, 1, 2, 7};
    commit_pack(CB_F, 4, f);
    release();
    // d maps + loss partial
    cb_wait_front(CB_F, 4);
    cb_wait_front(CB_C3, 3);
    cb_wait_front(CB_ACC, 1);
    tile_regs_acquire();
    copy_n(CB_F, 4, 0);
    copy_n(CB_C3, 3, 4);
    copy_tile(CB_ACC, 0, 7);
    MATH((loss_pt()));
    cb_pop_front(CB_F, 4);
    cb_pop_front(CB_C3, 3);
    cb_pop_front(CB_ACC, 1);
    const uint32_t dd[3] = {1, 2, 3}, acc = 7;
    commit_pack(CB_D, 3, dd);
    pack_more(CB_ACC, 1, &acc);
    release();
}
#endif

// Runtime args: u0, u1 (work units u = (ch * ng + g) * nh + row of SSIM_K-column groups g, see
// reader_ssim.cpp), nh, nw.
void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    const uint32_t nw = get_arg_val<uint32_t>(3);
    const uint32_t ng = (nw + SSIM_K - 1) / SSIM_K;
    unary_op_init_common(CB_X, CB_D);
    {
        tile_regs_acquire();
        MATH((zero_tile<7>()));
        const uint32_t acc = 7;
        commit_pack(CB_ACC, 1, &acc);
        release();
    }
    for (uint32_t u = u0; u < u1;) {
        const uint32_t r0 = u % nh, r1 = (u1 - u < nh - r0) ? r0 + (u1 - u) : nh;
        const uint32_t j0 = (u / nh % ng) * SSIM_K, kw = nw - j0 < SSIM_K ? nw - j0 : SSIM_K;
        u += r1 - r0;
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;  // wraps for r0 = 0
            if (row < nh) {
                h_row(q % 3, kw);
            } else {
                push_zero(q % 3, kw);
            }
            if (q >= 2) {
                const uint32_t a = (q - 2) % 3;
                for (uint32_t k = 0; k < kw; k++) v_pass(a, (q - 1) % 3, q % 3, k);
                cb_pop_front(CB_HX + a, SSIM_K);
                cb_pop_front(CB_HXX + a, SSIM_K);
                cb_pop_front(CB_HXY + a, SSIM_K);
            }
        }
        q--;
        for (uint32_t s : {(q - 1) % 3, q % 3}) {
            cb_pop_front(CB_HX + s, SSIM_K);
            cb_pop_front(CB_HXX + s, SSIM_K);
            cb_pop_front(CB_HXY + s, SSIM_K);
        }
    }
    cb_wait_front(CB_ACC, 1);
    tile_regs_acquire();
    copy_n(CB_ACC, 1, 0);
    cb_pop_front(CB_ACC, 1);
    const uint32_t d0 = 0;
    commit_pack(CB_SUM, 1, &d0);
    release();
}
