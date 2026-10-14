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

// 3 = E_xy, 4 = mu_x, 5 = E_xx, 6 = mu_y, 7 = vy -> 0 = S, 1 = F_mu, 2 = F_xx, 7 = F_xy with
// d_mu = Sw F_mu, d_xx = Sw F_xx, d_xy = Sw F_xy (Sw = S wm, next block).
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
            dst_reg[32] = (vFloat(dst_reg[96]) - mxy) * 2.0f + SS_C2;            // B
            dst_reg[64] = mux * mux + muy * muy + SS_C1;                         // C
            dst_reg[160] = (vFloat(dst_reg[160]) - mux * mux) + dst_reg[224] + SS_C2;  // D
        }
        const vFloat rB = rcp(dst_reg[32]);
        dst_reg[224] = rB * 2.0f;  // F_xy
        {
            const vFloat rA = rcp(dst_reg[0]);
            dst_reg[96] = vFloat(dst_reg[192]) * (rA - rB);
        }
        const vFloat rC = rcp(dst_reg[64]), rD = rcp(dst_reg[160]);
        dst_reg[0] = (vFloat(dst_reg[0]) * dst_reg[32]) * (rC * rD);           // S
        dst_reg[32] = (vFloat(dst_reg[96]) + vFloat(dst_reg[128]) * (rD - rC)) * 2.0f;  // F_mu
        dst_reg[64] = -rD;                                                     // F_xx
        dst_reg++;
    }
    sfpu_end();
}

// 0 = S, 1..3 = F_mu, F_xx, F_xy, 4 = wm, 5 = x, 6 = y, 7 = acc -> 1..3 = d maps, 7 += loss.
inline void loss_pt() {
    using namespace sfpi;
    if constexpr (SSIM_ABL & 4) return;
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        const vFloat sw = vFloat(dst_reg[0]) * dst_reg[128];
        dst_reg[32] = sw * dst_reg[32];
        dst_reg[64] = sw * dst_reg[64];
        dst_reg[96] = sw * dst_reg[96];
        const vFloat d = vFloat(dst_reg[160]) - dst_reg[192];
        dst_reg[224] = vFloat(dst_reg[224]) + (sw + sfpi::abs(d) * SS_WL1);
        dst_reg++;
    }
    sfpu_end();
}
#endif

inline void wait3(uint32_t cb0, uint32_t a, uint32_t b, uint32_t c) {
    cb_wait_front(cb0 + a, 1);
    cb_wait_front(cb0 + b, 1);
    cb_wait_front(cb0 + c, 1);
}
inline void copy3(uint32_t cb0, uint32_t a, uint32_t b, uint32_t c, uint32_t d0) {
    copy_tile(cb0 + a, 0, d0);
    copy_tile(cb0 + b, 0, d0 + 1);
    copy_tile(cb0 + c, 0, d0 + 2);
}

inline void h_pass(uint32_t q) {
    cb_wait_front(CB_X, 3);
    cb_wait_front(CB_Y, 3);
    tile_regs_acquire();
    copy_n(CB_X, 3, 0);
    copy_n(CB_Y, 3, 4);
    transpose_tiles(0x77);
    MATH((h_sfpu()));
    transpose_tiles(0x98);
    cb_pop_front(CB_X, 3);
    cb_pop_front(CB_Y, 3);
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t d4 = 4, d3 = 3, d7 = 7;
    pack_more(CB_HX + q, 1, &d4);
    pack_more(CB_HXX + q, 1, &d3);
    pack_more(CB_HXY + q, 1, &d7);
    release();
}

inline void push_zero(uint32_t q) {
    tile_regs_acquire();
    MATH((zero_tile<3>()));
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t d3 = 3;
    pack_more(CB_HX + q, 1, &d3);
    pack_more(CB_HXX + q, 1, &d3);
    pack_more(CB_HXY + q, 1, &d3);
    release();
}

inline void v_pass(uint32_t a, uint32_t b, uint32_t c) {
    wait3(CB_HX, a, b, c);
    wait3(CB_HXX, a, b, c);
    wait3(CB_HXY, a, b, c);
    // mu_x, E_xx
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HX + a);
    copy3(CB_HX, a, b, c, 0);
    copy3(CB_HXX, a, b, c, 4);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
    const uint32_t m[2] = {3, 7};
    commit_pack(CB_M, 2, m);
    release();
    // E_xy, SSIM terms
    cb_wait_front(CB_M, 2);
    cb_wait_front(CB_C2, 2);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HXY + a);
    copy3(CB_HXY, a, b, c, 0);
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
    cb_pop_front(CB_HX + a, 1);
    cb_pop_front(CB_HXX + a, 1);
    cb_pop_front(CB_HXY + a, 1);
}

// Runtime args: t0, t1 (output tiles, t = strip * nh + row, strip = ch * nw + J), nh.
void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    unary_op_init_common(CB_X, CB_D);
    {
        tile_regs_acquire();
        MATH((zero_tile<7>()));
        const uint32_t acc = 7;
        commit_pack(CB_ACC, 1, &acc);
        release();
    }
    for (uint32_t t = t0; t < t1;) {
        const uint32_t r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;  // wraps for r0 = 0
            if (row < nh) {
                h_pass(q % 3);
            } else {
                push_zero(q % 3);
            }
            if (q >= 2) v_pass((q - 2) % 3, (q - 1) % 3, q % 3);
        }
        q--;
        for (uint32_t s : {(q - 1) % 3, q % 3}) {
            cb_pop_front(CB_HX + s, 1);
            cb_pop_front(CB_HXX + s, 1);
            cb_pop_front(CB_HXY + s, 1);
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
