// Task #736 (T7): fused L1 + D-SSIM, backward half. dL/dx = G*d_mu + 2x G*d_xx + y G*d_xy
// + wl1 sign(x - y) (G symmetric), the d maps from ssim_fwd_compute.cpp (9 DRAM planes: the
// filter needs their halo from other cores). Same strip walk and SFPU filter (ssim_sfpu.h).
#include "ssim_sfpu.h"

constexpr uint32_t CB_DIN = 0, CB_XY = 1, CB_HP = 6, CB_HQ = 9, CB_HR = 12, CB_G = 16, CB_PQ = 17;

#ifdef TRISC_MATH
// 3 = G*d_xy, 4 = P, 5 = Q, 6 = x, 7 = y -> 3 = grad.
inline void grad_pt() {
    using namespace sfpi;
    if constexpr (SSIM_ABL & 4) return;
    sfpu_begin();
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        const vFloat x = dst_reg[192], y = dst_reg[224];
        vFloat g = (vFloat(dst_reg[128]) + (x + x) * dst_reg[160]) + y * dst_reg[96];
        const vFloat d = x - y;
        v_if(d > 0.0f) { g = g + SS_WL1; }
        v_elseif(d < 0.0f) { g = g - SS_WL1; }
        v_endif;
        dst_reg[96] = g;
        dst_reg++;
    }
    sfpu_end();
}
#endif

// H pass of one tile row of a kw-column strip: column k filters pages k .. k+2 of each d-map
// plane row (SSIM_K + 2 pages per plane in CB_DIN) into page k of the slot q CBs.
inline void h_row(uint32_t q, uint32_t kw) {
    constexpr uint32_t RP = SSIM_K + 2;
    cb_wait_front(CB_DIN, 3 * RP);
    cb_reserve_back(CB_HP + q, SSIM_K);
    cb_reserve_back(CB_HQ + q, SSIM_K);
    cb_reserve_back(CB_HR + q, SSIM_K);
    for (uint32_t k = 0; k < kw; k++) {
        tile_regs_acquire();
        copy_at(CB_DIN, k, 3, 0);
        copy_tile(CB_DIN, RP + k, 4);
        copy_tile(CB_DIN, RP + k + 1, 5);
        copy_tile(CB_DIN, RP + k + 2, 6);
        transpose_tiles(0x77);
        MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
        transpose_tiles(0x88);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(3, CB_HP + q, k);
        pack_tile<true>(7, CB_HQ + q, k);
        release();
        tile_regs_acquire();
        copy_at(CB_DIN, 2 * RP + k, 3, 0);
        transpose_tiles(0x07);
        MATH((sfpu_begin(), vfilt_body<0, 3>(), sfpu_end()));
        transpose_tiles(0x08);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(3, CB_HR + q, k);
        release();
    }
    cb_pop_front(CB_DIN, 3 * RP);
    cb_push_back(CB_HP + q, SSIM_K);
    cb_push_back(CB_HQ + q, SSIM_K);
    cb_push_back(CB_HR + q, SSIM_K);
}

inline void push_zero(uint32_t q, uint32_t kw) {
    tile_regs_acquire();
    MATH((zero_tile<3>()));
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t cb : {CB_HP, CB_HQ, CB_HR}) {
        cb_reserve_back(cb + q, SSIM_K);
        for (uint32_t k = 0; k < kw; k++) pack_tile<true>(3, cb + q, k);
        cb_push_back(cb + q, SSIM_K);
    }
    release();
}

// V pass of output column c0 (page c0 of the slots a, b, c = rows I-1, I, I+1).
inline void v_pass(uint32_t a, uint32_t b, uint32_t c, uint32_t c0) {
    const uint32_t sl[3] = {a, b, c};
    for (uint32_t k = 0; k < 3; k++) {
        cb_wait_front(CB_HP + sl[k], SSIM_K);
        cb_wait_front(CB_HQ + sl[k], SSIM_K);
        cb_wait_front(CB_HR + sl[k], SSIM_K);
    }
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HP + a);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HP + sl[k], c0, k);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HQ + sl[k], c0, 4 + k);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
    const uint32_t pq[2] = {3, 7};
    commit_pack(CB_PQ, 2, pq);
    release();
    cb_wait_front(CB_PQ, 2);
    cb_wait_front(CB_XY, 2);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HR + a);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HR + sl[k], c0, k);
    copy_n(CB_PQ, 2, 4);
    copy_n(CB_XY, 2, 6);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), sfpu_end(), grad_pt()));
    cb_pop_front(CB_PQ, 2);
    cb_pop_front(CB_XY, 2);
    const uint32_t d3 = 3;
    commit_pack(CB_G, 1, &d3);
    release();
}

// Runtime args: u0, u1, nh, nw (as ssim_fwd_compute.cpp).
void kernel_main() {
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t u1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    const uint32_t nw = get_arg_val<uint32_t>(3);
    const uint32_t ng = (nw + SSIM_K - 1) / SSIM_K;
    unary_op_init_common(CB_DIN, CB_G);
    for (uint32_t u = u0; u < u1;) {
        const uint32_t r0 = u % nh, r1 = (u1 - u < nh - r0) ? r0 + (u1 - u) : nh;
        const uint32_t j0 = (u / nh % ng) * SSIM_K, kw = nw - j0 < SSIM_K ? nw - j0 : SSIM_K;
        u += r1 - r0;
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;
            if (row < nh) {
                h_row(q % 3, kw);
            } else {
                push_zero(q % 3, kw);
            }
            if (q >= 2) {
                const uint32_t a = (q - 2) % 3;
                for (uint32_t k = 0; k < kw; k++) v_pass(a, (q - 1) % 3, q % 3, k);
                cb_pop_front(CB_HP + a, SSIM_K);
                cb_pop_front(CB_HQ + a, SSIM_K);
                cb_pop_front(CB_HR + a, SSIM_K);
            }
        }
        q--;
        for (uint32_t s : {(q - 1) % 3, q % 3}) {
            cb_pop_front(CB_HP + s, SSIM_K);
            cb_pop_front(CB_HQ + s, SSIM_K);
            cb_pop_front(CB_HR + s, SSIM_K);
        }
    }
}
