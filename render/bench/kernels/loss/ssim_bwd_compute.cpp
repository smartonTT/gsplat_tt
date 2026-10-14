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

inline void h_pass(uint32_t q) {
    cb_wait_front(CB_DIN, 6);
    tile_regs_acquire();
    copy_n(CB_DIN, 3, 0);
    copy_tile(CB_DIN, 3, 4);
    copy_tile(CB_DIN, 4, 5);
    copy_tile(CB_DIN, 5, 6);
    transpose_tiles(0x77);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
    transpose_tiles(0x88);
    cb_pop_front(CB_DIN, 6);
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t d3 = 3, d7 = 7;
    pack_more(CB_HP + q, 1, &d3);
    pack_more(CB_HQ + q, 1, &d7);
    release();
    cb_wait_front(CB_DIN, 3);
    tile_regs_acquire();
    copy_n(CB_DIN, 3, 0);
    transpose_tiles(0x07);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), sfpu_end()));
    transpose_tiles(0x08);
    cb_pop_front(CB_DIN, 3);
    tile_regs_commit();
    tile_regs_wait();
    pack_more(CB_HR + q, 1, &d3);
    release();
}

inline void push_zero(uint32_t q) {
    tile_regs_acquire();
    MATH((zero_tile<3>()));
    tile_regs_commit();
    tile_regs_wait();
    const uint32_t d3 = 3;
    pack_more(CB_HP + q, 1, &d3);
    pack_more(CB_HQ + q, 1, &d3);
    pack_more(CB_HR + q, 1, &d3);
    release();
}

inline void v_pass(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t sl[3] = {a, b, c};
    for (uint32_t k = 0; k < 3; k++) {
        cb_wait_front(CB_HP + sl[k], 1);
        cb_wait_front(CB_HQ + sl[k], 1);
        cb_wait_front(CB_HR + sl[k], 1);
    }
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HP + a);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HP + sl[k], 0, k);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HQ + sl[k], 0, 4 + k);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), vfilt_body<4, 7>(), sfpu_end()));
    const uint32_t pq[2] = {3, 7};
    commit_pack(CB_PQ, 2, pq);
    release();
    cb_wait_front(CB_PQ, 2);
    cb_wait_front(CB_XY, 2);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(CB_HR + a);
    for (uint32_t k = 0; k < 3; k++) copy_tile(CB_HR + sl[k], 0, k);
    copy_n(CB_PQ, 2, 4);
    copy_n(CB_XY, 2, 6);
    MATH((sfpu_begin(), vfilt_body<0, 3>(), sfpu_end(), grad_pt()));
    cb_pop_front(CB_PQ, 2);
    cb_pop_front(CB_XY, 2);
    const uint32_t d3 = 3;
    commit_pack(CB_G, 1, &d3);
    release();
    cb_pop_front(CB_HP + a, 1);
    cb_pop_front(CB_HQ + a, 1);
    cb_pop_front(CB_HR + a, 1);
}

// Runtime args: t0, t1, nh (as ssim_fwd_compute.cpp).
void kernel_main() {
    const uint32_t t0 = get_arg_val<uint32_t>(0);
    const uint32_t t1 = get_arg_val<uint32_t>(1);
    const uint32_t nh = get_arg_val<uint32_t>(2);
    unary_op_init_common(CB_DIN, CB_G);
    for (uint32_t t = t0; t < t1;) {
        const uint32_t r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        uint32_t q = 0;
        for (uint32_t i = r0; i <= r1 + 1; i++, q++) {
            const uint32_t row = i - 1;
            if (row < nh) {
                h_pass(q % 3);
            } else {
                push_zero(q % 3);
            }
            if (q >= 2) v_pass((q - 2) % 3, (q - 1) % 3, q % 3);
        }
        q--;
        for (uint32_t s : {(q - 1) % 3, q % 3}) {
            cb_pop_front(CB_HP + s, 1);
            cb_pop_front(CB_HQ + s, 1);
            cb_pop_front(CB_HR + s, 1);
        }
    }
}
