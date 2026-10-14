// Task #736: reader for ssim_fwd_compute.cpp / ssim_bwd_compute.cpp (SSIM_BWD defined).
// Strip walk as reader_filt.cpp: for each tile row I = r0-1 .. r1 that exists, the tiles
// [I, J-1..J+1] (zero tiles off the plane) of the filter inputs; after row I, the pointwise
// inputs of output row I-1 when it is in the segment.
//   fwd: CB 0 = x[ch], CB 1 = y[ch] (3 each); CB 2 = mu_y[ch], vy[ch]; CB 3 = wm, x[ch], y[ch].
//        a = x, b = y, c = consts (7 planes: mu_y 0..2, vy 3..5, wm 6).
//   bwd: CB 0 = d_mu[ch], d_xx[ch], d_xy[ch] (9; d maps: planes m * 3 + ch); CB 1 = x[ch], y[ch].
//        a = d maps, b = x, c = y.
// Runtime args: a, b, c, t0, t1, nh, nw.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// #743 timing ablation: SSIM_DMABL bit 0 skips the DRAM reads (CB pages hold stale data).
#if defined(SSIM_DMABL) && (SSIM_DMABL & 1)
#define noc_async_read_tile(...) ((void)0)
#endif

template <typename A>
inline void read3(const A& acc, uint32_t plane, uint32_t i, uint32_t j, uint32_t nh, uint32_t nw, uint32_t w,
                  uint32_t tb) {
    for (uint32_t k = 0; k < 3; k++) {
        const uint32_t jj = j + k - 1;
        if (jj < nw) {
            noc_async_read_tile((plane * nh + i) * nw + jj, acc, w + k * tb);
        } else {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + k * tb);
            for (uint32_t e = 0; e < tb / 4; e++) p[e] = 0;
        }
    }
}

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t c_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t t1 = get_arg_val<uint32_t>(4);
    const uint32_t nh = get_arg_val<uint32_t>(5);
    const uint32_t nw = get_arg_val<uint32_t>(6);
    constexpr auto aa = TensorAccessorArgs<0>();
    constexpr auto ab = TensorAccessorArgs<aa.next_compile_time_args_offset()>();
    constexpr auto ac = TensorAccessorArgs<ab.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(0);
    const auto A = TensorAccessor(aa, a_addr, tb);
    const auto B = TensorAccessor(ab, b_addr, tb);
    const auto C = TensorAccessor(ac, c_addr, tb);
    for (uint32_t t = t0; t < t1;) {
        const uint32_t s = t / nh, r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        const uint32_t ch = s / nw, j = s % nw;
        for (uint32_t i = (r0 ? r0 - 1 : 0); i < nh + 1 && i <= r1; i++) {
            if (i < nh) {
#ifdef SSIM_BWD
                cb_reserve_back(0, 9);
                const uint32_t w = get_write_ptr(0);
                for (uint32_t m = 0; m < 3; m++) read3(A, m * 3 + ch, i, j, nh, nw, w + 3 * m * tb, tb);
                noc_async_read_barrier();
                cb_push_back(0, 9);
#else
                cb_reserve_back(0, 3);
                cb_reserve_back(1, 3);
                read3(A, ch, i, j, nh, nw, get_write_ptr(0), tb);
                read3(B, ch, i, j, nh, nw, get_write_ptr(1), tb);
                noc_async_read_barrier();
                cb_push_back(0, 3);
                cb_push_back(1, 3);
#endif
            }
            if (i >= 1 && i - 1 >= r0) {
                const uint32_t o = i - 1, tt = (ch * nh + o) * nw + j;
#ifdef SSIM_BWD
                cb_reserve_back(1, 2);
                noc_async_read_tile(tt, B, get_write_ptr(1));
                noc_async_read_tile(tt, C, get_write_ptr(1) + tb);
                noc_async_read_barrier();
                cb_push_back(1, 2);
#else
                cb_reserve_back(2, 2);
                cb_reserve_back(3, 3);
                noc_async_read_tile(tt, C, get_write_ptr(2));
                noc_async_read_tile(((3 + ch) * nh + o) * nw + j, C, get_write_ptr(2) + tb);
                noc_async_read_tile((6 * nh + o) * nw + j, C, get_write_ptr(3));
                noc_async_read_tile(tt, A, get_write_ptr(3) + tb);
                noc_async_read_tile(tt, B, get_write_ptr(3) + 2 * tb);
                noc_async_read_barrier();
                cb_push_back(2, 2);
                cb_push_back(3, 3);
#endif
            }
        }
    }
}
