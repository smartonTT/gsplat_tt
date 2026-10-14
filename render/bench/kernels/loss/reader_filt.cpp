// Task #706: reader for filt_compute.cpp. Sends the 12 constant band tiles once, then for
// each strip segment of output tiles [t0, t1) (t = strip * nh + row; strip s: channel s / nw,
// tile column s % nw) the tiles X[ch, I, J-1..J+1] (zero tiles off the plane) of the rows
// I = r0-1 .. r1 that exist (the compute kernel zero-fills the others).
// Runtime args: x_addr, r_addr (0: none), t0, t1, nh, nw. Any number of planes (ch = s / nw).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0, CB_R = 2;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    const uint32_t nh = get_arg_val<uint32_t>(4);
    const uint32_t nw = get_arg_val<uint32_t>(5);
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ar = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(CB_X);
    const auto acc_x = TensorAccessor(ax, x_addr, tb);
    const auto acc_r = TensorAccessor(ar, r_addr, tb);

    if (r_addr) {  // 0: no band constants (sfilt_compute.cpp, #720)
        cb_reserve_back(CB_R, 12);
        for (uint32_t t = 0; t < 12; t++) noc_async_read_tile(t, acc_r, get_write_ptr(CB_R) + t * tb);
        noc_async_read_barrier();
        cb_push_back(CB_R, 12);
    }

    for (uint32_t t = t0; t < t1;) {
        const uint32_t s = t / nh, r0 = t % nh, r1 = (t1 - t < nh - r0) ? r0 + (t1 - t) : nh;
        t += r1 - r0;
        const uint32_t ch = s / nw, j = s % nw;
        for (uint32_t i = (r0 ? r0 - 1 : 0); i < nh && i <= r1; i++) {
            cb_reserve_back(CB_X, 3);
            const uint32_t w = get_write_ptr(CB_X);
            for (uint32_t k = 0; k < 3; k++) {
                const uint32_t jj = j + k - 1;  // wraps to a huge value for j = 0, k = 0
                if (jj < nw) {
                    noc_async_read_tile((ch * nh + i) * nw + jj, acc_x, w + k * tb);
                } else {
                    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + k * tb);
                    for (uint32_t e = 0; e < tb / 4; e++) p[e] = 0;
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_X, 3);
        }
    }
}
