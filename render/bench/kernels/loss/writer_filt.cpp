// Task #706: writer for filt_compute.cpp: output tile (ch, I, J) per strip and tile row.
// Runtime args: o_addr, s0, s1, nh, nw.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t s0 = get_arg_val<uint32_t>(1);
    const uint32_t s1 = get_arg_val<uint32_t>(2);
    const uint32_t nh = get_arg_val<uint32_t>(3);
    const uint32_t nw = get_arg_val<uint32_t>(4);
    constexpr auto ao = TensorAccessorArgs<0>();
    const uint32_t tb = get_tile_size(CB_OUT);
    const auto acc_o = TensorAccessor(ao, o_addr, tb);
    for (uint32_t s = s0; s < s1; s++) {
        const uint32_t ch = s / nw, j = s % nw;
        for (uint32_t i = 0; i < nh; i++) {
            cb_wait_front(CB_OUT, 1);
            noc_async_write_tile((ch * nh + i) * nw + j, acc_o, get_read_ptr(CB_OUT));
            noc_async_writes_flushed();
            cb_pop_front(CB_OUT, 1);
        }
    }
    noc_async_write_barrier();
}
