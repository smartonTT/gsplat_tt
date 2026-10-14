// Task #690 (T7) L1 loss step, reader (NCRISC). Streams this core's x (render) and y (GT)
// fp32 tiles [t0, t1) into CB_X / CB_Y, and seeds the |x - y| accumulator CB with a zero
// tile. Runtime args: x_addr, y_addr, t0, t1.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_X = 0, CB_Y = 1;

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    constexpr auto ax = TensorAccessorArgs<0>();
    constexpr auto ay = TensorAccessorArgs<ax.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(CB_X);
    const auto acc_x = TensorAccessor(ax, x_addr, tb);
    const auto acc_y = TensorAccessor(ay, y_addr, tb);

    for (uint32_t t = t0; t < t1; t++) {
        cb_reserve_back(CB_X, 1);
        cb_reserve_back(CB_Y, 1);
        noc_async_read_tile(t, acc_x, get_write_ptr(CB_X));
        noc_async_read_tile(t, acc_y, get_write_ptr(CB_Y));
        noc_async_read_barrier();
        cb_push_back(CB_X, 1);
        cb_push_back(CB_Y, 1);
    }
}
