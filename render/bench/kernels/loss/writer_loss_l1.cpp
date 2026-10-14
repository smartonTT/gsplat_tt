// Task #690 (T7) L1 loss step, writer (BRISC). Writes dL/dimage tiles [t0, t1) and this
// core's |x - y| partial-sum tile (slot `core`). Runtime args: g_addr, part_addr, t0, t1, core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_G = 16, CB_PART = 17;

void kernel_main() {
    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);
    constexpr auto ag = TensorAccessorArgs<0>();
    constexpr auto ap = TensorAccessorArgs<ag.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(CB_G);
    const auto acc_g = TensorAccessor(ag, g_addr, tb);
    const auto acc_p = TensorAccessor(ap, p_addr, tb);
    for (uint32_t t = t0; t < t1; t++) {
        cb_wait_front(CB_G, 1);
        noc_async_write_tile(t, acc_g, get_read_ptr(CB_G));
        noc_async_writes_flushed();
        cb_pop_front(CB_G, 1);
    }
    cb_wait_front(CB_PART, 1);
    noc_async_write_tile(core, acc_p, get_read_ptr(CB_PART));
    noc_async_write_barrier();
    cb_pop_front(CB_PART, 1);
}
