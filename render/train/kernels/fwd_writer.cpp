// T0 blend forward writer (RISCV_0, #716). Per tile: drops the 3 bf16 colour tiles of
// CB_COLOR_OUT and writes the 4 fp32 R, G, B, T tiles of CB_TRAIN_OUT (task #708) to DRAM
// page t (16 KB) of the train buffer.
// Runtime args: train buffer DRAM address, first tile, tile count.
// Compile args: TensorAccessorArgs(train).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_COLOR_OUT = 16, CB_TRAIN_OUT = 17;
constexpr uint32_t TRAIN_PAGE = 4 * 4096;
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, TRAIN_PAGE);
    for (uint32_t t = t0; t < t0 + nt; t++) {
        cb_wait_front(CB_COLOR_OUT, 3);
        cb_pop_front(CB_COLOR_OUT, 3);
        cb_wait_front(CB_TRAIN_OUT, 4);
        noc_async_write(get_read_ptr(CB_TRAIN_OUT), get_noc_addr(t, out), TRAIN_PAGE);
        noc_async_write_barrier();
        cb_pop_front(CB_TRAIN_OUT, 4);
    }
}
