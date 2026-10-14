// T1 blend backward writer (RISCV_0, #693): one packed fp32 ACC tile per pair (vectors 0-8
// hold the pair's per-lane term sums) from CB_OUT to DRAM page <global record index>.
// The host sums the lanes and applies the finish (blend_bwd_bench.cpp, v1: lane reduce on host).
// Runtime args: out DRAM address, first record, record count. Compile args: TensorAccessorArgs(out).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_OUT = 16, TILE_B = 4096;
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t r0 = get_arg_val<uint32_t>(1);
    const uint32_t nr = get_arg_val<uint32_t>(2);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, TILE_B);
    for (uint32_t r = r0; r < r0 + nr; r++) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write(get_read_ptr(CB_OUT), get_noc_addr(r, out), TILE_B);
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    }
    noc_async_write_barrier();
}
