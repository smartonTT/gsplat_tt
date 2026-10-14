// #784 T2 heavy-row split, phase A writer (RISCV_0). Per heavy group (index j from the reader's
// header): the packed sums tile (stream s of lane i at word 64 s + i) is turned into one 64 B
// record per lane (16 words, stream order, like a T1 slot) and written as records
// h = j * T2_HL .. j * T2_HL + T2_HL - 1 of the hsum buffer (4 KB pages of 64 records), which
// phase B (t2g_reader T2_SPLIT) reads as a heavy lane's plane 0.
// Runtime args: hsum DRAM address, group count. Compile args: TensorAccessorArgs(hsum).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_WH = 5, CB_FO = 19, CB_HR = 7, PAGE = 4096, RECS = 64;
}  // namespace

void kernel_main() {
    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    constexpr auto ha = TensorAccessorArgs<0>();
    const auto HS = TensorAccessor(ha, h_addr, PAGE);
    const uint32_t rec_l1 = get_write_ptr(CB_HR);
    volatile uint32_t* rec = reinterpret_cast<volatile uint32_t*>(rec_l1);
    for (uint32_t q = 0; q < n; q++) {
        cb_wait_front(CB_WH, 1);
        const uint32_t j = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_WH))[0];
        cb_pop_front(CB_WH, 1);
        cb_wait_front(CB_FO, 1);
        const volatile uint32_t* fo = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_FO));
        noc_async_writes_flushed();  // the previous group's records have left rec
        for (uint32_t i = 0; i < T2_HL; i++)
            for (uint32_t s = 0; s < 16; s++) rec[16 * i + s] = fo[64 * s + i];
        cb_pop_front(CB_FO, 1);
        const uint32_t h0 = j * T2_HL;
        noc_async_write(rec_l1, get_noc_addr(h0 / RECS, HS, (h0 % RECS) * 64), T2_HL * 64);
    }
    noc_async_write_barrier();
}
