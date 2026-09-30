// Shared helpers for the T-A hardware-ceiling microbenchmark kernels.
//
// RUNTIME ARGS (all uint32, identical layout for every probe):
//   0: results_addr  (DRAM, 64 B page per (core, risc) slot)
//   1: slot          (= core_linear * 2 + risc; risc 0 = BRISC, 1 = NCRISC)
//   2: n             (operations timed by this RISC)
//   3: scratch_off   (byte offset into CB_SCRATCH; movers sharing a core use
//                     disjoint halves)
//   4: data_addr     (DRAM target buffer for NoC probes)
//   5: data_npages   (power of two)
//   6: salt          (core_linear; spreads cores over DRAM pages)
//   7: mode          (probe-specific)
// COMPILE-TIME ARGS: TensorAccessorArgs(results), TensorAccessorArgs(data).
//
// Timing is the RISC-V debug wall clock (get_timestamp); the host converts
// ticks -> ns with a clock measured in the same run (p0 spin calibration).
// Every timed loop also sits in one DeviceZoneScopedN so a profiler build shows
// it in Tracy.
#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace mb {

constexpr uint32_t CB_SCRATCH = 0;
constexpr uint32_t CB_RESULT = 1;
constexpr uint32_t RES_BYTES = 64;
constexpr uint32_t MAGIC = 0x4D42A5A5u;

struct Args {
    uint32_t results_addr, slot, n, scratch, data_addr, data_npages, salt, mode, res_l1;
};

FORCE_INLINE Args args() {
    Args a;
    a.results_addr = get_arg_val<uint32_t>(0);
    a.slot = get_arg_val<uint32_t>(1);
    a.n = get_arg_val<uint32_t>(2);
    a.scratch = get_write_ptr(CB_SCRATCH) + get_arg_val<uint32_t>(3);
    a.data_addr = get_arg_val<uint32_t>(4);
    a.data_npages = get_arg_val<uint32_t>(5);
    a.salt = get_arg_val<uint32_t>(6);
    a.mode = get_arg_val<uint32_t>(7);
    a.res_l1 = get_write_ptr(CB_RESULT) + (a.slot & 1u) * RES_BYTES;
    return a;
}

FORCE_INLINE uint64_t now() { return get_timestamp(); }

// Result record: [0] magic [1] ops [2,3] total ticks [4,5] aux A [6,7] aux B
// [8] checksum (keeps the timed loop observable) [9] noc x | y << 8.
FORCE_INLINE void publish(const Args& a, uint32_t ops, uint64_t total, uint64_t aux_a,
                          uint64_t aux_b, uint32_t chk) {
    volatile uint32_t* r = reinterpret_cast<volatile uint32_t*>(a.res_l1);
    r[0] = MAGIC;
    r[1] = ops;
    r[2] = static_cast<uint32_t>(total);
    r[3] = static_cast<uint32_t>(total >> 32);
    r[4] = static_cast<uint32_t>(aux_a);
    r[5] = static_cast<uint32_t>(aux_a >> 32);
    r[6] = static_cast<uint32_t>(aux_b);
    r[7] = static_cast<uint32_t>(aux_b >> 32);
    r[8] = chk;
    r[9] = my_x[noc_index] | (my_y[noc_index] << 8);
    constexpr auto res_args = TensorAccessorArgs<0>();
    const auto acc = TensorAccessor(res_args, a.results_addr, RES_BYTES);
    noc_async_write(a.res_l1, get_noc_addr(a.slot, acc), RES_BYTES);
    noc_async_write_barrier();
}

}  // namespace mb

#define MB_DATA_ACC(a)                                                            \
    constexpr auto mb_res_args = TensorAccessorArgs<0>();                         \
    constexpr auto mb_data_args =                                                 \
        TensorAccessorArgs<mb_res_args.next_compile_time_args_offset()>();        \
    const auto data_acc = TensorAccessor(mb_data_args, (a).data_addr, MB_DATA_PAGE)
