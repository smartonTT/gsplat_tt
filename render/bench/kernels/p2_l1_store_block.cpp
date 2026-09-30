// p2: 32 B record built in registers (non-volatile local) then one
// __builtin_memcpy to L1 — R3 change 1. n counts 4 B words so cycles/op is
// directly comparable with p1; one block = 8 ops.
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p2_l1_store_block");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += 8) {
            uint32_t rec[8] = {i,      i ^ 1u, i ^ 2u, i ^ 3u,
                               i ^ 4u, i ^ 5u, i ^ 6u, i ^ 7u};
            __builtin_memcpy(reinterpret_cast<void*>(a.scratch + ((i >> 3) & 15u) * 32u), rec, 32);
            // Compiler barrier only: keeps every block store emitted (no
            // dead-store elimination across the ring), no hardware fence.
            asm volatile("" : : : "memory");
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, reinterpret_cast<volatile uint32_t*>(a.scratch)[3]);
}
