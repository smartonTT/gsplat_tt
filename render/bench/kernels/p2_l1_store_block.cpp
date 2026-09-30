// p2: 32 B record built in registers (non-volatile local), stored as a block —
// R3 change 1. n counts 4 B words so ticks/op is directly comparable with p1;
// one block = 8 ops.
//   MB_P2=0: __builtin_memcpy(void*, rec, 32)  (as written in the plan; the
//            destination alignment is unknown, so GCC emits `jal memcpy`)
//   MB_P2=1: non-volatile uint32_t* stores (8 x sw, no call)
//   MB_P2=2: __builtin_memcpy with __builtin_assume_aligned(dst, 16)
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
            const uint32_t dst = a.scratch + ((i >> 3) & 15u) * 32u;
#if MB_P2 == 0
            __builtin_memcpy(reinterpret_cast<void*>(dst), rec, 32);
#elif MB_P2 == 1
            uint32_t* d = reinterpret_cast<uint32_t*>(dst);
            for (uint32_t k = 0; k < 8; k++) d[k] = rec[k];
#else
            __builtin_memcpy(__builtin_assume_aligned(reinterpret_cast<void*>(dst), 16), rec, 32);
#endif
            // Compiler barrier only: keeps every block store emitted (no
            // dead-store elimination across the ring), no hardware fence.
            asm volatile("" : : : "memory");
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, reinterpret_cast<volatile uint32_t*>(a.scratch)[3]);
}
