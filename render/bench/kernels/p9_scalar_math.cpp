// p9: per-op cost of scalar math that the emit/pack loops do per pair, as the
// data-mover toolchain compiles it (soft-float / soft-divide when the RISC has
// no hardware unit — the disassembly shows which).
//   MB_OP=0: fp32 sub of an int->float conversion  (pack_rec tile-local mean)
//   MB_OP=1: fp32 mul
//   MB_OP=2: uint32 divide by a runtime divisor      (tt / l1_tiles_x off the
//            pow2 path)
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    uint32_t acc = a.salt | 1u;
    const uint32_t div = a.data_npages | 3u;  // runtime, non-pow2
    float f = static_cast<float>(a.salt) * 0.37f + 1.5f;
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p9_scalar_math");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i++) {
#if MB_OP == 0
            float r = f - static_cast<float>(i * 16u);
            uint32_t bits;
            __builtin_memcpy(&bits, &r, 4);
            acc += bits;
#elif MB_OP == 1
            f = f * 1.0000001f;
#else
            acc += (i ^ acc) / div;
#endif
            asm volatile("" : "+r"(acc));
        }
        t1 = mb::now();
    }
    uint32_t fb;
    __builtin_memcpy(&fb, &f, 4);
    mb::publish(a, a.n, t1 - t0, 0, 0, acc ^ fb);
}
