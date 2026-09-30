// p0: loop-overhead baseline (mode 0) and wall-clock spin for the host-side
// tick->ns calibration (mode 1: spin until n << 8 ticks have elapsed).
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    uint32_t chk = 0;
    uint64_t t0, t1;
    if (a.mode == 0) {
        DeviceZoneScopedN("p0_loop_overhead");
        t0 = mb::now();
        // Same index/address arithmetic as p1/p2, no memory op.
        for (uint32_t i = 0; i < a.n; i += 8) {
            uint32_t q = a.scratch + ((i >> 3) & 15u) * 32u;
            asm volatile("" : : "r"(q), "r"(i));
        }
        t1 = mb::now();
    } else {
        DeviceZoneScopedN("p0_clock_spin");
        const uint64_t ticks = static_cast<uint64_t>(a.n) << 8;
        t0 = mb::now();
        do {
            t1 = mb::now();
            chk++;
        } while (t1 - t0 < ticks);
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, chk);
}
