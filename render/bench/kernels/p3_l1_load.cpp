// p3: volatile uint32 load from local L1.
//   aux A = independent loads (8 per iteration, summed) -> issue throughput
//   aux B = dependent pointer chase over a 256-word ring  -> load-to-use latency
// n loads each; total = A + B.
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    volatile uint32_t* w = reinterpret_cast<volatile uint32_t*>(a.scratch);
    for (uint32_t k = 0; k < 256; k++) w[k] = a.scratch + ((k * 97u + 1u) & 255u) * 4u;
    uint32_t s = 0;
    uint64_t t0, t1, t2;
    {
        DeviceZoneScopedN("p3_l1_load_tput");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += 8) {
            volatile uint32_t* p = w + ((i >> 3) & 31u) * 8u;
            uint32_t v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];
            uint32_t v4 = p[4], v5 = p[5], v6 = p[6], v7 = p[7];
            s += (v0 ^ v1) + (v2 ^ v3) + (v4 ^ v5) + (v6 ^ v7);
        }
        t1 = mb::now();
    }
    uint32_t q = a.scratch;
    {
        DeviceZoneScopedN("p3_l1_load_chase");
        for (uint32_t i = 0; i < a.n; i++) q = *reinterpret_cast<volatile uint32_t*>(q);
        t2 = mb::now();
    }
    mb::publish(a, a.n, t2 - t0, t1 - t0, t2 - t1, s ^ q);
}
