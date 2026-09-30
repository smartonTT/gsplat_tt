// p1: one volatile uint32 store to local L1. Eight per iteration into a 32 B
// slot of a 16-slot ring — the exact shape of sort_bin.cpp pack_rec today.
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p1_l1_store_volatile");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += 8) {
            volatile uint32_t* p =
                reinterpret_cast<volatile uint32_t*>(a.scratch + ((i >> 3) & 15u) * 32u);
            p[0] = i;
            p[1] = i ^ 1u;
            p[2] = i ^ 2u;
            p[3] = i ^ 3u;
            p[4] = i ^ 4u;
            p[5] = i ^ 5u;
            p[6] = i ^ 6u;
            p[7] = i ^ 7u;
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, reinterpret_cast<volatile uint32_t*>(a.scratch)[3]);
}
