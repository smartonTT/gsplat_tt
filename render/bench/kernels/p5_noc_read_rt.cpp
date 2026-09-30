// p5: exposed noc_async_read latency, 64 B DRAM -> L1 (a blendrec page read).
//   MB_DEPTH=1: read + immediate noc_async_read_barrier (sort_bin today)
//   MB_DEPTH=8: 8 reads outstanding, one barrier (the iter-137 pattern)
// get_noc_addr(page, accessor) is inside the loop, as in production.
#include "mb_common.h"

#define MB_DATA_PAGE 1024u

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    const uint32_t pmask = a.data_npages - 1u;
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p5_noc_read_rt");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += MB_DEPTH) {
            for (uint32_t d = 0; d < MB_DEPTH; d++) {
                const uint32_t j = i + d;
                noc_async_read(get_noc_addr((a.salt * 131u + j * 5u) & pmask, data_acc) + (j & 15u) * 64u,
                               a.scratch + d * 64u, 64u);
            }
            noc_async_read_barrier();
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, reinterpret_cast<volatile uint32_t*>(a.scratch)[0]);
}
