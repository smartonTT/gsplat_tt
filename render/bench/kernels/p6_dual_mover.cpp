// p6: the sort_bucket_emit store-and-scatter loop — pack a 32 B record with
// eight volatile stores into a 16-slot L1 staging ring, scatter each full batch
// of 16 as 32 B noc_async_writes (get_noc_addr per write), one write barrier per
// batch. The host runs it on NCRISC only, BRISC only, and split N/2 + N/2 over
// both movers (disjoint scratch halves, each on its default NoC).
//   MB_READ=1: additionally one 64 B DRAM read + immediate barrier per record
//   (the 1-deep blendrec read), so the loop is latency-bound rather than
//   NoC-write-throughput-bound — closer to production intensity.
#include "mb_common.h"

#define MB_DATA_PAGE 1024u

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    const uint32_t pmask = a.data_npages - 1u;
    const uint32_t salt = a.salt * 2u + (a.slot & 1u);
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p6_dual_mover");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += 16) {
            for (uint32_t b = 0; b < 16; b++) {
                const uint32_t j = i + b;
#if MB_READ
                noc_async_read(get_noc_addr((salt * 61u + j * 13u) & pmask, data_acc) + (j & 15u) * 64u,
                               a.scratch + 512u, 64u);
                noc_async_read_barrier();
#endif
                volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(a.scratch + b * 32u);
                p[0] = j;
                p[1] = j ^ 1u;
                p[2] = j ^ 2u;
                p[3] = j ^ 3u;
                p[4] = j ^ 4u;
                p[5] = j ^ 5u;
                p[6] = j ^ 6u;
                p[7] = j ^ 7u;
            }
            for (uint32_t b = 0; b < 16; b++) {
                const uint32_t j = i + b;
                noc_async_write(a.scratch + b * 32u,
                                get_noc_addr((salt * 131u + j * 5u) & pmask, data_acc) + (j & 31u) * 32u, 32u);
            }
            noc_async_write_barrier();
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, 0);
}
