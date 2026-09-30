// p4: noc_async_write issue cost, 32 B L1 -> DRAM scatter, batches of 16,
// one noc_async_write_barrier per batch (the sort_bin flush_recs shape).
//   aux A = ticks spent issuing (barrier excluded), aux B = ticks in barriers.
//   MB_ACC=0: destination NoC addresses precomputed (pure issue cost)
//   MB_ACC=1: get_noc_addr(page, accessor) per write, as production does.
#include "mb_common.h"

#define MB_DATA_PAGE 1024u

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    const uint32_t pmask = a.data_npages - 1u;
    uint64_t pre[16];
    for (uint32_t b = 0; b < 16; b++) pre[b] = get_noc_addr((a.salt * 131u + b * 7u) & pmask, data_acc);
    uint64_t issue = 0, bar = 0, t0, t1;
    {
        DeviceZoneScopedN("p4_noc_write_issue");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i += 16) {
            const uint64_t ta = mb::now();
            for (uint32_t b = 0; b < 16; b++) {
                const uint32_t j = i + b;
                const uint32_t off = (j & 31u) * 32u;
#if MB_ACC
                const uint64_t dst = get_noc_addr((a.salt * 131u + j * 7u) & pmask, data_acc) + off;
#else
                const uint64_t dst = pre[b] + off;
#endif
                noc_async_write(a.scratch + b * 32u, dst, 32u);
            }
            const uint64_t tb = mb::now();
            noc_async_write_barrier();
            const uint64_t tc = mb::now();
            issue += tb - ta;
            bar += tc - tb;
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, issue, bar, 0);
}
