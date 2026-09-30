// p7: achieved DRAM bandwidth, large sequential transfers. n = 8 KiB pages per
// core; 8 transfers outstanding per barrier into a 4-page (32 KiB) L1 ring, so
// two movers on one core fit in disjoint halves of the 64 KiB scratch CB.
//   mode 0: noc_async_read  (DRAM -> L1)
//   mode 1: noc_async_write (L1 -> DRAM)
#include "mb_common.h"

#define MB_DATA_PAGE 8192u

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    const uint32_t pmask = a.data_npages - 1u;
    const uint32_t start = a.salt * 64u;
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p7_dram_bw");
        t0 = mb::now();
        if (a.mode == 0) {
            for (uint32_t i = 0; i < a.n; i += 8) {
                for (uint32_t d = 0; d < 8; d++)
                    noc_async_read(get_noc_addr((start + i + d) & pmask, data_acc), a.scratch + (d & 3u) * MB_DATA_PAGE,
                                   MB_DATA_PAGE);
                noc_async_read_barrier();
            }
        } else {
            for (uint32_t i = 0; i < a.n; i += 8) {
                for (uint32_t d = 0; d < 8; d++)
                    noc_async_write(a.scratch + (d & 3u) * MB_DATA_PAGE, get_noc_addr((start + i + d) & pmask, data_acc),
                                    MB_DATA_PAGE);
                noc_async_write_barrier();
            }
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, 0);
}
