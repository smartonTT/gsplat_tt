// p8: fixed cost of a barrier with nothing outstanding.
//   aux A = noc_async_write_barrier(), aux B = noc_async_read_barrier(); n each.
#include "mb_common.h"

void kernel_main() {
    const mb::Args a = mb::args();
    uint64_t t0, t1, t2;
    {
        DeviceZoneScopedN("p8_write_barrier_empty");
        t0 = mb::now();
        for (uint32_t i = 0; i < a.n; i++) noc_async_write_barrier();
        t1 = mb::now();
    }
    {
        DeviceZoneScopedN("p8_read_barrier_empty");
        for (uint32_t i = 0; i < a.n; i++) noc_async_read_barrier();
        t2 = mb::now();
    }
    mb::publish(a, a.n, t2 - t0, t1 - t0, t2 - t1, 0);
}
