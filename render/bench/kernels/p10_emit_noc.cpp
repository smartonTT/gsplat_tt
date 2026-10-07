// p10 (task #365): emit-shaped DRAM traffic per mover, to separate raw per-core
// NoC/DRAM throughput from the sort emit's kernel logic. n transfers of MB_XFER
// bytes, MB_DEPTH outstanding per barrier (reads) or flush (writes):
//   mode 0: reads from the 64 B-page buffer (blendrec / pair pages), each a run of
//           MB_XFER / 64 pages inside one DRAM bank (the emit's bulk reads).
//   mode 1: writes into the 2 KB-page bucket buffer at page t * MB_STRIDE + k, t
//           cycling over MB_TILES tiles, k = this mover's prefix position (slot /
//           slots * MB_KSPAN, advancing every 1024 writes): the emit's ring flushes.
#include "mb_common.h"

#ifndef MB_DATA_PAGE
#define MB_DATA_PAGE 64u
#endif
#ifndef MB_XFER
#define MB_XFER 64u
#endif
#ifndef MB_DEPTH
#define MB_DEPTH 8u
#endif
#ifndef MB_STRIDE
#define MB_STRIDE 512u
#endif
#ifndef MB_TILES
#define MB_TILES 256u
#endif
#ifndef MB_KSPAN
#define MB_KSPAN 32u
#endif
#ifndef MB_SLOTS
#define MB_SLOTS 220u
#endif

void kernel_main() {
    const mb::Args a = mb::args();
    MB_DATA_ACC(a);
    const uint32_t k0 = (a.slot * MB_KSPAN) / MB_SLOTS;
    uint32_t h = a.slot * 2654435761u + 12345u;
    uint64_t t0, t1;
    {
        DeviceZoneScopedN("p10_emit_noc");
        t0 = mb::now();
        if (a.mode == 0) {
            const uint32_t pmask = (a.data_npages >> 2) - 1u;  // runs stay inside the buffer
            for (uint32_t i = 0; i < a.n; i += MB_DEPTH) {
                for (uint32_t d = 0; d < MB_DEPTH; d++) {
                    h = h * 1664525u + 1013904223u;
                    noc_async_read(get_noc_addr((h >> 8) & pmask, data_acc), a.scratch + d * MB_XFER, MB_XFER);
                }
                noc_async_read_barrier();
            }
        } else {
            for (uint32_t i = 0; i < a.n; i += MB_DEPTH) {
                for (uint32_t d = 0; d < MB_DEPTH; d++) {
                    const uint32_t j = i + d;
                    const uint32_t t = (j * 7u + a.slot) % MB_TILES;
                    const uint32_t k = (k0 + (j >> 10)) % MB_KSPAN;
                    const uint32_t off = ((j / MB_TILES) * MB_XFER) & (MB_DATA_PAGE - 1u);
                    noc_async_write(a.scratch + d * MB_XFER, get_noc_addr(t * MB_STRIDE + k, data_acc) + off, MB_XFER);
                }
                noc_async_writes_flushed();
            }
            noc_async_write_barrier();
        }
        t1 = mb::now();
    }
    mb::publish(a, a.n, t1 - t0, 0, 0, 0);
}
