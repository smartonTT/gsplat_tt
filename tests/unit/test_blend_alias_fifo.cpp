// Task #544: blend_fifo_shrink_iface (render/kernels/blend_alias_fifo.h).
//
//   tests/unit/run_cpp.sh tests/unit/test_blend_alias_fifo.cpp
//
// Checks, on a stand-in for tt-metal's LocalCBInterface initialised the way
// circular_buffer_init.h does for one index of a multi-index config (fifo size =
// the config's total size): the shrink keeps the base, cuts size/limit/num_pages to
// the ring's own size on both the mover (bytes) and TRISC (16 B words) scales; a
// ring that wraps after the shrink wraps at base + size, never in the mat region
// beyond it; and a CB already at its own size (unfused program) is left alone.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#define BLEND_ALIAS_FIFO_HOST_TEST 1
#include "render/kernels/blend_alias_fifo.h"

static int fails = 0;
#define CHECK(c)                                                             \
    do {                                                                     \
        if (!(c)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                         \
        }                                                                    \
    } while (0)

struct Iface {
    uint32_t fifo_size, fifo_limit, fifo_page_size, fifo_num_pages, fifo_rd_ptr, fifo_wr_ptr;
};

// circular_buffer_init.h for one index: fields in (bytes >> shift), num_pages from the host
// (config total / this index's page size).
static Iface init(uint32_t addr, uint32_t total, uint32_t page, uint32_t shift) {
    Iface c{};
    c.fifo_size = total >> shift;
    c.fifo_limit = (addr >> shift) + (total >> shift);
    c.fifo_page_size = page >> shift;
    c.fifo_num_pages = total / page;
    c.fifo_rd_ptr = c.fifo_wr_ptr = addr >> shift;
    return c;
}

// cb_push_back / cb_pop_front pointer step with wrap (llk_io_pack.h, dataflow_api.h).
static void step(const Iface& c, uint32_t& ptr, uint32_t pages) {
    ptr += pages * c.fifo_page_size;
    if (ptr >= c.fifo_limit) ptr -= c.fifo_size;
}

int main() {
    // CB_OUT on mat CB 6 (fused default): 264,192 B config, 2 KB bf16 pages, real ring 6 pages.
    for (uint32_t shift : {0u, 4u}) {
        const uint32_t addr = 0x60000u;
        Iface c = init(addr, 264192u, 2048u, shift);
        CHECK(c.fifo_num_pages == 129u);
        blend_fifo_shrink_iface(c, 12288u, 2048u, shift);
        CHECK(c.fifo_size == (12288u >> shift));
        CHECK(c.fifo_limit == ((addr + 12288u) >> shift));
        CHECK(c.fifo_num_pages == 6u);
        CHECK(c.fifo_rd_ptr == (addr >> shift) && c.fifo_wr_ptr == (addr >> shift));
        // The writer pops 3 pages at a time: the ring returns to the base after 2 pops
        // and never reaches the mat region past base + 12,288 B.
        uint32_t ptr = c.fifo_wr_ptr;
        for (int i = 0; i < 1000; ++i) {
            step(c, ptr, 3u);
            CHECK(ptr < ((addr + 12288u) >> shift));
        }
        CHECK(ptr == (addr >> shift));  // 1000 pushes of 3 pages = 500 laps
    }
    // CB_MB_COUNTS on mat CB 5: 132,096 B config, 128 B pages, real ring 64 pages.
    {
        Iface c = init(0x10000u, 132096u, 128u, 4u);
        CHECK(c.fifo_num_pages == 1032u);
        blend_fifo_shrink_iface(c, 8192u, 128u, 4u);
        CHECK(c.fifo_num_pages == 64u && c.fifo_size == 512u && c.fifo_limit == 0x1000u + 512u);
    }
    // Unfused program: the CB already has its own size, nothing changes.
    {
        Iface c = init(0x20000u, 12288u, 2048u, 0u);
        const Iface before = c;
        blend_fifo_shrink_iface(c, 12288u, 2048u, 0u);
        CHECK(c.fifo_size == before.fifo_size && c.fifo_limit == before.fifo_limit &&
              c.fifo_num_pages == before.fifo_num_pages);
    }
    if (fails == 0) std::printf("test_blend_alias_fifo: OK\n");
    return fails == 0 ? 0 : 1;
}
