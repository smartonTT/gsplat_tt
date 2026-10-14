// Task #736: writer for ssim_fwd_compute.cpp: per output tile (t = strip * nh + row) the d maps
// d_mu, d_xx, d_xy (CB 16, 3 pages) -> planes m * 3 + ch of the d buffer; at the end the core's
// loss partial tile (CB 20) -> page `core` of the partials buffer.
// Runtime args: d_addr, s_addr, u0, u1, nh, nw, core. #753: work units u of SSIM_K-column groups
// (reader_ssim.cpp), output tiles in the compute's order (unit by unit, columns j0 .. j0+kw-1).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

#ifndef SSIM_K
#define SSIM_K 1
#endif

constexpr uint32_t CB_D = 16, CB_SUM = 20;

void kernel_main() {
    const uint32_t d_addr = get_arg_val<uint32_t>(0);
    const uint32_t s_addr = get_arg_val<uint32_t>(1);
    const uint32_t u0 = get_arg_val<uint32_t>(2);
    const uint32_t u1 = get_arg_val<uint32_t>(3);
    const uint32_t nh = get_arg_val<uint32_t>(4);
    const uint32_t nw = get_arg_val<uint32_t>(5);
    const uint32_t core = get_arg_val<uint32_t>(6);
    constexpr auto ad = TensorAccessorArgs<0>();
    constexpr auto as = TensorAccessorArgs<ad.next_compile_time_args_offset()>();
    const uint32_t tb = get_tile_size(CB_D);
    const auto D = TensorAccessor(ad, d_addr, tb);
    const auto S = TensorAccessor(as, s_addr, tb);
    const uint32_t ng = (nw + SSIM_K - 1) / SSIM_K;
    for (uint32_t u = u0; u < u1; u++) {
      const uint32_t sx = u / nh, i = u % nh, ch = sx / ng, j0 = (sx % ng) * SSIM_K;
      for (uint32_t j = j0; j < j0 + SSIM_K && j < nw; j++) {
        cb_wait_front(CB_D, 3);
        const uint32_t r = get_read_ptr(CB_D);
#if !(defined(SSIM_DMABL) && (SSIM_DMABL & 2))  // #743 timing ablation: skip the writes
        for (uint32_t m = 0; m < 3; m++) noc_async_write_tile(((m * 3 + ch) * nh + i) * nw + j, D, r + m * tb);
#endif
        noc_async_writes_flushed();
        cb_pop_front(CB_D, 3);
      }
    }
    cb_wait_front(CB_SUM, 1);
    noc_async_write_tile(core, S, get_read_ptr(CB_SUM));
    noc_async_write_barrier();
    cb_pop_front(CB_SUM, 1);
}
