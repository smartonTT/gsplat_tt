// Task #706: writer for filt_compute.cpp: output tiles t in [t0, t1), t = strip * nh + row,
// strip s = (channel s / nw, tile column s % nw).
// Runtime args: o_addr, t0, t1, nh, nw.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t t0 = get_arg_val<uint32_t>(1);
    const uint32_t t1 = get_arg_val<uint32_t>(2);
    const uint32_t nh = get_arg_val<uint32_t>(3);
    const uint32_t nw = get_arg_val<uint32_t>(4);
    constexpr auto ao = TensorAccessorArgs<0>();
    const uint32_t tb = get_tile_size(CB_OUT);
    const auto acc_o = TensorAccessor(ao, o_addr, tb);
#ifdef SSIM_K  // #753 ssim bwd: t0, t1 are work units of SSIM_K-column groups (reader_ssim.cpp)
    const uint32_t ng = (nw + SSIM_K - 1) / SSIM_K;
    for (uint32_t u = t0; u < t1; u++) {
      const uint32_t sx = u / nh, i = u % nh, ch = sx / ng, j0 = (sx % ng) * SSIM_K;
      for (uint32_t j = j0; j < j0 + SSIM_K && j < nw; j++) {
#else
    for (uint32_t t = t0; t < t1; t++) {
      {
        const uint32_t sx = t / nh, i = t % nh, ch = sx / nw, j = sx % nw;
#endif
        cb_wait_front(CB_OUT, 1);
#if !(defined(SSIM_DMABL) && (SSIM_DMABL & 2))  // #743 timing ablation: skip the writes
        noc_async_write_tile((ch * nh + i) * nw + j, acc_o, get_read_ptr(CB_OUT));
#endif
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
      }
    }
    noc_async_write_barrier();
}
