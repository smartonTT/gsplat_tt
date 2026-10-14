// #751: reader for the t2sh pre / post passes (t2sh_compute.cpp). Same block walk and group skip
// as t2g_reader (blocks b0, b0 + bstride, ...; groups with no visible lane are skipped), so the
// passes touch exactly T2's groups. Pages per group g:
//   pre : CB_A params page g, CB_B rest pages 3 g .. 3 g + 2
//   post: CB_A aux page g, CB_G T2 grads page g, then per r: CB_B rest, CB_M m, CB_V v page 3 g + r
// Runtime args: pre: params, rest, vis addresses, b0, bstride, nblocks
//               post: aux, grads, rest, m, v, vis addresses, b0, bstride, nblocks
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_A = 1, CB_B = 2, CB_M = 3, CB_V = 4, CB_WH = 5, CB_WS = 6, CB_G = 7;
constexpr uint32_t GPB = 16, P_PAGE = 4096, MV_PAGE = 2048, VIS_PAGE = 128;
#ifdef T2SH_POST
constexpr uint32_t NA = 6;
#else
constexpr uint32_t NA = 3;
#endif
}  // namespace

void kernel_main() {
    const uint32_t b0 = get_arg_val<uint32_t>(NA);
    const uint32_t bstride = get_arg_val<uint32_t>(NA + 1);
    const uint32_t nblocks = get_arg_val<uint32_t>(NA + 2);
#ifdef T2SH_POST
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto ga = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ra = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const auto AX = TensorAccessor(xa, get_arg_val<uint32_t>(0), P_PAGE);
    const auto GR = TensorAccessor(ga, get_arg_val<uint32_t>(1), P_PAGE);
    const auto R = TensorAccessor(ra, get_arg_val<uint32_t>(2), P_PAGE);
    const auto M = TensorAccessor(ma, get_arg_val<uint32_t>(3), MV_PAGE);
    const auto V = TensorAccessor(va, get_arg_val<uint32_t>(4), MV_PAGE);
    const auto VI = TensorAccessor(ia, get_arg_val<uint32_t>(5), VIS_PAGE);
#else
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ra = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    const auto P = TensorAccessor(pa, get_arg_val<uint32_t>(0), P_PAGE);
    const auto R = TensorAccessor(ra, get_arg_val<uint32_t>(1), P_PAGE);
    const auto VI = TensorAccessor(ia, get_arg_val<uint32_t>(2), VIS_PAGE);
#endif
    const uint32_t vis_l1 = get_write_ptr(CB_WS);
    volatile uint32_t* vis = reinterpret_cast<volatile uint32_t*>(vis_l1);

    for (uint32_t b = b0; b < nblocks; b += bstride) {
        noc_async_read(get_noc_addr(b, VI), vis_l1, VIS_PAGE);
        noc_async_read_barrier();
        uint32_t any = 0;
        for (uint32_t w = 0; w < 32; w++) any |= vis[w];
        if (any == 0) continue;
        for (uint32_t gi = 0; gi < GPB; gi++) {
            if ((vis[2 * gi] | vis[2 * gi + 1]) == 0) continue;
            const uint32_t g = b * GPB + gi;
            cb_reserve_back(CB_WH, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_WH))[0] = g;
            cb_push_back(CB_WH, 1);
#ifdef T2SH_POST
            cb_reserve_back(CB_A, 1);
            noc_async_read(get_noc_addr(g, AX), get_write_ptr(CB_A), P_PAGE);
            cb_reserve_back(CB_G, 1);
            noc_async_read(get_noc_addr(g, GR), get_write_ptr(CB_G), P_PAGE);
            noc_async_read_barrier();
            cb_push_back(CB_A, 1);
            cb_push_back(CB_G, 1);
            for (uint32_t r = 0; r < 3; r++) {
                cb_reserve_back(CB_B, 1);
                noc_async_read(get_noc_addr(3 * g + r, R), get_write_ptr(CB_B), P_PAGE);
                cb_reserve_back(CB_M, 1);
                noc_async_read(get_noc_addr(3 * g + r, M), get_write_ptr(CB_M), MV_PAGE);
                cb_reserve_back(CB_V, 1);
                noc_async_read(get_noc_addr(3 * g + r, V), get_write_ptr(CB_V), MV_PAGE);
                noc_async_read_barrier();
                cb_push_back(CB_B, 1);
                cb_push_back(CB_M, 1);
                cb_push_back(CB_V, 1);
            }
#else
            cb_reserve_back(CB_A, 1);
            noc_async_read(get_noc_addr(g, P), get_write_ptr(CB_A), P_PAGE);
            cb_reserve_back(CB_B, 3);
            const uint32_t bl1 = get_write_ptr(CB_B);
            for (uint32_t r = 0; r < 3; r++) noc_async_read(get_noc_addr(3 * g + r, R), bl1 + r * P_PAGE, P_PAGE);
            noc_async_read_barrier();
            cb_push_back(CB_A, 1);
            cb_push_back(CB_B, 3);
#endif
        }
    }
}
