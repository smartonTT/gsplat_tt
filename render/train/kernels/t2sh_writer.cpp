// #751: writer for the t2sh passes. pre: aux page g from CB_PO. post: per r, rest params (CB_PO),
// m (CB_MO), v (CB_VO) and rest grads (CB_GO) to page 3 g + r (params / m / v in place).
// Runtime args: pre: aux address, n; post: rest, m, v, rest-grads addresses, n.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_WH = 5, CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_GO = 20;
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048;
}  // namespace

void kernel_main() {
#ifdef T2SH_POST
    constexpr auto ra = TensorAccessorArgs<0>();
    constexpr auto ma = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    const auto R = TensorAccessor(ra, get_arg_val<uint32_t>(0), P_PAGE);
    const auto M = TensorAccessor(ma, get_arg_val<uint32_t>(1), MV_PAGE);
    const auto V = TensorAccessor(va, get_arg_val<uint32_t>(2), MV_PAGE);
    const auto GR = TensorAccessor(ga, get_arg_val<uint32_t>(3), P_PAGE);
    const uint32_t n = get_arg_val<uint32_t>(4);
#else
    constexpr auto xa = TensorAccessorArgs<0>();
    const auto AX = TensorAccessor(xa, get_arg_val<uint32_t>(0), P_PAGE);
    const uint32_t n = get_arg_val<uint32_t>(1);
#endif
    for (uint32_t j = 0; j < n; j++) {
        cb_wait_front(CB_WH, 1);
        const uint32_t g = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_WH))[0];
        cb_pop_front(CB_WH, 1);
#ifdef T2SH_POST
        for (uint32_t r = 0; r < 3; r++) {
            const uint32_t pg = 3 * g + r;
            cb_wait_front(CB_PO, 1);
            noc_async_write(get_read_ptr(CB_PO), get_noc_addr(pg, R), P_PAGE);
            cb_wait_front(CB_MO, 1);
            noc_async_write(get_read_ptr(CB_MO), get_noc_addr(pg, M), MV_PAGE);
            cb_wait_front(CB_VO, 1);
            noc_async_write(get_read_ptr(CB_VO), get_noc_addr(pg, V), MV_PAGE);
            cb_wait_front(CB_GO, 1);
            noc_async_write(get_read_ptr(CB_GO), get_noc_addr(pg, GR), P_PAGE);
            noc_async_writes_flushed();
            cb_pop_front(CB_PO, 1);
            cb_pop_front(CB_MO, 1);
            cb_pop_front(CB_VO, 1);
            cb_pop_front(CB_GO, 1);
        }
#else
        cb_wait_front(CB_PO, 1);
        noc_async_write(get_read_ptr(CB_PO), get_noc_addr(g, AX), P_PAGE);
        noc_async_writes_flushed();
        cb_pop_front(CB_PO, 1);
#endif
    }
    noc_async_write_barrier();
}
