// T2 param pass writer (RISCV_0, #697). Per processed group (index from the reader's header):
// params (fp32), m and v (bf16) pages back in place, the 13 pfwc streams as 256 B SoA pieces
// (page s * n_groups + g), and with T2_GRADS_OUT the grads page.
// Runtime args: params, m, v, pfwc, grads DRAM addresses, group count, n_groups (total).
// Compile args: TensorAccessorArgs(params, m, v, pfwc, grads).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_WH = 5, CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_FO = 19, CB_GO = 20;
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048, F_PAGE = 256, NPFWC = 13;
}  // namespace

void kernel_main() {
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t f_addr = get_arg_val<uint32_t>(3);
    const uint32_t g_addr = get_arg_val<uint32_t>(4);
    const uint32_t n = get_arg_val<uint32_t>(5);
    const uint32_t n_groups = get_arg_val<uint32_t>(6);
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ma = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto fa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<fa.next_compile_time_args_offset()>();
    const auto P = TensorAccessor(pa, p_addr, P_PAGE);
    const auto M = TensorAccessor(ma, m_addr, MV_PAGE);
    const auto V = TensorAccessor(va, v_addr, MV_PAGE);
    const auto F = TensorAccessor(fa, f_addr, F_PAGE);
    const auto GR = TensorAccessor(ga, g_addr, P_PAGE);
    (void)GR;
    for (uint32_t j = 0; j < n; j++) {
        cb_wait_front(CB_WH, 1);
        const uint32_t g = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_WH))[0];
        cb_pop_front(CB_WH, 1);
        cb_wait_front(CB_PO, 1);
        noc_async_write(get_read_ptr(CB_PO), get_noc_addr(g, P), P_PAGE);
        cb_wait_front(CB_MO, 1);
        noc_async_write(get_read_ptr(CB_MO), get_noc_addr(g, M), MV_PAGE);
        cb_wait_front(CB_VO, 1);
        noc_async_write(get_read_ptr(CB_VO), get_noc_addr(g, V), MV_PAGE);
        cb_wait_front(CB_FO, 1);
        const uint32_t fo = get_read_ptr(CB_FO);
        for (uint32_t s = 0; s < NPFWC; s++) {
            noc_async_write(fo + s * F_PAGE, get_noc_addr(s * n_groups + g, F), F_PAGE);
        }
#ifdef T2_GRADS_OUT
        cb_wait_front(CB_GO, 1);
        noc_async_write(get_read_ptr(CB_GO), get_noc_addr(g, GR), P_PAGE);
#endif
        noc_async_writes_flushed();
        cb_pop_front(CB_PO, 1);
        cb_pop_front(CB_MO, 1);
        cb_pop_front(CB_VO, 1);
        cb_pop_front(CB_FO, 1);
#ifdef T2_GRADS_OUT
        cb_pop_front(CB_GO, 1);
#endif
    }
    noc_async_write_barrier();
}
