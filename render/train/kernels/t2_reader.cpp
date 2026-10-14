// T2 param pass reader (RISCV_1, #697, segment matmul #713). Blocks of 1024 rows (16 groups of 64)
// round-robin over the cores; a block whose PFWC_VIS words are all zero is skipped whole, a group
// whose two words are zero too. Per group: header to compute (window count) and writer (group
// index), the param / m / v pages, then one window per T1 v2 slot page of the group's span
// [P[g0], P[g0+64]): the page's bf16 pieces hi, mid, lo (DRAM pages 3q..3q+2; each 64 slots x 16
// streams = one bf16 tile, slot j = face j/16, row j%16, streams in columns) to CB_K, and four bf16 0/1 segment tiles to CB_A. Tile t row r,
// column rho is 1 when slot sigma(rho) of the page belongs to the group and to lane 32*(t&1)+r;
// tiles 0/1 cover the left column half of the page tile (slots 0-15, 32-47), tiles 2/3 the right
// (16-31, 48-63). Slots of other groups get no 1. The compute does the segment sum as A x page on
// the FPU. CB_A pages start zeroed; each window clears the 1s its ring entry set last time.
// Runtime args: params, m, v, slots, meta, vis DRAM addresses, first block, block stride,
// block count (total). Compile args: TensorAccessorArgs(params, m, v, slots, meta, vis).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_WH = 5, CB_WS = 6, CB_A = 7;
constexpr uint32_t A_TILE = 2048, A_ENTRY = 4 * A_TILE, BF16_ONE = 0x3F80u;
constexpr uint32_t GPB = 16, SLOT_PAGE = 2048, SLOTS_PER_PAGE = 64;
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048, META_PAGE = 512, VIS_PAGE = 128;
}  // namespace

void kernel_main() {
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t meta_addr = get_arg_val<uint32_t>(4);
    const uint32_t vis_addr = get_arg_val<uint32_t>(5);
    const uint32_t b0 = get_arg_val<uint32_t>(6);
    const uint32_t bstride = get_arg_val<uint32_t>(7);
    const uint32_t nblocks = get_arg_val<uint32_t>(8);
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ma = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    const auto P = TensorAccessor(pa, p_addr, P_PAGE);
    const auto M = TensorAccessor(ma, m_addr, MV_PAGE);
    const auto V = TensorAccessor(va, v_addr, MV_PAGE);
    const auto SL = TensorAccessor(sa, s_addr, SLOT_PAGE);
    const auto MT = TensorAccessor(ta, meta_addr, META_PAGE);
    const auto VI = TensorAccessor(ia, vis_addr, VIS_PAGE);
    // Scratch (CB_WS, never pushed): vis words, meta page, then per CB_A ring entry the 64 u16
    // positions it set (0xFFFF none).
    const uint32_t vis_l1 = get_write_ptr(CB_WS);
    const uint32_t meta_l1 = vis_l1 + VIS_PAGE;
    volatile uint32_t* vis = reinterpret_cast<volatile uint32_t*>(vis_l1);
    volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(meta_l1);
    volatile uint16_t* recs = reinterpret_cast<volatile uint16_t*>(meta_l1 + META_PAGE);
    cb_reserve_back(CB_A, 4 * T2_A_ENTRIES);
    const uint32_t a_base = get_write_ptr(CB_A);
    {
        volatile uint32_t* z = reinterpret_cast<volatile uint32_t*>(a_base);
        for (uint32_t w = 0; w < T2_A_ENTRIES * A_ENTRY / 4; w++) z[w] = 0;
        for (uint32_t j = 0; j < T2_A_ENTRIES * 64; j++) recs[j] = 0xFFFF;
    }

    for (uint32_t b = b0; b < nblocks; b += bstride) {
        noc_async_read(get_noc_addr(b, VI), vis_l1, VIS_PAGE);
        noc_async_read_barrier();
        uint32_t any = 0;
        for (uint32_t w = 0; w < 32; w++) any |= vis[w];
        if (any == 0) continue;
        for (uint32_t gi = 0; gi < GPB; gi++) {
            if ((vis[2 * gi] | vis[2 * gi + 1]) == 0) continue;
            const uint32_t g = b * GPB + gi;
            noc_async_read(get_noc_addr(g, MT), meta_l1, META_PAGE);
            cb_reserve_back(CB_P, 1);
            noc_async_read(get_noc_addr(g, P), get_write_ptr(CB_P), P_PAGE);
            cb_reserve_back(CB_M, 1);
            noc_async_read(get_noc_addr(g, M), get_write_ptr(CB_M), MV_PAGE);
            cb_reserve_back(CB_V, 1);
            noc_async_read(get_noc_addr(g, V), get_write_ptr(CB_V), MV_PAGE);
            noc_async_read_barrier();
            // Slot span: P of lane 0 to P + count of lane 63 (P is an exclusive prefix sum).
            const uint32_t s0 = meta[0], s1 = meta[63] + meta[127];
            const uint32_t pg0 = s0 / SLOTS_PER_PAGE;
            const uint32_t npg = s1 > s0 ? (s1 - 1) / SLOTS_PER_PAGE - pg0 + 1 : 0;
            // Headers (the compute reads its own; the writer gets the group index).
            cb_reserve_back(CB_HDR, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_HDR))[0] = npg;
            cb_push_back(CB_HDR, 1);
            cb_reserve_back(CB_WH, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_WH))[0] = g;
            cb_push_back(CB_WH, 1);
            cb_push_back(CB_P, 1);
            cb_push_back(CB_M, 1);
            cb_push_back(CB_V, 1);
            uint32_t li = 0;  // lane cursor (slots are in lane order)
            for (uint32_t q = 0; q < npg; q++) {
                cb_reserve_back(CB_K, 3);
                const uint32_t kl1 = get_write_ptr(CB_K);
                for (uint32_t k = 0; k < 3; k++)
                    noc_async_read(get_noc_addr(3 * (pg0 + q) + k, SL), kl1 + k * SLOT_PAGE, SLOT_PAGE);
                cb_reserve_back(CB_A, 4);
                const uint32_t al1 = get_write_ptr(CB_A);
                volatile uint32_t* aw = reinterpret_cast<volatile uint32_t*>(al1);
                volatile uint16_t* rec = recs + ((al1 - a_base) / A_ENTRY) * 64;
                for (uint32_t j = 0; j < 64; j++) {
                    if (rec[j] != 0xFFFF) aw[rec[j] >> 1] = 0;
                    rec[j] = 0xFFFF;
                }
                const uint32_t pb = (pg0 + q) * SLOTS_PER_PAGE;
                const uint32_t lo = s0 > pb ? s0 : pb;
                const uint32_t hi = s1 < pb + SLOTS_PER_PAGE ? s1 : pb + SLOTS_PER_PAGE;
                for (uint32_t sl = lo; sl < hi; sl++) {
                    while (li < 63 && sl >= meta[li] + meta[64 + li]) li++;
                    const uint32_t j = sl - pb;
                    const uint32_t rho = (j & 15) | ((j >> 5) << 4);
                    const uint32_t t = ((j >> 4) & 1) * 2 + (li >> 5), r = li & 31;
                    const uint32_t pos = t * 1024 + (((r >> 4) << 1) | (rho >> 4)) * 256 + (r & 15) * 16 + (rho & 15);
                    aw[pos >> 1] |= BF16_ONE << ((pos & 1) * 16);
                    rec[j] = pos;
                }
                noc_async_read_barrier();
                cb_push_back(CB_K, 3);
                cb_push_back(CB_A, 4);
            }
        }
    }
}
