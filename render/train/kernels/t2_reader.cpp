// T2 param pass reader (RISCV_1, #697). Blocks of 1024 rows (16 groups of 64) round-robin over
// the cores; a block whose PFWC_VIS words are all zero is skipped whole, a group whose two
// words are zero too. Per group: header to compute (rank-plane count) and writer (group index),
// the param / m / v pages, then the rank planes: the group's slot span [P[g0], P[g0+64]) is one
// contiguous run of T1 v2 slot pages (64 per 4 KB DRAM page), read in bulk into L1; plane k holds
// slot P[g] + k of each lane (term t at stream t), zero past counts[g]. Spans over the staging
// size fall back to one 64 B read per slot.
// Runtime args: params, m, v, slots, meta, vis DRAM addresses, first block, block stride,
// block count (total). Compile args: TensorAccessorArgs(params, m, v, slots, meta, vis).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_WH = 5, CB_WS = 6;
constexpr uint32_t GPB = 16, SLOT_PAGE = 4096, SLOTS_PER_PAGE = 64, SLOT_WORDS = 16;
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
    // Scratch (CB_WS, never pushed): vis words, meta page, then the slot staging pages.
    const uint32_t vis_l1 = get_write_ptr(CB_WS);
    const uint32_t meta_l1 = vis_l1 + VIS_PAGE;
    const uint32_t stg_l1 = meta_l1 + META_PAGE;
    volatile uint32_t* vis = reinterpret_cast<volatile uint32_t*>(vis_l1);
    volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(meta_l1);
    volatile uint32_t* stg = reinterpret_cast<volatile uint32_t*>(stg_l1);

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
            uint32_t kmax = 0;
            for (uint32_t i = 0; i < 64; i++) kmax = meta[64 + i] > kmax ? meta[64 + i] : kmax;
            const uint32_t nk = kmax ? kmax : 1;
            // Slot span: P of lane 0 to P + count of lane 63 (P is an exclusive prefix sum).
            const uint32_t s0 = meta[0], s1 = meta[63] + meta[127];
            const uint32_t pg0 = s0 / SLOTS_PER_PAGE;
            const uint32_t npg = s1 > s0 ? (s1 - 1) / SLOTS_PER_PAGE - pg0 + 1 : 0;
            const bool staged = npg <= T2_STAGE_PAGES;
            if (staged) {
                for (uint32_t q = 0; q < npg; q++) {
                    noc_async_read(get_noc_addr(pg0 + q, SL), stg_l1 + q * SLOT_PAGE, SLOT_PAGE);
                }
            }
            // Headers (the compute reads its own; the writer gets the group index).
            cb_reserve_back(CB_HDR, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_HDR))[0] = nk;
            cb_push_back(CB_HDR, 1);
            cb_reserve_back(CB_WH, 1);
            reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_WH))[0] = g;
            cb_push_back(CB_WH, 1);
            cb_push_back(CB_P, 1);
            cb_push_back(CB_M, 1);
            cb_push_back(CB_V, 1);
            noc_async_read_barrier();
            const uint32_t base = pg0 * SLOTS_PER_PAGE;
            for (uint32_t k = 0; k < nk; k++) {
                cb_reserve_back(CB_K, 1);
                const uint32_t kl1 = get_write_ptr(CB_K);
                volatile uint32_t* t = reinterpret_cast<volatile uint32_t*>(kl1);
                for (uint32_t i = 0; i < 64; i++) {
                    if (k < meta[64 + i]) {
                        const uint32_t sl = meta[i] + k;
                        const volatile uint32_t* src;
                        if (staged) {
                            src = stg + (sl - base) * SLOT_WORDS;
                        } else {
                            noc_async_read(
                                get_noc_addr(sl / SLOTS_PER_PAGE, SL, (sl % SLOTS_PER_PAGE) * 64), stg_l1, 64);
                            noc_async_read_barrier();
                            src = stg;
                        }
                        for (uint32_t s = 0; s < 9; s++) t[64 * s + i] = src[s];
                    } else {
                        for (uint32_t s = 0; s < 9; s++) t[64 * s + i] = 0;
                    }
                }
                cb_push_back(CB_K, 1);
            }
        }
    }
}
