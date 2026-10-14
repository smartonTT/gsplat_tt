// #748: the t697 gather-reduce T2 kernel (origin/ttp/t697 head), kept for the on-device training
// step chain (render/train/train_step.cpp): T1 cannot emit #726's block-float pieces (docs/t740).
// T2 param pass reader (RISCV_1, #697). Blocks of 1024 rows (16 groups of 64) round-robin over
// the cores; a block whose PFWC_VIS words are all zero is skipped whole, a group whose two
// words are zero too. Per group: header to compute (rank-plane count) and writer (group index),
// the param / m / v pages, then the rank planes: the group's slot span [P[g0], P[g0+64]) is one
// contiguous run of T1 v2 slot pages (64 per 4 KB DRAM page), read in bulk into L1; plane k holds
// slot P[g] + k of each lane (term t at stream t), zero past counts[g]. Spans over the staging
// size fall back to one 64 B read per slot.
// Runtime args: params, m, v, slots, meta, vis DRAM addresses, first block, block stride,
// block count (total). Compile args: TensorAccessorArgs(params, m, v, slots, meta, vis).
// #784 heavy-row split (gsplat/t2_split.py, docs/t784):
//  T2_PHASE_A  heavy groups (T2_HL lanes of rows with k > kcap, host table htab in the meta page
//              layout). Lane i walks its own slot run P[i] .. P[i]+k[i]-1, read in chunks of
//              T2_STAGE_PAGES * 64 / T2_HL slots per lane; the group runs k[0] planes (k[0] is the
//              largest). Only headers and planes go to the compute (no param pages).
//              Runtime args: slots, htab DRAM addresses, group count, then the group ids.
//  T2_SPLIT    light groups [first, end) (one contiguous range per core). Meta is the host's
//              split copy: a heavy lane's count word is 0x80000000 | h, and its plane 0 is the
//              phase-A sum (64 B record h of hsum), planes 1.. are zero. Only the slot pages the
//              light lanes touch are staged. Runtime args: params, m, v, slots, meta, vis, hsum
//              DRAM addresses, first group, end group (T2_SH: then the aux address, #798).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_WH = 5, CB_WS = 6, CB_AUX = 7;
constexpr uint32_t GPB = 16, SLOT_PAGE = 4096, SLOTS_PER_PAGE = 64, SLOT_WORDS = 16;
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048, META_PAGE = 512, VIS_PAGE = 128;
}  // namespace

#if defined(T2_PHASE_A)
void kernel_main() {
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t h_addr = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto ha = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto SL = TensorAccessor(sa, s_addr, SLOT_PAGE);
    const auto HT = TensorAccessor(ha, h_addr, META_PAGE);
    constexpr uint32_t C = T2_STAGE_PAGES * SLOTS_PER_PAGE / T2_HL;  // planes per chunk
    const uint32_t meta_l1 = get_write_ptr(CB_WS) + VIS_PAGE;
    const uint32_t stg_l1 = meta_l1 + META_PAGE;
    volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(meta_l1);
    volatile uint32_t* stg = reinterpret_cast<volatile uint32_t*>(stg_l1);
    for (uint32_t q = 0; q < n; q++) {
        const uint32_t j = get_arg_val<uint32_t>(3 + q);
        noc_async_read(get_noc_addr(j, HT), meta_l1, META_PAGE);
        noc_async_read_barrier();
        const uint32_t nk = meta[64];  // lanes sorted by count, descending
        cb_reserve_back(CB_HDR, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_HDR))[0] = nk;
        cb_push_back(CB_HDR, 1);
        cb_reserve_back(CB_WH, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_WH))[0] = j;
        cb_push_back(CB_WH, 1);
        for (uint32_t k0 = 0; k0 < nk; k0 += C) {
            for (uint32_t i = 0; i < T2_HL; i++) {
                const uint32_t c = meta[64 + i];
                if (c <= k0) continue;
                uint32_t sl = meta[i] + k0, left = (c - k0 < C ? c - k0 : C);
                uint32_t dst = stg_l1 + i * C * 64;
                while (left) {
                    const uint32_t o = sl % SLOTS_PER_PAGE;
                    const uint32_t m = left < SLOTS_PER_PAGE - o ? left : SLOTS_PER_PAGE - o;
                    noc_async_read(get_noc_addr(sl / SLOTS_PER_PAGE, SL, o * 64), dst, m * 64);
                    sl += m;
                    dst += m * 64;
                    left -= m;
                }
            }
            noc_async_read_barrier();
            const uint32_t k1 = nk - k0 < C ? nk : k0 + C;
            for (uint32_t k = k0; k < k1; k++) {
                cb_reserve_back(CB_K, 1);
                volatile uint32_t* t = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_K));
                for (uint32_t i = 0; i < T2_HL; i++) {
                    if (k < meta[64 + i]) {
                        const volatile uint32_t* src = stg + (i * C + (k - k0)) * SLOT_WORDS;
                        for (uint32_t s = 0; s < 9; s++) t[64 * s + i] = src[s];
                    } else {
                        for (uint32_t s = 0; s < 9; s++) t[64 * s + i] = 0;
                    }
                }
                // Lanes T2_HL.. are unused: zero them in both CB pages (the group's first two planes).
                if (k < 2)
                    for (uint32_t s = 0; s < 9; s++)
                        for (uint32_t i = T2_HL; i < 64; i++) t[64 * s + i] = 0;
                cb_push_back(CB_K, 1);
            }
        }
    }
}
#elif defined(T2_SPLIT)
void kernel_main() {
    const uint32_t p_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_arg_val<uint32_t>(2);
    const uint32_t s_addr = get_arg_val<uint32_t>(3);
    const uint32_t meta_addr = get_arg_val<uint32_t>(4);
    const uint32_t vis_addr = get_arg_val<uint32_t>(5);
    const uint32_t hs_addr = get_arg_val<uint32_t>(6);
    const uint32_t g0 = get_arg_val<uint32_t>(7);
    const uint32_t g1 = get_arg_val<uint32_t>(8);
    constexpr auto pa = TensorAccessorArgs<0>();
    constexpr auto ma = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ma.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto ta = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    constexpr auto hsa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
#ifdef T2_SH
    // #798: the t2sh pre pass aux page per group (arg 9, accessor 8), as in the one-program path.
    constexpr auto xa = TensorAccessorArgs<hsa.next_compile_time_args_offset()>();
    const auto AX = TensorAccessor(xa, get_arg_val<uint32_t>(9), P_PAGE);
#endif
    const auto P = TensorAccessor(pa, p_addr, P_PAGE);
    const auto M = TensorAccessor(ma, m_addr, MV_PAGE);
    const auto V = TensorAccessor(va, v_addr, MV_PAGE);
    const auto SL = TensorAccessor(sa, s_addr, SLOT_PAGE);
    const auto MT = TensorAccessor(ta, meta_addr, META_PAGE);
    const auto VI = TensorAccessor(ia, vis_addr, VIS_PAGE);
    const auto HS = TensorAccessor(hsa, hs_addr, SLOT_PAGE);
    constexpr uint32_t HFLAG = 0x80000000u;
    // Scratch (CB_WS): vis words, meta page, slot staging pages, then 64 heavy-sum records.
    const uint32_t vis_l1 = get_write_ptr(CB_WS);
    const uint32_t meta_l1 = vis_l1 + VIS_PAGE;
    const uint32_t stg_l1 = meta_l1 + META_PAGE;
    constexpr uint32_t HREC = T2_STAGE_PAGES * SLOT_PAGE / 4;  // word offset of the records
    volatile uint32_t* vis = reinterpret_cast<volatile uint32_t*>(vis_l1);
    volatile uint32_t* meta = reinterpret_cast<volatile uint32_t*>(meta_l1);
    volatile uint32_t* stg = reinterpret_cast<volatile uint32_t*>(stg_l1);
    uint32_t off[64], cnt[64];  // per lane: staged word offset of plane 0, planes it feeds
    uint32_t vb = 0xffffffffu;
    for (uint32_t g = g0; g < g1; g++) {
        if (g / GPB != vb) {
            vb = g / GPB;
            noc_async_read(get_noc_addr(vb, VI), vis_l1, VIS_PAGE);
            noc_async_read_barrier();
        }
        const uint32_t gi = g % GPB;
        if ((vis[2 * gi] | vis[2 * gi + 1]) == 0) continue;
        noc_async_read(get_noc_addr(g, MT), meta_l1, META_PAGE);
        cb_reserve_back(CB_P, 1);
        noc_async_read(get_noc_addr(g, P), get_write_ptr(CB_P), P_PAGE);
        cb_reserve_back(CB_M, 1);
        noc_async_read(get_noc_addr(g, M), get_write_ptr(CB_M), MV_PAGE);
        cb_reserve_back(CB_V, 1);
        noc_async_read(get_noc_addr(g, V), get_write_ptr(CB_V), MV_PAGE);
#ifdef T2_SH
        cb_reserve_back(CB_AUX, 1);
        noc_async_read(get_noc_addr(g, AX), get_write_ptr(CB_AUX), P_PAGE);
#endif
        noc_async_read_barrier();
        // Stage only the pages light lanes touch, in slot order (a lane's <= 2 pages land next to
        // each other, so plane k of a lane is at off + 16 k); heavy lanes read their record.
        uint32_t nk = 1, npg = 0, nh = 0, last = 0xffffffffu;
        bool staged = true;
        for (uint32_t i = 0; i < 64; i++) {
            const uint32_t c = meta[64 + i];
            if (c & HFLAG) {
                const uint32_t h = c & ~HFLAG;
                noc_async_read(get_noc_addr(h / SLOTS_PER_PAGE, HS, (h % SLOTS_PER_PAGE) * 64),
                               stg_l1 + (HREC + nh * SLOT_WORDS) * 4, 64);
                off[i] = HREC + nh * SLOT_WORDS;
                cnt[i] = 1;
                nh++;
                continue;
            }
            cnt[i] = c;
            if (c == 0) continue;
            nk = c > nk ? c : nk;
            const uint32_t a = meta[i] / SLOTS_PER_PAGE, b = (meta[i] + c - 1) / SLOTS_PER_PAGE;
            for (uint32_t q = a; q <= b; q++) {
                if (q == last) continue;
                if (npg == T2_STAGE_PAGES) staged = false;
                if (staged) noc_async_read(get_noc_addr(q, SL), stg_l1 + npg * SLOT_PAGE, SLOT_PAGE);
                last = q;
                npg++;
            }
            off[i] = ((npg - 1 - (b - a)) * SLOTS_PER_PAGE + meta[i] % SLOTS_PER_PAGE) * SLOT_WORDS;
        }
        cb_reserve_back(CB_HDR, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_HDR))[0] = nk;
        cb_push_back(CB_HDR, 1);
        cb_reserve_back(CB_WH, 1);
        reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_WH))[0] = g;
        cb_push_back(CB_WH, 1);
        cb_push_back(CB_P, 1);
        cb_push_back(CB_M, 1);
        cb_push_back(CB_V, 1);
#ifdef T2_SH
        cb_push_back(CB_AUX, 1);
#endif
        noc_async_read_barrier();
        for (uint32_t k = 0; k < nk; k++) {
            cb_reserve_back(CB_K, 1);
            volatile uint32_t* t = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_K));
            for (uint32_t i = 0; i < 64; i++) {
                if (k < cnt[i]) {
                    const volatile uint32_t* src;
                    if (staged || (meta[64 + i] & HFLAG)) {
                        src = stg + off[i] + k * SLOT_WORDS;
                    } else {  // more than T2_STAGE_PAGES pages: one 64 B read per slot
                        const uint32_t sl = meta[i] + k;
                        noc_async_read(get_noc_addr(sl / SLOTS_PER_PAGE, SL, (sl % SLOTS_PER_PAGE) * 64),
                                       stg_l1, 64);
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
#else
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
#ifdef T2_SH
    // #751: the t2sh pre pass aux page per group (arg 9, accessor 7), consumed after the reduce.
    constexpr auto xa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    const auto AX = TensorAccessor(xa, get_arg_val<uint32_t>(9), P_PAGE);
#endif
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
#ifdef T2_SH
            cb_reserve_back(CB_AUX, 1);
            noc_async_read(get_noc_addr(g, AX), get_write_ptr(CB_AUX), P_PAGE);
#endif
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
#ifdef T2_SH
            cb_push_back(CB_AUX, 1);
#endif
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
#endif
