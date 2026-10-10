// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// pfwc WRITER SPLIT (task #207, GSPLAT_TT_PFWC_WRITER_SPLIT=1): the lever B
// fused writer (writer_pfwc_fuse.cpp) on both data-movement RISCs. Chunk k
// (tile chunk_start + k * stride) is classified and written by
//   WSPLIT_ROLE 0, BRISC / NoC0: even k, from compute output CBs 9..16, 35, 36;
//   WSPLIT_ROLE 1, NCRISC / NoC1: odd k, from the odd output set 41..50
//     (project_pfwc_compute.cpp PFWC_WSPLIT). NCRISC is also the reader
//     (reader_pfwc.cpp, PFWC_VIS): rd_poll() tops up the 10 input CBs between
//     its own steps, and every wait on NCRISC polls it, since compute may be
//     waiting on input. With PFWC_RD_COLS (task #232) BRISC reads a runtime
//     subset of the 10 input tiles per chunk (set arg 40 on the physical NoC0
//     columns x with bit x of arg 39 set, else set arg 41) and NCRISC the rest:
//     each RISC runs its own reader over its own input CBs, which moves that
//     share of the input traffic from NoC1 to NoC0.
// pfwc_wsplit.h hands (m, pr) and the shared page from chunk to chunk, so the
// DRAM bytes are writer_pfwc_fuse.cpp's (md5-identical).
//
// RUNTIME ARGS: 0..24 as writer_pfwc_fuse.cpp; 25..28 the mailbox slot
// semaphore ids (pfwc_wsplit::slot_index order); role 1 or PFWC_RD_COLS: 29..38
// the reader's DRAM bases (mcx, mcy, mcz, c00, c01, c02, c11, c12, c22, opacity);
// PFWC_RD_COLS: 39 the column mask, 40 / 41 BRISC's input tile set (bit o = tile o
// above) on / off those columns. PFWC_TILE_LIST (task #433 chunk cull): two more
// args after those, the DRAM bank-0 offset of this core's tile ids and their page
// bytes; chunk k is tile list[k] instead of chunk_start + k * stride (k keeps its
// parity, so the roles split the list as they split the strided deal).
// COMPILE-TIME ARGS: the writer's 9 TensorAccessorArgs; role 1 or PFWC_RD_COLS:
// + the reader's 10.
// Not supported: FUSE_ABL.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfwc_fuse.h"
#include "pfwc_wsplit.h"
#include "sort_bin_fp32.h"
#include "vis_tile.h"

#ifndef WSPLIT_ROLE
#define WSPLIT_ROLE 0
#endif
#if WSPLIT_ROLE == 1 || defined(PFWC_RD_COLS)
#define WS_READER 1  // this RISC may run the reader
#else
#define WS_READER 0
#endif

// GSPLAT_TT_PFWC_STEPCYC: wall cycles per part, recorded at the end (profiler
// builds) as "pfwc_ws", word i = (role * 16 + i) << 32 | value: "n wall wait cls
// pfx rec opn tail rd fl m" (n = chunks written, wait = the color reads and the 10
// compute tiles, pfx = PREFIX wait / send, opn = OPEN wait and head merge,
// rd = time in rd_poll, on the reader's RISC only, fl = time in the record and
// page flushes (write issue and noc_async_writes_flushed); rd and fl are included
// in the other parts; m = records written).
#ifdef PFWC_STEPCYC
#define WS_NOW() (reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L)[0])
#define WS_MARK(acc) do { const uint32_t n_ = WS_NOW(); acc += n_ - ws_t; ws_t = n_; } while (0)
#else
#define WS_MARK(acc) ((void)0)
#endif

void kernel_main() {
    constexpr uint32_t ROLE = WSPLIT_ROLE;
    using namespace pfwc_wsplit;
    uint32_t addr[9];
    for (uint32_t k = 0; k < 9; k++) addr[k] = get_arg_val<uint32_t>(k);
    [[maybe_unused]] const uint32_t chunk_start = get_arg_val<uint32_t>(9);
    const uint32_t num_chunks = get_arg_val<uint32_t>(10);
    [[maybe_unused]] const uint32_t stride = get_arg_val<uint32_t>(11);
    const uint32_t N = get_arg_val<uint32_t>(12);
    vis_tile::Params prm;
    prm.k_near = get_arg_val<uint32_t>(13);
    prm.min_opacity = get_arg_val<uint32_t>(14);
    prm.img_w = get_arg_val<uint32_t>(15);
    prm.img_h = get_arg_val<uint32_t>(16);
    prm.max_radius = get_arg_val<uint32_t>(17);
    prm.tiles_x = get_arg_val<uint32_t>(18);
    prm.tiles_y = get_arg_val<uint32_t>(19);
    const uint32_t tile_size = get_arg_val<uint32_t>(20);
    prm.tile_shift = dm_fp32::pow2_shift(tile_size);
    prm.inv_tile = 1.0f / static_cast<float>(tile_size);
    const uint32_t seg_base = get_arg_val<uint32_t>(21);
    const uint32_t core = get_arg_val<uint32_t>(22);
    const uint32_t pub01 = get_arg_val<uint32_t>(23);
    const uint32_t pub23 = get_arg_val<uint32_t>(24);
    uint32_t sem[NUM_SEMS];
    for (uint32_t i = 0; i < NUM_SEMS; i++) sem[i] = get_semaphore(get_arg_val<uint32_t>(25 + i));
#ifdef PFWC_TILE_LIST
    // Task #433: each role reads the list into its own half of CB 38 (no cross-RISC sync).
#ifdef PFWC_RD_COLS
    constexpr uint32_t TL_ARG = 42;
#else
    constexpr uint32_t TL_ARG = ROLE ? 39 : 29;
#endif
    constexpr uint32_t CB_TLIST = 38, TLIST_MAX_PAGE = 1024;
    const uint32_t tl_l1 = get_write_ptr(CB_TLIST) + ROLE * TLIST_MAX_PAGE;
    if (num_chunks != 0) {
        const InterleavedAddrGen<true> tl_gen{get_arg_val<uint32_t>(TL_ARG), get_arg_val<uint32_t>(TL_ARG + 1)};
        noc_async_read(get_noc_addr(0, tl_gen), tl_l1, tl_gen.page_size);  // single-page buffer: bank 0
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* tl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tl_l1);
    auto tile_of = [&](uint32_t k) -> uint32_t { return tl[k]; };
#else
    auto tile_of = [&](uint32_t k) -> uint32_t { return chunk_start + k * stride; };
#endif

    constexpr uint32_t LO = ROLE ? 32 : 0, HI = ROLE ? 14 : 0;
    constexpr uint32_t CB_M2X = 9 + LO, CB_M2Y = 10 + LO, CB_DEP = 11 + LO, CB_A = 12 + LO,
                       CB_B = 13 + LO, CB_C = 14 + LO, CB_RX = 15 + LO, CB_RY = 16 + LO,
                       CB_TPG = 35 + HI, CB_AABB = 36 + HI;
    constexpr uint32_t CB_STG = ROLE ? CB_STG_ODD : 40;  // STG_BYTES each
    constexpr uint32_t OUT_CB[10] = {CB_M2X, CB_M2Y, CB_DEP, CB_A,  CB_B,
                                     CB_C,   CB_RX,  CB_RY,  CB_TPG, CB_AABB};
    static_assert(odd_cb(9) == 41 && odd_cb(36) == 50, "odd output set");
    constexpr uint32_t PB = PW * 4;

    const uint32_t tile_bytes = get_tile_size(CB_M2X);

    constexpr auto a0 = TensorAccessorArgs<0>();
    constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
    constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
    constexpr auto a3 = TensorAccessorArgs<a2.next_compile_time_args_offset()>();
    constexpr auto a4 = TensorAccessorArgs<a3.next_compile_time_args_offset()>();
    constexpr auto a5 = TensorAccessorArgs<a4.next_compile_time_args_offset()>();
    constexpr auto a6 = TensorAccessorArgs<a5.next_compile_time_args_offset()>();
    constexpr auto a7 = TensorAccessorArgs<a6.next_compile_time_args_offset()>();
    constexpr auto a8 = TensorAccessorArgs<a7.next_compile_time_args_offset()>();

    const auto i_op = TensorAccessor(a0, addr[0], tile_bytes);
    const auto i_cr = TensorAccessor(a1, addr[1], tile_bytes);
    const auto i_cg = TensorAccessor(a2, addr[2], tile_bytes);
    const auto i_cb = TensorAccessor(a3, addr[3], tile_bytes);
    const auto i_q01 = TensorAccessor(a0, pub01, tile_bytes);
    const auto i_q23 = TensorAccessor(a0, pub23, tile_bytes);
    const auto o_dep = TensorAccessor(a4, addr[4], PB);
    const auto o_rec = TensorAccessor(a5, addr[5], PB);
    const auto o_offs = TensorAccessor(a6, addr[6], PB);
    const auto o_aabb = TensorAccessor(a7, addr[7], PB);
    const auto o_cnt = TensorAccessor(a8, addr[8], PB);

    // Staging (pfwc_wsplit::STG_BYTES).
    const uint32_t l1_op = (get_write_ptr(CB_STG) + 63u) & ~63u;
    const uint32_t l1_cr = l1_op + TILE_BYTES, l1_cg = l1_cr + TILE_BYTES, l1_cb = l1_cg + TILE_BYTES;
    const uint32_t l1_q01 = l1_cb + TILE_BYTES, l1_q23 = l1_q01 + TILE_BYTES;
    const uint32_t l1_rec = l1_q23 + TILE_BYTES;
    const uint32_t l1_pg = l1_rec + NB_MAX * RL * PB;  // dep, offs, aabb, counts, head dep, offs, aabb
    const uint32_t l1_mask = l1_pg + 7 * PB;
    const uint32_t l1_mbx = get_write_ptr(CB_MBX);
#if PFWC_REC32
    static_assert(EMIT_PUBOC, "PFWC_REC32 keeps the PUBOC words only");
    // Task #467: the head record page after the mask (STG_BYTES_REC32).
    auto w_hrec = reinterpret_cast<volatile uint32_t*>(l1_mask + 128);
#endif

    auto mw = reinterpret_cast<volatile uint32_t*>(l1_mask);
    auto opw = reinterpret_cast<volatile uint32_t*>(l1_op);
    auto p_cr = reinterpret_cast<volatile uint32_t*>(l1_cr);
    auto p_cg = reinterpret_cast<volatile uint32_t*>(l1_cg);
    auto p_cb = reinterpret_cast<volatile uint32_t*>(l1_cb);
    [[maybe_unused]] auto q01 = reinterpret_cast<volatile uint32_t*>(l1_q01);
    [[maybe_unused]] auto q23 = reinterpret_cast<volatile uint32_t*>(l1_q23);
    auto w_rec = reinterpret_cast<volatile uint32_t*>(l1_rec);
    auto pgw = [&](uint32_t i) { return reinterpret_cast<volatile uint32_t*>(l1_pg + i * PB); };
    auto w_cnt = pgw(3);
    auto msg_of = [&](uint32_t j) {
        return reinterpret_cast<volatile uint32_t*>(l1_mbx + slot_index(j) * MSG_WORDS * 4);
    };
    auto flag_of = [&](uint32_t j) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem[slot_index(j)]);
    };

    // Record words 9..15 are zero (gather_vis_scatter.cpp); EMIT_PUBOC sets 10..12.
    for (uint32_t s = 0; s < NB_MAX * RL; ++s)
        for (uint32_t w = 9; w < PW; ++w) w_rec[s * PW + w] = 0;

#ifdef PFWC_STEPCYC
    const uint32_t ws_w0 = WS_NOW();
    uint32_t ws_t = ws_w0, ws_wait = 0, ws_cls = 0, ws_pfx = 0, ws_rec = 0, ws_opn = 0, ws_tail = 0,
             ws_rd = 0, ws_fl = 0, ws_m = 0, ws_n = 0;
#define WS_FL_BEGIN() const uint32_t f_t0 = WS_NOW()
#define WS_FL_END() ws_fl += WS_NOW() - f_t0
#else
#define WS_FL_BEGIN() ((void)0)
#define WS_FL_END() ((void)0)
#endif

#if WS_READER
    // The reader (reader_pfwc.cpp with PFWC_VIS, stride deal), non-blocking:
    // reads chunk rd_k once its input CBs (rd_set) have room, pushes it once landed.
#ifdef PFWC_RD_COLS
    // NOC_NODE_ID is the physical NoC0 position (my_x is the translated one).
    const uint32_t x0 = static_cast<uint32_t>(NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID) & NOC_NODE_ID_MASK);
    const uint32_t b_set = get_arg_val<uint32_t>(((get_arg_val<uint32_t>(39) >> x0) & 1u) ? 40 : 41);
    const uint32_t rd_set = (ROLE == 0 ? b_set : ~b_set) & 0x3FFu;
    const bool rd_here = rd_set != 0;
#else
    constexpr uint32_t rd_set = 0x3FFu;
    constexpr bool rd_here = true;
#endif
    auto rd_has = [&](uint32_t o) { return ((rd_set >> o) & 1u) != 0; };
    const uint32_t in_bytes = get_tile_size(0);
    constexpr auto r0 = TensorAccessorArgs<a8.next_compile_time_args_offset()>();
    constexpr auto r1 = TensorAccessorArgs<r0.next_compile_time_args_offset()>();
    constexpr auto r2 = TensorAccessorArgs<r1.next_compile_time_args_offset()>();
    constexpr auto r3 = TensorAccessorArgs<r2.next_compile_time_args_offset()>();
    constexpr auto r4 = TensorAccessorArgs<r3.next_compile_time_args_offset()>();
    constexpr auto r5 = TensorAccessorArgs<r4.next_compile_time_args_offset()>();
    constexpr auto r6 = TensorAccessorArgs<r5.next_compile_time_args_offset()>();
    constexpr auto r7 = TensorAccessorArgs<r6.next_compile_time_args_offset()>();
    constexpr auto r8 = TensorAccessorArgs<r7.next_compile_time_args_offset()>();
    constexpr auto r9 = TensorAccessorArgs<r8.next_compile_time_args_offset()>();
    const auto in0 = TensorAccessor(r0, get_arg_val<uint32_t>(29), in_bytes);
    const auto in1 = TensorAccessor(r1, get_arg_val<uint32_t>(30), in_bytes);
    const auto in2 = TensorAccessor(r2, get_arg_val<uint32_t>(31), in_bytes);
    const auto in3 = TensorAccessor(r3, get_arg_val<uint32_t>(32), in_bytes);
    const auto in4 = TensorAccessor(r4, get_arg_val<uint32_t>(33), in_bytes);
    const auto in5 = TensorAccessor(r5, get_arg_val<uint32_t>(34), in_bytes);
    const auto in6 = TensorAccessor(r6, get_arg_val<uint32_t>(35), in_bytes);
    const auto in7 = TensorAccessor(r7, get_arg_val<uint32_t>(36), in_bytes);
    const auto in8 = TensorAccessor(r8, get_arg_val<uint32_t>(37), in_bytes);
    const auto in9 = TensorAccessor(r9, get_arg_val<uint32_t>(38), in_bytes);
    constexpr uint32_t IN_CB[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 30};
    uint32_t rd_k = 0;
    bool rd_pend = false;
    auto rd_step = [&]() {
        invalidate_l1_cache();  // callers may spin on CB state across early returns
        if (rd_pend) {
            if (!ncrisc_noc_reads_flushed(noc_index)) return;
            for (uint32_t o = 0; o < 10; o++)
                if (rd_has(o)) cb_push_back(IN_CB[o], 1);
            rd_pend = false;
            rd_k++;
        }
        if (rd_k >= num_chunks) return;
        invalidate_l1_cache();  // compute's tiles_acked (as cb_reserve_back)
        for (uint32_t o = 0; o < 10; o++)
            if (rd_has(o) && !cb_pages_reservable_at_back(IN_CB[o], 1)) return;
        for (uint32_t o = 0; o < 10; o++)
            if (rd_has(o)) cb_reserve_back(IN_CB[o], 1);
        const uint32_t t = tile_of(rd_k);
        if (rd_has(0)) noc_async_read_tile(t, in0, get_write_ptr(IN_CB[0]));
        if (rd_has(1)) noc_async_read_tile(t, in1, get_write_ptr(IN_CB[1]));
        if (rd_has(2)) noc_async_read_tile(t, in2, get_write_ptr(IN_CB[2]));
        if (rd_has(3)) noc_async_read_tile(t, in3, get_write_ptr(IN_CB[3]));
        if (rd_has(4)) noc_async_read_tile(t, in4, get_write_ptr(IN_CB[4]));
        if (rd_has(5)) noc_async_read_tile(t, in5, get_write_ptr(IN_CB[5]));
        if (rd_has(6)) noc_async_read_tile(t, in6, get_write_ptr(IN_CB[6]));
        if (rd_has(7)) noc_async_read_tile(t, in7, get_write_ptr(IN_CB[7]));
        if (rd_has(8)) noc_async_read_tile(t, in8, get_write_ptr(IN_CB[8]));
        if (rd_has(9)) noc_async_read_tile(t, in9, get_write_ptr(IN_CB[9]));
        rd_pend = true;
    };
    auto rd_poll = [&]() {
#ifdef PFWC_STEPCYC
        const uint32_t r_t0 = WS_NOW();
        rd_step();
        ws_rd += WS_NOW() - r_t0;
#else
        rd_step();
#endif
    };
#define WS_POLL() do { if (rd_here) rd_poll(); } while (0)
#else
#define WS_POLL() ((void)0)
#endif
    // Spin until ready(), polling the reader (on its RISC).
    auto wait_until = [&](auto&& ready) {
        do {
            invalidate_l1_cache();
            WS_POLL();
        } while (!ready());
        invalidate_l1_cache();
    };

    // Records are one 64 B page each and page g sits on DRAM bank g % nb (see
    // writer_pfwc_fuse.cpp); nb is read off the accessor, else per page.
    uint32_t nb = 1;
    {
        const uint64_t a = get_noc_addr(seg_base, o_rec);
        while (nb <= NB_MAX && get_noc_addr(seg_base + nb, o_rec) != a + PB) nb++;
    }
    const bool per_page = nb > NB_MAX;
    if (per_page) nb = NB_MAX;  // staging layout only
#if PFWC_REC32
    Rec32Stage rs;  // pages g / 2 (seg_base is even); bank count as above
#else
    RecStage rs;
#endif
    rs.init(nb);
    // Staged records [gs, ge) of group G0, one write per bank.
    auto flush_rec = [&](uint32_t G0, uint32_t gs, uint32_t ge) {
        WS_FL_BEGIN();
        rs.writes(G0, gs, ge, per_page, [&](uint32_t s, uint32_t g, uint32_t n) {
            noc_async_write(l1_rec + s * PB, get_noc_addr(g, o_rec), n * PB);
        });
        noc_async_writes_flushed();  // staging reusable; completion at the end
        WS_FL_END();
    };
    auto l1a = [](volatile uint32_t* p) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)); };
    auto flush_pg = [&](volatile uint32_t* d, volatile uint32_t* o, volatile uint32_t* a, uint32_t page) {
        WS_FL_BEGIN();
        noc_async_write(l1a(d), get_noc_addr(page, o_dep), PB);
        noc_async_write(l1a(o), get_noc_addr(page, o_offs), PB);
        noc_async_write(l1a(a), get_noc_addr(page, o_aabb), PB);
        noc_async_writes_flushed();
        WS_FL_END();
    };
    auto write_counts = [&](uint32_t m, uint32_t pr) {
        for (uint32_t w = 0; w < PW; w++) w_cnt[w] = 0;
        w_cnt[pfwc_fuse::T_M] = m;
        w_cnt[pfwc_fuse::T_P] = pr;
        noc_async_write(l1a(w_cnt), get_noc_addr(core, o_cnt), PB);
    };
    ChunkPages pg;
    pg.sd = pgw(0);
    pg.so = pgw(1);
    pg.sa = pgw(2);
    pg.hd = pgw(4);
    pg.ho = pgw(5);
    pg.ha = pgw(6);

    for (uint32_t k = ROLE; k < num_chunks; k += 2) {
        const uint32_t t = tile_of(k);
        noc_async_read_tile(t, i_op, l1_op);
        noc_async_read_tile(t, i_cr, l1_cr);
        noc_async_read_tile(t, i_cg, l1_cg);
        noc_async_read_tile(t, i_cb, l1_cb);
#if EMIT_PUBOC
        if (pub01 != 0) {
            noc_async_read_tile(t, i_q01, l1_q01);
            noc_async_read_tile(t, i_q23, l1_q23);
        }
#endif
#if WS_READER
        if (rd_here)
            for (uint32_t o = 0; o < 10; o++)
                while (!cb_pages_available_at_front(OUT_CB[o], 1)) {  // compute may wait on input
                    invalidate_l1_cache();
                    rd_poll();
                }
#endif
        for (uint32_t o = 0; o < 10; o++) cb_wait_front(OUT_CB[o], 1);
        noc_async_read_barrier();
        WS_MARK(ws_wait);

        auto p_m2x = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2X));
        auto p_m2y = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_M2Y));
        auto p_dep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_DEP));
        auto p_a = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_A));
        auto p_b = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_B));
        auto p_c = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_C));
        auto p_rx = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RX));
        auto p_ry = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_RY));
        auto p_tpg = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_TPG));
        auto p_aabb = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_AABB));

        const uint32_t tbase = t * vis_tile::TILE_ELEMS;
        const uint32_t n_el = (tbase >= N) ? 0u
            : (N - tbase < vis_tile::TILE_ELEMS ? N - tbase : vis_tile::TILE_ELEMS);
        auto get_inputs = [&](uint32_t il, uint32_t& tz, uint32_t& op, uint32_t& mx,
                              uint32_t& my, uint32_t& rx, uint32_t& ry) {
            tz = p_dep[il];
            op = opw[il];
            mx = p_m2x[il];
            my = p_m2y[il];
            rx = p_rx[il];
            ry = p_ry[il];
        };
        uint32_t vc = 0, pc = 0;
        vis_tile::classify_tile(p_tpg, p_aabb, n_el, mw, prm, get_inputs, &vc, &pc);
#if EMIT_PUBOC
        if (pub01 == 0)  // NaN scene: pack the visible lanes here
            for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++)
                for (uint32_t bits = mw[w]; bits != 0; bits &= bits - 1) {
                    const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                    q01[il] = sort_bin_fp32::to_unorm16(opw[il]) |
                              (sort_bin_fp32::to_unorm16(p_cr[il]) << 16);
                    q23[il] = sort_bin_fp32::to_unorm16(p_cg[il]) |
                              (sort_bin_fp32::to_unorm16(p_cb[il]) << 16);
                }
#endif
        WS_POLL();
        WS_MARK(ws_cls);

        // PREFIX: chunk k's (m, pr) from chunk k - 1's writer; chunk k + 1's out.
        uint32_t m0 = 0, pr = 0;
        if (k != 0) {
            volatile tt_l1_ptr uint32_t* f = flag_of(k);
            wait_until([&] { return *f != F_EMPTY; });
            const volatile uint32_t* msg = msg_of(k);
            m0 = msg[MSG_M];
            pr = msg[MSG_PR];
            asm volatile("fence" ::: "memory");  // message loads before EMPTY
            if (m0 % PW == 0) noc_semaphore_set(f, F_EMPTY);  // no OPEN follows
        }
        if (k + 1 < num_chunks) {
            volatile tt_l1_ptr uint32_t* f = flag_of(k + 1);
            wait_until([&] { return *f == F_EMPTY; });
            volatile uint32_t* msg = msg_of(k + 1);
            msg[MSG_M] = m0 + vc;
            msg[MSG_PR] = pr + pc;
            asm volatile("fence" ::: "memory");
            noc_semaphore_set(f, F_PREFIX);
        }
        WS_MARK(ws_pfx);

        pg.begin(m0, seg_base / PW);
        rs.begin(seg_base + m0);
        for (uint32_t w = 0; w < vis_tile::MASK_WORDS; w++) {
#if WS_READER
            if (rd_here && (w & 7u) == 7u) rd_poll();
#endif
            for (uint32_t bits = mw[w]; bits != 0; bits &= bits - 1) {
                const uint32_t il = w * 32 + static_cast<uint32_t>(__builtin_ctz(bits));
                // Loads grouped ahead of their stores (writer_pfwc_fuse.cpp).
                const uint32_t dep = p_dep[il], aabb = p_aabb[il], tpg = p_tpg[il];
                pg.put(dep, pr, aabb & vis_tile::PAYLOAD, flush_pg);
                pr += tpg & vis_tile::PAYLOAD;
#if PFWC_REC32
                volatile uint32_t* r = rs.in_head() ? w_hrec + RW : w_rec + rs.stage_word();
#else
                volatile uint32_t* r = w_rec + rs.slot() * PW;
#endif
                {
                    const uint32_t a = p_a[il], b = p_b[il], cc = p_c[il];
                    const uint32_t mx = p_m2x[il], my = p_m2y[il];
                    r[0] = a;
                    r[1] = b;
                    r[2] = cc;
                    r[3] = mx;
                    r[4] = my;
                }
                {
#if PFWC_REC32
                    // Task #467: the emit's words only (no fp32 op / colour).
                    const uint32_t u01 = q01[il], u23 = q23[il];
                    r[5] = u01;
                    r[6] = u23;
                    r[7] = dep;
#else
                    const uint32_t op = opw[il], cr = p_cr[il], cg = p_cg[il], cb = p_cb[il];
#if EMIT_PUBOC
                    const uint32_t u01 = q01[il], u23 = q23[il];
#endif
                    r[5] = op;
                    r[6] = cr;
                    r[7] = cg;
                    r[8] = cb;
#if EMIT_PUBOC
                    r[10] = u01;
                    r[11] = u23;
                    r[12] = dep;
#endif
#endif  // PFWC_REC32
                }
                rs.next(flush_rec);
            }
        }
        rs.end(flush_rec);
        WS_MARK(ws_rec);

        // OPEN: the head page's first words from chunk k - 1's writer.
        if (pg.head_s != 0) {
            volatile tt_l1_ptr uint32_t* f = flag_of(k);
            wait_until([&] { return *f == F_OPEN; });
            pg.finish(msg_of(k), flush_pg);
#if PFWC_REC32
            if (rs.head) {
                const volatile uint32_t* msg = msg_of(k);
                for (uint32_t i = 0; i < RW; i++) w_hrec[i] = msg[MSG_REC + i];
                if (rs.head_used) {
                    WS_FL_BEGIN();
                    noc_async_write(l1a(w_hrec), get_noc_addr(rs.head_page, o_rec), PB);
                    noc_async_writes_flushed();
                    WS_FL_END();
                }
            }
#endif
            asm volatile("fence" ::: "memory");  // message loads before EMPTY
            noc_semaphore_set(f, F_EMPTY);
        }
        WS_MARK(ws_opn);
        if (k + 1 < num_chunks) {
            if (pg.slot != 0) {
                pg.export_open(msg_of(k + 1));
#if PFWC_REC32
                // m_{k+1} odd: the open half page's record (staged, or the
                // head's lower half passed through an empty chunk).
                volatile uint32_t* msg = msg_of(k + 1);
                const volatile uint32_t* src = rs.half ? w_rec + rs.open_word() : w_hrec;
                if (rs.half || rs.in_head())
                    for (uint32_t i = 0; i < RW; i++) msg[MSG_REC + i] = src[i];
#endif
                asm volatile("fence" ::: "memory");
                noc_semaphore_set(flag_of(k + 1), F_OPEN);
            }
        } else {
            pg.close(pr, flush_pg);
#if PFWC_REC32
            // The core's last page, when it holds one record: upper half 0.
            if (rs.half || rs.in_head()) {
                volatile uint32_t* p = rs.half ? w_rec + rs.open_word() : w_hrec;
                for (uint32_t i = RW; i < PW; i++) p[i] = 0;
                WS_FL_BEGIN();
                noc_async_write(l1a(p), get_noc_addr(rs.half ? rs.open_page() : rs.head_page, o_rec), PB);
                noc_async_writes_flushed();
                WS_FL_END();
            }
#endif
            write_counts(m0 + vc, pr);
        }
        for (uint32_t o = 0; o < 10; o++) cb_pop_front(OUT_CB[o], 1);
#ifdef PFWC_STEPCYC
        ws_n++;
        ws_m += vc;
#endif
        WS_MARK(ws_tail);
    }
    if (ROLE == 0 && num_chunks == 0) write_counts(0, 0);
#if WS_READER
    if (rd_here)
        while (rd_k < num_chunks) rd_poll();
#endif
    noc_async_write_barrier();
#ifdef PFWC_STEPCYC
    WS_MARK(ws_tail);
    const uint32_t ws_v[11] = {ws_n, WS_NOW() - ws_w0, ws_wait, ws_cls, ws_pfx, ws_rec, ws_opn, ws_tail, ws_rd, ws_fl,
                               ws_m};
    for (uint32_t i = 0; i < 11; i++) DeviceTimestampedData("pfwc_ws", (uint64_t(ROLE * 16 + i) << 32) | ws_v[i]);
#endif
}
