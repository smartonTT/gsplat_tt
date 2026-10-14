// T1 blend backward writer (RISCV_0, #693; v2 #704). One packed fp32 ACC tile per pack of up
// to NB pairs of a tile (blend_bwd_compute.cpp: pair i -> column i % 8 of vectors
// 24 + 3 (i / 8) + {0, 1, 2}, already lane-reduced and finished). Per pair it gathers the 9
// floats into a 64 B staging page and writes it to DRAM slot page rec[13] (P[g] + j).
// Runtime args: out, tiles, recs DRAM addresses, first tile, tile count[, stats address, core].
// Compile args: TensorAccessorArgs(out, tiles, recs[, stats]).
// BWD_STATS (#707): after the last tile one more CB_OUT tile holds the compute's counters
// (vector 2k / 2k + 1 = low / high 16 bits of counter k, all lanes); lane 0 of the first
// ST_WORDS vectors goes to 128 B page `core` of the stats buffer.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_OUT = 16, CB_WS = 17, SMALL = 64, NB = 16, REC_WORDS = 16;
// Packed-tile element of SFPU vector V, row j, column k (blend_backward_dev.vector_lanes).
constexpr uint32_t lane(uint32_t V, uint32_t j, uint32_t k) { return 64 * (V >> 1) + 16 * j + 2 * k + (V & 1); }
constexpr uint32_t ST_PAGE = 128, ST_WORDS = 32;
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t tiles_addr = get_arg_val<uint32_t>(1);
    const uint32_t recs_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto tiles_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto recs_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr, SMALL);
    const auto tiles = TensorAccessor(tiles_args, tiles_addr, SMALL);
    const auto recs = TensorAccessor(recs_args, recs_addr, SMALL);
    // Scratch (CB_WS, never pushed): tile header, NB records, NB staging pages.
    const uint32_t hdr_l1 = get_write_ptr(CB_WS);
    const uint32_t rec_l1 = hdr_l1 + SMALL;
    const uint32_t stg_l1 = rec_l1 + NB * SMALL;
    volatile uint32_t* hdr = reinterpret_cast<volatile uint32_t*>(hdr_l1);
    volatile uint32_t* rec = reinterpret_cast<volatile uint32_t*>(rec_l1);
    volatile uint32_t* stg = reinterpret_cast<volatile uint32_t*>(stg_l1);

    for (uint32_t t = t0; t < t0 + nt; t++) {
        noc_async_read(get_noc_addr(t, tiles), hdr_l1, SMALL);
        noc_async_read_barrier();
        const uint32_t rec_start = hdr[1], cnt = hdr[2];
        for (uint32_t b0 = 0; b0 < cnt; b0 += NB) {
            const uint32_t n = cnt - b0 < NB ? cnt - b0 : NB;
            for (uint32_t k = 0; k < n; k++) {
                noc_async_read(get_noc_addr(rec_start + b0 + k, recs), rec_l1 + k * SMALL, SMALL);
            }
            noc_async_read_barrier();
            noc_async_writes_flushed();  // staging pages of the previous pack have left L1
            cb_wait_front(CB_OUT, 1);
            const volatile uint32_t* o = reinterpret_cast<const volatile uint32_t*>(get_read_ptr(CB_OUT));
            for (uint32_t k = 0; k < n; k++) {
                const uint32_t V = 24 + 3 * (k >> 3), q = k & 7u;
                volatile uint32_t* d = stg + k * REC_WORDS;
                d[0] = o[lane(V + 1, 3, q)];  // dmx
                d[1] = o[lane(V + 2, 0, q)];  // dmy
                d[2] = o[lane(V + 1, 0, q)];  // dA
                d[3] = o[lane(V + 1, 1, q)];  // dB
                d[4] = o[lane(V + 1, 2, q)];  // dC
                d[5] = o[lane(V, 0, q)];      // dop
                d[6] = o[lane(V, 1, q)];      // dr
                d[7] = o[lane(V, 2, q)];      // dg
                d[8] = o[lane(V, 3, q)];      // db
                noc_async_write(stg_l1 + k * SMALL, get_noc_addr(rec[k * REC_WORDS + 13], out), SMALL);
            }
            cb_pop_front(CB_OUT, 1);
        }
    }
#ifdef BWD_STATS
    if (nt != 0) {
        constexpr auto st_args = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
        const auto st = TensorAccessor(st_args, get_arg_val<uint32_t>(5), ST_PAGE);
        noc_async_writes_flushed();
        cb_wait_front(CB_OUT, 1);
        const volatile uint32_t* o = reinterpret_cast<const volatile uint32_t*>(get_read_ptr(CB_OUT));
        for (uint32_t v = 0; v < ST_WORDS; v++) stg[v] = o[lane(v, 0, 0)];
        noc_async_write(stg_l1, get_noc_addr(get_arg_val<uint32_t>(6), st), ST_PAGE);
        cb_pop_front(CB_OUT, 1);
    }
#endif
    noc_async_write_barrier();
}
