// T0 blend forward reader (RISCV_1, #716). Feeds the production blend compute kernel
// (render/kernels/compute/alpha_blend_compute_mb.cpp) from the blend_forward_dev prep files
// with the same CB protocol as the production reader: tile count -> CB_CORE_TILES, ramps once
// -> CB_XRAMP/CB_YRAMP, then per tile subchunks of up to 2 * BULK_REC_SLOT records (#781: the
// hero has tiles of 25k): one counts page [num_g, flags] -> CB_MB_COUNTS (CONTINUE after the
// first subchunk, EMIT on the last), one BULK_REC_SLOT slot of PACK2 records -> CB_BUCKET_BULK
// and one slot of fp32 [op, r, g, b] (16 B a record, task #717) -> CB_TRAIN_OPC.
// The host puts each tile's opc list on a 64 B page and its first opc page in tile word 3.
// Runtime args: tiles, recs, ramps, opc DRAM addresses, first tile, tile count.
// Compile args: TensorAccessorArgs(tiles, recs, ramps, opc).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_XRAMP = 0, CB_YRAMP = 1, CB_MB_COUNTS = 3, CB_CORE_TILES = 7, CB_TRAIN_OPC = 9,
                   CB_BUCKET_BULK = 12;
constexpr uint32_t SMALL = 64, TILE_B = 4096, BULK_REC_SLOT = 4096, SUB_RECS = 2 * BULK_REC_SLOT;
constexpr uint32_t MB_FLAG_EMIT = 1u, MB_FLAG_CONTINUE = 2u, MB_FLAG_L1_BULK = 4u;
}  // namespace

void kernel_main() {
    const uint32_t tiles_addr = get_arg_val<uint32_t>(0);
    const uint32_t recs_addr = get_arg_val<uint32_t>(1);
    const uint32_t ramps_addr = get_arg_val<uint32_t>(2);
    const uint32_t opc_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t nt = get_arg_val<uint32_t>(5);

    constexpr auto tiles_args = TensorAccessorArgs<0>();
    constexpr auto recs_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    constexpr auto ramps_args = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
    const auto tiles = TensorAccessor(tiles_args, tiles_addr, SMALL);
    const auto recs = TensorAccessor(recs_args, recs_addr, SMALL);
    constexpr auto opc_args = TensorAccessorArgs<ramps_args.next_compile_time_args_offset()>();
    const auto ramps = TensorAccessor(ramps_args, ramps_addr, TILE_B);
    const auto opc = TensorAccessor(opc_args, opc_addr, SMALL);

    cb_reserve_back(CB_CORE_TILES, 1);
    reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_CORE_TILES))[0] = nt;
    cb_push_back(CB_CORE_TILES, 1);
    if (nt == 0) return;

    cb_reserve_back(CB_XRAMP, 1);
    cb_reserve_back(CB_YRAMP, 1);
    noc_async_read(get_noc_addr(0, ramps), get_write_ptr(CB_XRAMP), TILE_B);
    noc_async_read(get_noc_addr(1, ramps), get_write_ptr(CB_YRAMP), TILE_B);
    noc_async_read_barrier();
    cb_push_back(CB_XRAMP, 1);
    cb_push_back(CB_YRAMP, 1);

    for (uint32_t t = t0; t < t0 + nt; t++) {
        uint32_t hdr[4];
        {
            cb_reserve_back(CB_MB_COUNTS, 1);
            volatile uint32_t* cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
            noc_async_read(get_noc_addr(t, tiles), get_write_ptr(CB_MB_COUNTS), SMALL);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < 4; i++) hdr[i] = cnt[i];
        }
        const uint32_t rec_page = hdr[1], num_g = hdr[2], opc_page = hdr[3];
        uint32_t done = 0;
        do {
            const uint32_t n = num_g - done < SUB_RECS ? num_g - done : SUB_RECS;
            if (done != 0) cb_reserve_back(CB_MB_COUNTS, 1);
            volatile uint32_t* cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
            cnt[0] = n;
            cnt[1] = MB_FLAG_L1_BULK | (done != 0 ? MB_FLAG_CONTINUE : 0u) | (done + n == num_g ? MB_FLAG_EMIT : 0u);
            cb_push_back(CB_MB_COUNTS, 1);
            if (n == 0) break;

            cb_reserve_back(CB_BUCKET_BULK, BULK_REC_SLOT);
            cb_reserve_back(CB_TRAIN_OPC, BULK_REC_SLOT);
            const uint32_t l1 = get_write_ptr(CB_BUCKET_BULK), l1o = get_write_ptr(CB_TRAIN_OPC);
            const uint32_t rp = rec_page + done / 2, op = opc_page + done / 4;
            for (uint32_t p = 0; p < (n + 1) / 2; p++) {
                noc_async_read(get_noc_addr(rp + p, recs), l1 + p * SMALL, SMALL);
            }
            for (uint32_t p = 0; p < (n + 3) / 4; p++) {
                noc_async_read(get_noc_addr(op + p, opc), l1o + p * SMALL, SMALL);
            }
            noc_async_read_barrier();
            cb_push_back(CB_BUCKET_BULK, BULK_REC_SLOT);
            cb_push_back(CB_TRAIN_OPC, BULK_REC_SLOT);
            done += n;
        } while (done < num_g);
    }
}
