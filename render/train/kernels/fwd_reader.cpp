// T0 blend forward reader (RISCV_1, #716). Feeds the production blend compute kernel
// (render/kernels/compute/alpha_blend_compute_mb.cpp) from the blend_forward_dev prep files
// with the same CB protocol as the production reader: tile count -> CB_CORE_TILES, ramps once
// -> CB_XRAMP/CB_YRAMP, per tile one counts page [num_g, EMIT | L1_BULK] -> CB_MB_COUNTS and
// one BULK_REC_SLOT slot of PACK2 records -> CB_BUCKET_BULK (ceil(num_g / 2) pages filled;
// the host refuses tiles over 2 * BULK_REC_SLOT records).
// Runtime args: tiles, recs, ramps DRAM addresses, first tile, tile count.
// Compile args: TensorAccessorArgs(tiles, recs, ramps).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_XRAMP = 0, CB_YRAMP = 1, CB_MB_COUNTS = 3, CB_CORE_TILES = 7, CB_BUCKET_BULK = 12;
constexpr uint32_t SMALL = 64, TILE_B = 4096, BULK_REC_SLOT = 4096;
constexpr uint32_t MB_FLAG_EMIT = 1u, MB_FLAG_L1_BULK = 4u;
}  // namespace

void kernel_main() {
    const uint32_t tiles_addr = get_arg_val<uint32_t>(0);
    const uint32_t recs_addr = get_arg_val<uint32_t>(1);
    const uint32_t ramps_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t nt = get_arg_val<uint32_t>(4);

    constexpr auto tiles_args = TensorAccessorArgs<0>();
    constexpr auto recs_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    constexpr auto ramps_args = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
    const auto tiles = TensorAccessor(tiles_args, tiles_addr, SMALL);
    const auto recs = TensorAccessor(recs_args, recs_addr, SMALL);
    const auto ramps = TensorAccessor(ramps_args, ramps_addr, TILE_B);

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
        cb_reserve_back(CB_MB_COUNTS, 1);
        volatile uint32_t* cnt = reinterpret_cast<volatile uint32_t*>(get_write_ptr(CB_MB_COUNTS));
        noc_async_read(get_noc_addr(t, tiles), get_write_ptr(CB_MB_COUNTS), SMALL);
        noc_async_read_barrier();
        const uint32_t rec_page = cnt[1], num_g = cnt[2];
        cnt[0] = num_g;
        cnt[1] = MB_FLAG_EMIT | MB_FLAG_L1_BULK;
        cb_push_back(CB_MB_COUNTS, 1);
        if (num_g == 0) continue;

        cb_reserve_back(CB_BUCKET_BULK, BULK_REC_SLOT);
        const uint32_t l1 = get_write_ptr(CB_BUCKET_BULK);
        const uint32_t pages = (num_g + 1) / 2;
        for (uint32_t p = 0; p < pages; p++) {
            noc_async_read(get_noc_addr(rec_page + p, recs), l1 + p * SMALL, SMALL);
        }
        noc_async_read_barrier();
        cb_push_back(CB_BUCKET_BULK, BULK_REC_SLOT);
    }
}
