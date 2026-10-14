// T1 blend backward reader (RISCV_1, #693). Streams the blend_backward_dev prep files:
//   ramps once -> CB_RAMP (2 fp32 tiles); per tile: its 64 B header -> CB_HDR,
//   its 5 fp32 planes -> CB_PLANE, then one 64 B record page per pair -> CB_REC.
// Runtime args: tiles, recs, planes, ramps DRAM addresses, first tile, tile count.
// Compile args: TensorAccessorArgs(tiles, recs, planes, ramps).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t CB_HDR = 0, CB_PLANE = 1, CB_RAMP = 2, CB_REC = 3;
constexpr uint32_t SMALL = 64, TILE_B = 4096, N_PLANES = 5;
}  // namespace

void kernel_main() {
    const uint32_t tiles_addr = get_arg_val<uint32_t>(0);
    const uint32_t recs_addr = get_arg_val<uint32_t>(1);
    const uint32_t planes_addr = get_arg_val<uint32_t>(2);
    const uint32_t ramps_addr = get_arg_val<uint32_t>(3);
    const uint32_t t0 = get_arg_val<uint32_t>(4);
    const uint32_t nt = get_arg_val<uint32_t>(5);

    constexpr auto tiles_args = TensorAccessorArgs<0>();
    constexpr auto recs_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    constexpr auto planes_args = TensorAccessorArgs<recs_args.next_compile_time_args_offset()>();
    constexpr auto ramps_args = TensorAccessorArgs<planes_args.next_compile_time_args_offset()>();
    const auto tiles = TensorAccessor(tiles_args, tiles_addr, SMALL);
    const auto recs = TensorAccessor(recs_args, recs_addr, SMALL);
    const auto planes = TensorAccessor(planes_args, planes_addr, TILE_B);
    const auto ramps = TensorAccessor(ramps_args, ramps_addr, TILE_B);
    if (nt == 0) return;

    cb_reserve_back(CB_RAMP, 2);
    {
        const uint32_t l1 = get_write_ptr(CB_RAMP);
        noc_async_read(get_noc_addr(0, ramps), l1, TILE_B);
        noc_async_read(get_noc_addr(1, ramps), l1 + TILE_B, TILE_B);
        noc_async_read_barrier();
    }
    cb_push_back(CB_RAMP, 2);

    for (uint32_t t = t0; t < t0 + nt; t++) {
        cb_reserve_back(CB_HDR, 1);
        const uint32_t hl1 = get_write_ptr(CB_HDR);
        noc_async_read(get_noc_addr(t, tiles), hl1, SMALL);
        noc_async_read_barrier();
        const volatile uint32_t* h = reinterpret_cast<const volatile uint32_t*>(hl1);
        const uint32_t rec_start = h[1], cnt = h[2];
        cb_push_back(CB_HDR, 1);

        cb_reserve_back(CB_PLANE, N_PLANES);
        const uint32_t pl1 = get_write_ptr(CB_PLANE);
        for (uint32_t k = 0; k < N_PLANES; k++) {
            noc_async_read(get_noc_addr(t * N_PLANES + k, planes), pl1 + k * TILE_B, TILE_B);
        }
        noc_async_read_barrier();
        cb_push_back(CB_PLANE, N_PLANES);

        for (uint32_t r = rec_start; r < rec_start + cnt; r++) {
            cb_reserve_back(CB_REC, 1);
            noc_async_read(get_noc_addr(r, recs), get_write_ptr(CB_REC), SMALL);
            noc_async_read_barrier();
            cb_push_back(CB_REC, 1);
        }
    }
}
