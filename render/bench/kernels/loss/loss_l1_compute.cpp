// Task #690 (T7) L1 loss step, compute. Per fp32 tile (x render, y GT):
//   dL/dx = w * sign(x - y)  (w = 0.8 / (3 H W); sign(0) = 0, torch's abs grad)
//   acc  += |x - y|          (per-lane partial sums; the host adds the per-core tiles)
// Padding pixels are zero in x and y, so they add 0 to both. Runtime args: n_tiles, w (fp32 bits).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif

constexpr uint32_t CB_X = 0, CB_Y = 1, CB_ACC = 3, CB_G = 16, CB_PART = 17;

#ifdef TRISC_MATH
// DEST tiles: 0 = x -> grad, 1 = y, 2 = acc. 32 vectors of 32 lanes per fp32 tile.
inline void l1_tile(uint32_t w_bits) {
    using namespace sfpi;
    const vFloat w = __builtin_bit_cast(float, w_bits);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        const vFloat x = dst_reg[0];
        const vFloat y = dst_reg[32];
        const vFloat d = x - y;
        vFloat g = 0.0f;
        v_if (d > 0.0f) { g = w; }
        v_elseif (d < 0.0f) { g = -w; }
        v_endif;
        dst_reg[0] = g;
        const vFloat acc = dst_reg[64];
        dst_reg[64] = acc + sfpi::abs(d);
        dst_reg++;
    }
}
#endif

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t w_bits = get_arg_val<uint32_t>(1);
    init_sfpu(CB_X, CB_G);
    for (uint32_t t = 0; t < n; t++) {
        cb_wait_front(CB_X, 1);
        cb_wait_front(CB_Y, 1);
        cb_wait_front(CB_ACC, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_X);
        copy_tile(CB_X, 0, 0);
        copy_tile(CB_Y, 0, 1);
        copy_tile(CB_ACC, 0, 2);
        MATH((_llk_math_eltwise_unary_sfpu_start_(0)));
        MATH((l1_tile(w_bits)));
        MATH((_llk_math_eltwise_unary_sfpu_done_()));
        tile_regs_commit();
        cb_pop_front(CB_X, 1);
        cb_pop_front(CB_Y, 1);
        cb_pop_front(CB_ACC, 1);
        tile_regs_wait();
        cb_reserve_back(CB_G, 1);
        pack_tile(0, CB_G);
        cb_push_back(CB_G, 1);
        const uint32_t cb_acc_out = t + 1 == n ? CB_PART : CB_ACC;
        cb_reserve_back(cb_acc_out, 1);
        pack_tile(2, cb_acc_out);
        cb_push_back(cb_acc_out, 1);
        tile_regs_release();
    }
    if (n == 0) {  // keep the writer's partial-tile wait satisfied: pass the zero tile through
        cb_wait_front(CB_ACC, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(CB_ACC);
        copy_tile(CB_ACC, 0, 0);
        tile_regs_commit();
        cb_pop_front(CB_ACC, 1);
        tile_regs_wait();
        cb_reserve_back(CB_PART, 1);
        pack_tile(0, CB_PART);
        cb_push_back(CB_PART, 1);
        tile_regs_release();
    }
}
