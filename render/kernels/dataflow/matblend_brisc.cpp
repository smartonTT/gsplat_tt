// SPDX-License-Identifier: Apache-2.0
// Task #280/#285 (GSPLAT_TT_MATBLEND_FUSE=1): fused mat+blend program, BRISC.
// Mat mover 0 body (sort_subchunk_materialize.cpp, MAT_CB_BASE 16), then the
// blend writer body (writer_alpha_blend.cpp). See matblend_ncrisc.cpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "img_pack_u8.h"
#include "sort_bin_fp32.h"
#include "sort_radix_tile_algo.h"

#ifndef BLEND_RTA_BASE
#define BLEND_RTA_BASE 32
#endif

namespace matk {
#include "sort_subchunk_materialize.cpp"
}  // namespace matk

namespace blendk {
template <typename T>
inline T get_arg_val(int arg_idx) {
    return ::get_arg_val<T>(arg_idx + BLEND_RTA_BASE);
}
#include "writer_alpha_blend.cpp"
}  // namespace blendk

void kernel_main() {
    matk::kernel_main();
    noc_async_write_barrier();
    blendk::kernel_main();
}
