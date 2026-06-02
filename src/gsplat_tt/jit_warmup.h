#pragma once

namespace gsplat_tt {

// Compile/enqueue all device programs on the ideal TILE_BUCKET resident path
// once per process when GSPLAT_TT_JIT_WARMUP=1. Call at scene/view open before
// the first timed frame so frame-1 Tracy gaps are not dominated by JIT.
void jit_warmup_ideal_path();

}  // namespace gsplat_tt
