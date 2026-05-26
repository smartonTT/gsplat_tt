# iter-007 — Stage D1 fuse: FPU `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>`

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-006 d1-fuse-contrib (REJECT — perf regression)

## Hypothesis

iter-006 collapsed the two D1 acquires (alpha·T_state then ·sat_mask) into one
using SFPU `mul_binary_tile` for the second multiply. Bit-identical (PSNR ∞,
diff10 black) but kernel ms regressed 99.95 → 100.69. The SFPU DST-DST mul is
slower than the acquire+CB_T_TMP roundtrip it saves.

iter-007 keeps the single-acquire structure but replaces the SFPU mul with
the FPU `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>` path:

```
acquire {
  mul_tiles(CB_ALPHA, CB_T_STATE) → dst[0];                       // FPU mul
  binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK);
  binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK, 0, 0);// FPU dst[0] = SRCA(=dst[0]) * SRCB(sat_mask)
  pack dst[0] → CB_CONTRIB
} release
```

Both ops are FPU; we still save the acquire/commit/wait/release cycle and the
CB_T_TMP reserve/pack/push/wait/pop roundtrip. Only difference from iter-006
is the second multiply uses the FPU unpack-A path (loading dst[0] back into
SRCA) instead of an SFPU scalar-loop mul.

## Validation gate

- PSNR ≥ 40 dB (perceptual floor) — expect ∞ since algebra is unchanged.
- Visual checks at perceptual 1× must pass (no tile seams).
- Kernel ms must improve vs prev_best (99.95 ms). Any improvement counts.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Removed iter-006's `#include "api/compute/eltwise_binary_sfpu.h"` (no
    longer needed once `mul_binary_tile` is gone).
  - Replaced the SFPU `copy_tile`+`mul_binary_tile` fragment with the FPU
    `binary_dest_reuse_tiles_init` / `binary_dest_reuse_tiles` pair.

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-006 state; one more revert to 044f398 restores the baseline.
