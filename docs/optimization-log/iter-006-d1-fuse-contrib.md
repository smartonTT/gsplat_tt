# iter-006 — Stage D1 fuse: contrib = alpha · T_state · sat_mask in single acquire

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-005 verify-revert-baseline (99.95 ms, PSNR ∞)

## Hypothesis

The baseline Stage D1 computes `contrib = alpha · T_state · sat_mask` in 2 separate acquire blocks (via the scratch CB_T_TMP):

```
acquire { mul_tiles(CB_ALPHA, CB_T_STATE) → dst[0]; pack dst[0] → CB_T_TMP } release
acquire { mul_tiles(CB_T_TMP, CB_SAT_MASK) → dst[0]; pack dst[0] → CB_CONTRIB } release
```

Fuse to a single acquire using FPU + SFPU DST-DST:

```
acquire {
  mul_tiles(CB_ALPHA, CB_T_STATE) → dst[0];                // FPU mul
  copy_tile(CB_SAT_MASK) → dst[1];                          // load sat_mask
  mul_binary_tile(0, 1, 0);                                 // SFPU dst[0] *= dst[1]
  pack dst[0] → CB_CONTRIB
} release
```

This saves:
- 1 tile_regs_acquire/commit/wait/release cycle
- 1 cb_reserve_back / pack_tile / cb_push_back / cb_wait_front / cb_pop_front cycle on CB_T_TMP for the D1 portion

CB_T_TMP itself is still used by D2 and E for now, so it stays allocated.

## Why this is different from iter-004

iter-004 attempted D1+D2+E fusion AND ran on top of iter-003 M1 basis-form. The basis-form was the source of the visible tile-grid artifact; iter-004 inherited it AND added SFPU DST-DST ops that turned out to be ~2 ms slower than the basis-form FPU mul_tiles approach.

iter-006 is the D1 fusion alone, on top of iter-000 baseline. No basis-form, no artifact. The DST-DST SFPU mul vs the prior FPU mul_tiles is a single multiplication, so the SFPU/FPU latency difference is smaller than iter-004's multi-stage version.

If iter-006 is bit-identical (or near-identical PSNR) and faster — even by 0.5 ms — we land it. If it's slower (e.g., SFPU mul_binary turns out to cost more than the saved acquire+pack), reject and try a different fusion.

## Validation gate

- **PSNR ≥ 40 dB** (perceptual floor)
- **Visual checks must pass at perceptual 1× scale** — no tile seams, no banding, no NaN. This is the gate iter-003 failed.
- **Kernel ms** must improve vs prev_best (99.95 ms). Any improvement counts.

## Files to edit

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp` — collapse the two D1 acquires (around lines 439-461 of baseline) into one.

## Rollback plan

`git checkout HEAD~1 -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`. Already at baseline as of 044f398, so a single-file revert restores it.
