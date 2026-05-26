# iter-008 — Stage D2 fuse: per-channel mul_unary + add via FPU `DEST_TO_SRCA`

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-007 d1-fuse-fpu (KEEP, 99.34 ms)

## Hypothesis

Apply the same single-acquire pattern that worked for iter-007 to Stage D2.
For each channel C ∈ {R, G, B} the baseline is:

```
acquire { copy(CB_CONTRIB)→dst[0]; mul_unary(color_c_bits); pack dst[0]→CB_T_TMP } release
acquire { add_tiles(CB_COLOR_C_STATE, CB_T_TMP)→dst[0]; pop state; pack dst[0]→CB_COLOR_C_STATE } release
```

Collapse to one acquire using `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`:

```
acquire {
  copy(CB_CONTRIB)→dst[0];                                          // SFPU copy
  mul_unary(color_c_bits);                                          // SFPU unary mul (scalar in tile)
  binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(CB_COLOR_C_STATE, 0, 0);  // FPU dst[0] += state
  pop CB_COLOR_C_STATE; pack dst[0]→CB_COLOR_C_STATE
} release
```

Saves per channel: 1 acquire/commit/wait/release cycle + 1 CB_T_TMP
reserve/pack/push/wait/pop. With 3 channels, that's 3× the per-iter savings
we got in iter-007.

iter-007 measured ~0.6 ms saved by fusing one similar pair. Three pairs may
yield ~1.5–1.8 ms naively, though the SFPU copy + mul_unary still happens
so the FPU win is only on the add half.

## Validation gate

- PSNR ≥ 40 dB perceptual floor.
- No tile seams / cross-hatch in diff10.
- kernel ms ≤ prev_best (99.34 ms).

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Replace the two-acquire R/G/B pattern (×3) with a single-acquire
    FPU `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` per channel.

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-007 state.
