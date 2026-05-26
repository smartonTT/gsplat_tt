# iter-013 — Drop Stage E's sat_mask multiplication (algorithmically redundant)

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: planned
- Predecessor: iter-010 e-fuse-fpu (KEEP, 97.77 ms / 40.40 / 43.70 / 40.15 dB).

## Hypothesis

Stage E currently does `T_state ← T_state · (1 - alpha) · sat_mask`, fused
via iter-010's pattern:

```
acquire {
  mul_tiles(CB_T_STATE, CB_ONE_MINUS_ALPHA, 0, 0, 0);
  binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK, 0, 0);
}
```

The `· sat_mask` multiplication is **algorithmically redundant** because
Stage D1 already gates by sat_mask via:

```
acquire {
  mul_tiles(CB_ALPHA, CB_T_STATE, 0, 0, 0);
  binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK, 0, 0);
}  → contrib
```

For an "inactive" pixel (sat_mask=0):
- D1: contrib = α · T_state · 0 = 0 ✓ (no contribution)
- E original: T_state := T_state · (1-α) · 0 = 0 (zeroed)
- E proposed: T_state := T_state · (1-α) (stays small but nonzero)

But D1's sat_mask zeroing is what makes the pixel actually inactive — E's
sat_mask was just keeping T_state nominally clean.

### Why T_state staying nonzero is harmless

Before any Stage F refresh, sat_mask = 1 everywhere (initial state).
First refresh is at g=16: sat_mask becomes 0 for pixels where T_state <
1e-4. For those pixels in subsequent Gaussians:

```
T_state (iter-013) = T_low · (1-α)^k     for k = 1, 2, ...
                  ≤ T_low                 (since 0 ≤ 1-α ≤ 1)
                  < 1e-4
```

At the next Stage F refresh, `T_state >= 1e-4` is still false → sat_mask
stays 0. The pixel remains permanently inactive. Same algorithmic outcome
as the original.

### PSNR impact

Each `binary_dest_reuse<ELWMUL, DEST_TO_SRCA>` application has cost ~1 dB
in iter-007 + iter-010 + iter-011 history (cumulative truncation of fp32 →
bf16 during DEST_TO_SRCA reload). Removing one application should **add
back ~1 dB** of headroom: expected hero ~41 dB (up from 40.40).

This is the inverse of iter-010's PSNR cost: iter-010 added the
binary_dest_reuse to fuse the acquire (-1 dB, +0 ms). iter-013 removes the
binary_dest_reuse (algorithmically redundant op, +1 dB, possibly tiny ms
gain from one fewer FPU op per Gaussian).

### Perf impact

Inner-loop acquire count is unchanged (still 1 acquire for Stage E). But
the acquire is shorter: one `mul_tiles` instead of `mul_tiles +
binary_dest_reuse_init + binary_dest_reuse`. Expected: small improvement
(0.1-0.3 ms).

## Validation gate

- PSNR per view: expected hero ≥ 40.4 dB, side ≥ 43.7 dB, top ≥ 40.1 dB
  (or BETTER, ideally +0.5-1 dB on all views). No view below the 40 dB
  perceptual floor.
- All 8 visual checks must pass. diff10 should be dominated by the iter-010
  noise floor (uniform speckle).
- kernel ms ≤ prev_best (97.77 ms). Expected: 97.4-97.7 ms range.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - In Stage E's iter-010 fused acquire, remove the two lines:
    - `binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK);`
    - `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK, 0, 0);`

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-010 state.

## Why not also remove D1's sat_mask?

D1's sat_mask is THE algorithmic gate. Removing it would let inactive
pixels accumulate tiny (< 1e-4 per Gaussian) color contributions. Over a
typical Gaussian count of ~10K per tile, cumulative bleed could reach
1e-3, near 1/255 visibility. Riskier and defers to a future iter once we
verify Stage E's removal is clean.
