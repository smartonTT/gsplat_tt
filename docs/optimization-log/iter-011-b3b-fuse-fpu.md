# iter-011 — Stage B3b fuse: drop CB_POWER via binary_dest_reuse&lt;ELWADD&gt;

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-010 e-fuse-fpu (KEEP, 97.77 ms).

## Hypothesis

Stage B3b is structured as two consecutive FPU `add_tiles` calls, with the
second one consuming the first's output through CB_POWER:

```
acquire { add_tiles(CB_Q, CB_Q, 0, 1, 0) → pack CB_POWER } release    // B3b1
cb_wait_front(CB_POWER, 1)
acquire {
  add_tiles(CB_POWER, CB_Q, 0, 2, 0)   → dst[0] = Q
  mul_unary(NEG_HALF_BITS)             → dst[0] = -0.5·Q
  copy_tile(CB_CONST_ZERO, 1)
  binary_min_tile(0, 1, 0)
  exp_tile<true>(0)
  mul_unary(opacity_bits)
  copy_tile(CB_CONST_099, 1)
  binary_min_tile(0, 1, 0)             → dst[0] = alpha
  pack CB_ALPHA
} release
```

This is *exactly* the iter-010 shape: a producer `add_tiles` whose output
flows into a second op that takes it as `in0`. Replace with the
DEST_TO_SRCA fusion:

```
acquire {
  add_tiles_init(CB_Q, CB_Q);
  add_tiles(CB_Q, CB_Q, 0, 1, 0);                                     // dst[0] = CB_Q[0]+CB_Q[1]
  binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(CB_Q);
  binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(CB_Q, 2, 0);          // dst[0] += CB_Q[2]
  mul_unary(NEG_HALF_BITS);
  copy_tile(CB_CONST_ZERO, 1);
  binary_min_tile(0, 1, 0);
  exp_tile<true>(0);
  mul_unary(opacity_bits);
  copy_tile(CB_CONST_099, 1);
  binary_min_tile(0, 1, 0);
  pack CB_ALPHA
} release
```

Saves 1 acquire/commit/wait/release + the CB_POWER reserve/pack/push/
wait/pop roundtrip per Gaussian. Pure FPU on the add side, then SFPU as
before — the SFPU mul_unary that follows is *not* across binary_dest_reuse,
so this doesn't replicate the iter-008 suspect pattern.

`binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` is the ELWADD variant of
the same primitive iter-007 (ELWMUL on CB_SAT_MASK) and iter-010 (ELWMUL
on CB_SAT_MASK) both proved out. ELWADD is the standard eltwise add type
used by tt-metal's elsewhere in the kernel.

## Bonus payoff

CB_POWER becomes unused. Could potentially be removed from the CB layout
in a follow-up, but for this iter we just leave it allocated; no
push/pop traffic on it.

## Validation gate

- PSNR ≈ iter-010 levels expected: 40-44 dB per view. Algebra-identical
  but DEST_TO_SRCA path quantizes intermediates slightly differently
  than CB pack/unpack roundtrips (same pattern observed in iter-007 and
  iter-010).
- All 8 visual checks must pass with diff10 dominated by diffuse speckle.
- kernel ms ≤ prev_best (97.77 ms).

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Delete the entire B3b1 acquire block (lines ~360-369 plus its
    `cb_wait_front(CB_POWER, 1)`).
  - Replace the first `add_tiles(CB_POWER, CB_Q, 0, 2, 0)` in the
    B3b2+C acquire with the two-line `add_tiles(CB_Q, CB_Q, 0, 1, 0)` +
    `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(CB_Q, 2, 0)`.
  - Drop the `cb_pop_front(CB_POWER, 1)` (no longer pushed).

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-010 state.
