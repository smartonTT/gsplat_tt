# iter-010 — Stage E fuse: T·(1-α) and ·sat_mask in one acquire

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-007 d1-fuse-fpu (KEEP, 99.34 ms). iter-009 (REJECT, perf regression) showed that fusing INDEPENDENT FPU mul_tiles into one acquire loses pack/setup overlap.

## Hypothesis

Stage E computes the front-to-back transmittance update as three acquire blocks:

```
acquire { rsub_unary_tile(alpha, 1.0)                 → pack CB_ONE_MINUS_ALPHA }   // E1
acquire { mul_tiles(T_STATE, ONE_MINUS_ALPHA)         → pack CB_T_TMP          }   // E2
acquire { mul_tiles(T_TMP,   SAT_MASK)                → spill CB_T_STATE       }   // E3
```

Steps E2 and E3 are exactly the iter-007 D1 pattern in disguise: a FPU
`mul_tiles` followed by another FPU multiply that consumes its output as one
operand. Fuse them with `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>`,
keeping the intermediate in dst[0] rather than spilling to L1 via CB_T_TMP:

```
acquire {
  mul_tiles_init(CB_T_STATE, CB_ONE_MINUS_ALPHA);
  mul_tiles(CB_T_STATE, CB_ONE_MINUS_ALPHA, 0, 0, 0);          // dst[0] = T · (1-α)
  binary_dest_reuse_tiles_init<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK);
  binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>(CB_SAT_MASK, 0, 0); // dst[0] *= sat_mask
  pack dst[0] → CB_T_STATE (after pop_front)
} release
```

Saves 1 acquire/commit/wait/release + the CB_T_TMP reserve/pack/push/wait/pop
roundtrip per Gaussian. Pure FPU — same op type repeated, no SFPU mixed in.
This matches the iter-007 safe pattern; iter-009's regression was about
INDEPENDENT FPU ops (no producer-consumer chain), here we have a
producer-consumer chain so the second op overlaps naturally with the first's
result staying live in dst.

This is NOT a iter-009-style "collapse independent ops" fusion. It's an
iter-007-style "keep the intermediate in dst instead of spilling" fusion.
iter-006 already proved the saving in D1 with a SFPU equivalent (regressed
0.75 ms), iter-007 used the FPU variant of the same trick and won 0.6 ms.

## Validation gate

- PSNR = ∞ expected (algebra-identical: same two multiplications, just
  fewer Dst transactions). 40 dB perceptual floor far below this.
- No tile seams / cross-hatch in diff10. Bit-identical output to iter-007.
- kernel ms ≤ prev_best (99.34 ms) — any improvement counts.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Replace Stage E steps 2 and 3 (two `mul_tiles` acquires through
    CB_T_TMP) with a single acquire holding `mul_tiles` +
    `binary_dest_reuse_tiles<ELWMUL, DEST_TO_SRCA>`. Drop CB_T_TMP
    pack/wait/pop in Stage E only — CB_T_TMP is still used in Stage D2.

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores the iter-007/iter-009-revert state.
