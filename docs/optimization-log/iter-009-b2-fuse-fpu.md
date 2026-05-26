# iter-009 — Stage B2 fuse: dx², dy², dx·dy in one acquire

- Class: kernel-algebra
- Track: post-basis-form-revert
- Date: 2026-05-26
- Status: dispatched
- Predecessor: iter-007 d1-fuse-fpu (KEEP, 99.34 ms). iter-008 (REJECT, deadlock) tried a SFPU-FPU mixed chain in D2 — avoided here.

## Hypothesis

Stage B2 computes three pairwise products in three separate acquire blocks,
one acquire per multiply. The block lives inside the per-Gaussian inner loop,
so on a ~341k-Gaussian frame this fires ~341k×3 = ~1M acquire/commit/wait/
release cycles per frame.

```
acquire { mul_tiles(CB_DX, CB_DX) → dst[0]; pack → CB_DX2  } release    // B2a
acquire { mul_tiles(CB_DY, CB_DY) → dst[0]; pack → CB_DY2  } release    // B2b
acquire { mul_tiles(CB_DX, CB_DY) → dst[0]; pack → CB_DXDY } release    // B2c
```

The author's comment ("Dst is limited; can't keep many tiles live at once
with fp32 acc") predates the current `fp32_dest_acc_en=true` config, where
DST holds 4 fp32-sized slots on Wormhole/Blackhole. We can fit three live
products at once:

```
acquire {
  mul_tiles_init(CB_DX, CB_DX); mul_tiles(CB_DX, CB_DX, 0, 0, 0);  // dst[0] = dx²
  mul_tiles_init(CB_DY, CB_DY); mul_tiles(CB_DY, CB_DY, 0, 0, 1);  // dst[1] = dy²
  mul_tiles_init(CB_DX, CB_DY); mul_tiles(CB_DX, CB_DY, 0, 0, 2);  // dst[2] = dx·dy
  pack dst[0]→CB_DX2; pack dst[1]→CB_DY2; pack dst[2]→CB_DXDY
} release
```

Saves 2 acquire/commit/wait/release cycles per Gaussian. The chain is
**pure FPU** — same op type repeated, no SFPU mixed in. That matches the
iter-007 safe pattern; iter-008's deadlock came from mixing SFPU copy/mul
with FPU `binary_dest_reuse_tiles` in one acquire.

Stage B1 (lines 266-281) already packs from two DST slots to two CBs in one
acquire, so the local precedent for multi-slot packing exists.

## Validation gate

- PSNR = ∞ expected (math is algebra-identical; same multiplications, just
  fewer Dst transactions). 40 dB perceptual floor is far below this.
- No tile seams / cross-hatch in diff10.
- kernel ms ≤ prev_best (99.34 ms) — any improvement counts.

## Files edited

- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  - Replace the three separate B2a/B2b/B2c acquire blocks with one
    acquire containing three `mul_tiles_init` + `mul_tiles` pairs and
    three `pack_tile` calls.

## Rollback plan

`git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
restores iter-007/iter-008-revert state.
