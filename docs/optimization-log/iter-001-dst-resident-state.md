# iter-001 — Layer A: Dst-resident state

- Class: kernel-algebra
- Track: kernel-layer-a
- Date: 2026-05-25
- Status: dispatched

## Hypothesis

The current `alpha_blend_compute.cpp` `tile_regs_acquire/release` around each per-Gaussian R/G/B/T accumulator update, spilling state back through CBs every iteration. Refactor to:

- Acquire `Dst` **once** per output tile,
- Keep `R, G, B, T` resident in `Dst[0..3]` across the entire Gaussian loop for that tile,
- Release once at writer handoff.

This is pure storage relocation. fp32 Dst accumulation is already the underlying compute mode, so the result is expected to be bit-identical (PSNR=∞ vs iter-0). The win is eliminating ~4 spill/reload pairs per Gaussian.

## Expected impact

Modest — likely 5-15% kernel-ms reduction (target band ~85-95 ms). The opt-v2 baseline is 99.95 ms.

## Risk

Register pressure. P300 has 16 fp32 Dst tiles with `fp32_dest_acc_en=true`. R/G/B/T occupy 4. Per-Gaussian temps must fit in the remaining 12 across pipeline stages. If the compiler spills, revert and BACKBURNER.

## Validator criteria (kernel-algebra class)

- PSNR ≥ 100 dB on every view (this is a storage-only refactor, so bit-identity is the target).
- No artifacts.
- Kernel ms median ≤ 99.95 ms (current best) to commit.

## Rollback plan

Single-file edit to `alpha_blend_compute.cpp`. On failure: `git checkout HEAD -- backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`, rebuild, re-run iter-0 baseline to confirm rollback bit-identical.
