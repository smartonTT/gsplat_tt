# iter-002 — Layer B: Basis-form Q with tile-local centered coords

- Class: kernel-algebra
- Track: kernel-layer-b
- Date: 2026-05-25
- Status: dispatched
- Prereq: iter-001 (Layer A) REJECTed; this iter is on top of main, not on top of A (per spec §2 composition rule).

## Hypothesis

Today the inner kernel computes `Q = a·dx² + 2b·dx·dy + c·dy²` per-Gaussian with global `dx = x - μ_x`, `dy = y - μ_y`. Layer B rewrites this as:

```
Q = A·x_local² + B·x_local·y_local + C·y_local² + D·x_local + E·y_local + F
```

where:

- `A..F` are 6 fp32 coefficients precomputed per-Gaussian on host from `(μ, Σ⁻¹)`.
- `x_local, y_local ∈ [-15.5, +15.5]` (tile-local centered coords for a 32×32 tile).
- `x_local², x_local·y_local, y_local², x_local, y_local, 1` are 6 **constant basis tiles** generated once per program launch (same for every tile, every frame, every scene).

The per-Gaussian inner loop becomes 6 × `mul_tiles` with `acc_to_dest`, then `exp_tile` and the alpha clamp. Heavy ops move SFPU → FPU.

The prior basis-form attempt (opt-stable iter-057) used **global** pixel coords and PSNR collapsed to 14 dB from catastrophic fp32 cancellation. Tile-local coords bound `x², xy, y²` to `<256`, giving fp32 plenty of headroom.

## Expected impact

Per the spec, this is the largest single kernel-algebra win — kernel ms could fall well below the iter-0 99.95 ms baseline if FPU substitution lands.

## Risk

1. fp32 cancellation if the coefficient sums (A..F) accidentally use global coords on the host side. Tile-local must be enforced.
2. Per-Gaussian payload grows 9 → 10 fp32 scalars. Reader signature changes, CT-args bump → JIT cache wipe required (run_iter.sh detects kernel cpp/hpp edits and wipes automatically).
3. Validator gate: PSNR ≥ 100 dB on every view. If basis-form drifts below 100 dB after tile-local fix, BACKBURNER per spec — never the kernel.

## Files to edit

- `gsplat/rasterization.py` — `prepare_kernel_inputs`: compute A..F from (μ, Σ⁻¹) and pack 10 fp32 per Gaussian instead of 9.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp` — host: shift payload 9→10 fp32, allocate 6 basis tile CBs, run init compute kernel once per program launch to fill them.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp` — read 10 fp32 per Gaussian instead of 9.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp` — replace `dx²/2dxdy/dy²` recomputation with 6 `mul_tiles` against the per-tile basis CBs.

## Rollback plan

`git checkout HEAD -- gsplat/rasterization.py backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/` followed by full rebuild. Confirm iter-0 baseline reproduces 99.95 ms median ±2%.
