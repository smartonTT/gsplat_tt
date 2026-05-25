# iter-003 — M1: Basis-form host coefficients + FPU q-accumulation (tile-global basis)

- Class: kernel-algebra
- Track: fpu-heavy-m1
- Date: 2026-05-25
- Status: dispatched
- Spec: `docs/superpowers/specs/2026-05-25-fpu-heavy-architecture.md` §M1
- Predecessor: iter-2 attempted similar basis-form via tile-local centered coords + SFPU, got 47 dB / 106 ms; reclassified as building block under 40 dB perceptual floor.

## Hypothesis

Rewrite per-Gaussian Q in **basis form** with **host-side coefficients**:

```
Q(x, y) = A·x² + B·xy + C·y² + D·x + E·y + F
```

where (mx, my are tile-local means in [0, 32)):
```
A = qxx
B = 2·qxy
C = qyy
D = -2·(qxx·mx + qxy·my)
E = -2·(qyy·my + qxy·mx)
F = qxx·mx² + 2·qxy·mx·my + qyy·my²
```

The kernel then builds Q in one DST slot via **6 FPU `mul_tiles_bcast_scalar` calls with acc_to_dest** (the default on Blackhole) against precomputed basis tiles. This replaces the SFPU dx/dy/sq/mul chain with pure FPU FMA accumulation.

## API (Blackhole)

- `mul_tiles_bcast_scalar(cb_basis, cb_scalar, tile_basis_idx, tile_scalar_idx, dst_idx)` — broadcast-multiply where `cb_scalar` contains a scalar encoded as a tile (only [0,0] element is used). On WH/BH, accumulation is the default behavior (see `eltwise_binary.h:100`).

## Basis tiles (per-tile constants)

6 fp32 tiles, identical across all owner tiles (tile-local coords [0, 31]):
- `x²_tile[i,j] = (j + 0.5)²`
- `xy_tile[i,j] = (j + 0.5) * (i + 0.5)`
- `y²_tile[i,j] = (i + 0.5)²`
- `x_tile[i,j]  = j + 0.5`
- `y_tile[i,j]  = i + 0.5`
- `one_tile[i,j] = 1.0`

For M1, generate host-side once at program init and push to 6 dedicated CBs (or one CB with depth=6).

## Inner-loop kernel logic (M1, keeping current per-Gaussian acquire scheme)

```cpp
// existing acquire ...
// dst[4] = Q (rebuilt per Gaussian)
copy_tile(cb_zero, 0, 4);   // Q = 0 (cb_zero is a constant-zero tile)
mul_tiles_bcast_scalar(cb_x2,  cb_scalars_tile, 0, IDX_A, 4);  // Q += A·x²
mul_tiles_bcast_scalar(cb_xy,  cb_scalars_tile, 0, IDX_B, 4);  // Q += B·xy
mul_tiles_bcast_scalar(cb_y2,  cb_scalars_tile, 0, IDX_C, 4);  // Q += C·y²
mul_tiles_bcast_scalar(cb_x,   cb_scalars_tile, 0, IDX_D, 4);  // Q += D·x
mul_tiles_bcast_scalar(cb_y,   cb_scalars_tile, 0, IDX_E, 4);  // Q += E·y
mul_tiles_bcast_scalar(cb_one, cb_scalars_tile, 0, IDX_F, 4);  // Q += F·1
// negate + half: Q := -0.5 * Q  (use sfpu mul_unary or pack -0.5 into a constant)
// ... existing exp, alpha, blend chain continues using cb_q ...
```

## Validation gate

- **PSNR ≥ 40 dB** on hero/side/top (perceptual floor; previous 100 dB retired).
- **Visual checks** must pass (no NaN holes, no tile-grid seams, no color-clipping bands, no gross geometry shift). See `prompts/validator.md`.
- **Timing budget:** M1 may regress slightly vs iter-0 baseline (99.95 ms). 6 mul_tiles_bcast + bookkeeping is comparable to the 3 mul + SFPU chain it replaces. M3 (DST-resident) reclaims the cost.

## Files to edit

- `gsplat/rasterization.py` — `prepare_kernel_inputs`: compute (A..F) from (mx, my, qxx, qxy, qyy); pack 10 fp32 per Gaussian instead of 9.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h` — `SCALAR_PACK_BYTES` 36 → 40; add new CB indices for basis + cb_zero.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp` — encode 10 fp32 per pack; allocate basis tile CBs; one-shot init-compute (or host-fill) for basis tiles.
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp` — pack page bytes update if needed (still fits in 64B page, so no change to padded page size).
- `backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp` — replace stages B1-B3 (dx/dy/sq/mul/sum) with 6 mul_tiles_bcast_scalar building Q in dst[4], then continue the existing exp/alpha/blend chain.

## Rollback plan

`git checkout HEAD -- gsplat/rasterization.py backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/` then rebuild. Confirm iter-0 baseline reproduces.

## Anti-drift rules

- Tile-local means MUST be enforced (mx, my in [0, 32)). Global coords blow up fp32 precision in A..F coefficients (opt-stable iter-057 had 14 dB collapse from this).
- bf16 in CB_SCALARS is fine; the [0,0] scalar element doesn't suffer the 8-step bf16 issue at small magnitudes.
- Do not commit. Supervisor decide_and_log commits if KEEP.
