# Microblock CPU Spec (gsplat_tt_2)

Status: **frozen at iter-000**. Source: `../gsplat_tt/docs/optimization-log/microblock-kernel-design.md` §0, §2, §3, §7, §9.1.
TT-specific sections (§1, §4, §5, §6, §8, §11 of the source) are deferred to Phase 5.

This document is the **implementation contract** for iter-006 through iter-009. The
worker reads this; the validator reads this; the supervisor enforces it.

## 0. Why microblocks at all

The current per-pair Mahalanobis cull (`gsplat/rasterization.py::_assign_gaussians_to_tiles`)
drops pairs whose closest-pixel-in-tile contribution is below `contrib_floor`. But
within a *kept* tile, most Gaussians touch only a fraction of the 32×32 pixels.
Microblock binning extends the cull to 4r×8c sub-regions: a `(Gaussian, microblock)`
pair is dropped iff its closest-pixel-in-microblock peak alpha is below
`contrib_floor`. The inner per-tile loop then iterates microblock-major and only
processes the surviving `(m, g)` pairs.

The CPU does not have an SFPU; the 4×8 granularity is preserved purely to keep
the algorithm bit-portable to the eventual metal kernel. CPU performance gain
is from culling pair-iterations and increasing data locality in the inner loop.

## 1. Microblock enumeration (host ↔ device agreement)

32 microblocks per tile, raster order (rows 0..3 first, left-to-right):

```
m = (tile_row_band) * 4 + (tile_col_group)
tile_row_band in [0, 8)    # 8 row-bands of 4 rows each
tile_col_group in [0, 4)   # 4 col-groups of 8 cols each
```

This is the same indexing the metal kernel will use (source doc §3.4 + §6.3). The CPU
code never decodes m back into (face, band, colhi) — that arithmetic is metal-specific.
CPU just uses m as a flat 0..31 index plus the (ox, oy) pair for the microblock's
pixel origin within its tile:

```
mb_origin_x[m] = (m & 3) * 8           # 0, 8, 16, 24
mb_origin_y[m] = (m >> 2) * 4          # 0, 4, 8, 12, 16, 20, 24, 28
```

## 2. Per-microblock cull math (source doc §3.2, verbatim)

For each surviving (gaussian, tile) pair after the existing per-tile Mahalanobis cull,
compute a (32,) bool mask:

```python
mb_ox = tx_tile + mb_origin_x          # (32,) for this pair
mb_oy = ty_tile + mb_origin_y          # (32,)

# closest point inside microblock m to gaussian g's mean:
cx = clamp(mean_x, min=mb_ox, max=mb_ox + 8)
cy = clamp(mean_y, min=mb_oy, max=mb_oy + 4)
dx, dy = cx - mean_x, cy - mean_y

# Mahalanobis quadratic at closest point:
m2 = (c*dx*dx - 2*b*dx*dy + a*dy*dy) / det
keep_mb = opacity * exp(-0.5 * m2) >= contrib_floor   # (32,) bool
```

This is conservative (closest-point minimizes m², overestimates the peak) so any
pair admitted by the per-tile cull whose every microblock fails this test
contributes no visible pixels by definition — invariant 1 (§4) holds.

Vectorize over P pairs × 32 microblocks: `keep_mb` shape `(P, 32)`.

## 3. Host emission (source doc §3.1, adapted for CPU)

Per tile, three blobs:

```python
coeff_table[tile]: (L, 9) fp32        # one row per gaussian-of-tile
                                       # lanes: a, b, c, det_inv, opacity, color_r, color_g, color_b, depth
                                       # (CPU layout; metal repacks to A..F basis-form in Phase 5)

mb_header[tile]:   (32, 2) uint32     # (offset, count) per microblock

mb_stream[tile]:   (L_prime,) uint32  # depth-sorted local_gaussian_idx
                                       # per microblock, concatenated
                                       # L_prime = sum_m count[m]
```

The per-microblock streams are depth-sorted by construction (we filter `keep_mb`
against the tile's already-depth-sorted gaussian list).

## 4. Correctness invariants (source doc §7, restated)

**Invariant 0 (north star).** At a fixed `contrib_floor`, PSNR of any microblock-aware
render vs the per-tile reference must be ≥ 60 dB. Only floating-point
accumulation order differences are admissible.

**Invariant 1 (mask completeness).** Every pixel whose true alpha contribution from
Gaussian g exceeds `contrib_floor` lies inside at least one microblock m with
`keep_mb[g, m] == True`. Holds by construction of §2.

**Invariant 2 (depth order preservation).** Per microblock, `mb_stream` indices appear
in the same relative order as in the tile's depth-sorted list. No inversions.

**Invariant 3 (drop-pair budget).** Pairs that pass per-tile cull but fail every
microblock: < 5%. If the host measures > 5%, the per-tile cull is too loose —
fail loudly and investigate.

## 5. Validation tests (source doc §9.1, adapted for CPU)

These run as Catch2 unit tests AND pytest tests on every iter starting at iter-006:

- `mb_stream` length sums correctly: `sum(mb_header[m].count) == L_prime`
- Per-microblock depth monotonicity: no inversions
- Mask completeness: for randomized (cov, opacity, mean, tile), every pixel
  with true alpha ≥ `contrib_floor` lies inside an active microblock
- Drop rate < 5% on stitch_doll (regression alert if exceeded)
- End-to-end PSNR vs per-tile reference ≥ 60 dB

## 6. Deferred to metal (Phase 5)

Sections §1, §4, §5, §6, §8, §11 of the original design doc — DST slot map,
LREG allocation, SFPU instruction sequence, replay buffer, cycle counting,
file diffs in `backends/tt/`. The CPU implementation does NOT attempt to mirror
these; the metal port consults the original doc directly when Phase 5 begins.

## 7. Implementation order (the iter sequence)

| iter | scope |
|---|---|
| 006 | numpy: add `microblock_cull(gaussian_ids, tile_ids, depths, …)` → `(mb_header, mb_stream)`. Drop-rate logging. Invariants 1/2/3 unit tests (pytest). |
| 007 | C++: port `microblock_cull` and the new `(mb_header, mb_stream)` data path. Catch2 tests for mask completeness on synthetic inputs. |
| 008 | C++: change `alpha_blend` inner loop from per-pair to microblock-major. R/G/B/T state in scalars across each microblock's Gaussian list. Per-microblock early-term hook (not yet active). |
| 009 | C++: activate per-microblock early-term (`T_max_in_mb < 1/256`). Expose `contrib_floor` slider in viewer (Phase 4 dependency). |
