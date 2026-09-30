# contrib_floor 1/255 visual check — REJECTED (task #28, 2026-09-30)

Board: **yyzo-bh-07 (Blackhole p100a)**. Scene: bicycle, 30 views, 1024x1024.
Code base: 7df44de (iter-150). Old = default floor 1/16384, new = 1/255 (only
`contrib_floor` in a copy of `benchmarks/cameras_v2.json` changed).
Golden and default are **unchanged**.

## Result

| | old 1/16384 | new 1/255 |
|---|---|---|
| ms/view (1 run each, with view dumps) | 145.3 | 138.7 (-4.6%) |
| stage_blend ms | 63.6 | 57.0 |
| hero vs current golden | 100.00 dB | 51.97 dB |

All 30 views: PSNR 51.8-52.6 dB vs old, max |diff| 30-40 levels (8-bit).

## Why rejected

The thin wheel spokes lose brightness in short segments: up to 36 levels darker
over ~7 px (hero pixel y=459, x=600). The segments start and stop sharply on
microblock lines. The hero worst dip ends exactly at row 460 (a 4-row boundary).
Across all 30 views, 33% of sharp dip edges (1092 of 3325) sit on 8x4 microblock lines, against
~19% expected by chance. So a spoke Gaussian that clearly contributes is dropped
in some 8x4 microblocks and kept in the neighbouring ones. That is a tile
artifact, which the user's rule says to reject.

Dropping only per-pixel contributions below 1/255 (the GPU 3DGS rule) cannot remove
36 levels. The likely cause is that the per-microblock keep test
(`render/kernels/compute/microblock_cull_compute.cpp`: constrained box-min of the
Mahalanobis distance, approximate reciprocals) overestimates the distance for very
thin, anisotropic Gaussians. At 1/16384 the ~8.3 m² threshold margin hides the
error. This is not yet verified. The next step is to compare the device keep mask
with an exact fp64 box-min for the hero spoke Gaussians.

The rest of the image is clean to the eye. The background diff is ±1 level. It
shows faint grid-aligned rectangles in the 16x heatmap, but they cannot be seen in
the images. Seam ratio of the new image vs the old one (step across grid lines,
new/old) is +0.5% on 32/16/8/4-px lines and 1.000 off-grid.

## Files

- `sbs/hero.png` — full-res old | new
- `heat/{hero,orb_00,orb_07,orb_14,orb_21}.png` — |new-old| x16, inferno
- `tilemax/*.png` — per-32x32-tile max |diff|, all 30 views
- `crops/*_worst.png` — 4x zoom old | new | heat around each view's worst tile
- `crops/hero_spoke_zoom8x.png` — 8x zoom of the worst spoke dip (old | new | signed diff x4)
- `crops/hero_bgpatch_zoom4x.png` — background patch (old | new | signed diff x32)
- `stats.json` — per-view PSNR, diff stats, tile-max stats, seam ratios
- `compare_floors.py` — the script (`compare_floors.py OLD NEW OUT --sbs hero`)
