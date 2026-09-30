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

# Follow-up: task #41 — cause found, fixed, 1/255 ADOPTED (iter-156, 2026-09-30)

Board: **yyzo-bh-07 (Blackhole p100a)**. Scene: bicycle, 30 views, 1024x1024.

## The keep test is correct

`GSPLAT_TT_DUMP_CULL=<dir>` (new, default off) dumps the first frame's slab
records (conic, mean, opacity, word3 mask). `cull_mask_check.py` compares every
(gaussian, tile, microblock) keep bit with an exact fp64 box-min test. Hero at
1/255: 3,369,033 pairs x 32 microblocks. 1,657 device culls the fp64 test would
keep; all are borderline (m2 within 1e-3 of the threshold, from the SFPU log)
and their peak pixel alpha is <= 0.0039 < 1/255, so none changes a pixel. An
fp32 emulation of the kernel matches fp64 exactly. Result: `t41/mask_check_hero_f255.json`.

## The real cause

The mask keeps a whole 8x4 block when any point of it reaches the floor, and the
blend then adds the gaussian to all 32 pixels, including pixels where it is
below the floor. At pixel (459, 600), 4,988 of 5,168 pairs are below 1/255 there
(faint, bright haze in front of the spoke, alpha ~0.003 each); together they add
~52 levels. Blocks that keep them are bright, the next block is dark: the seam.
A CPU render of the dumped records reproduces the device image within 1 level.

## The fix

Per-pixel floor in the blend: `alpha < contrib_floor -> 0` (the GPU 3DGS rule),
compiled in by `BLEND_PIXEL_FLOOR`, on by default (`GSPLAT_TT_BLEND_PIXEL_FLOOR=0`
turns it off; that path at the old floor gives the old golden, 100.00 dB).
Device output now equals an fp64 per-pixel render of the same records within
1 level (5 tiles checked). Model test: `tests/spec/test_pixel_floor_seams.py`.

## Visual check (new = 1/255 + pixel floor, old = 1/16384)

| | mask-only 1/255 (task #28) | 1/255 + pixel floor |
|---|---|---|
| diff seam ratio, max over 30 views and 4/8/16/32 px lines | 1.11-1.19 | 1.00-1.07 (mean y32 1.008) |
| PSNR vs old, 30 views | 51.8-52.6 dB | 42.2-43.2 dB |
| max \|diff\| | 40 | 56 |

No grid structure in the full-res side-by-side or the x16 heatmaps
(`t41/sbs_hero.png`, `t41/heat_hero.png`, `t41/heat_orb_19.png`). The diff is
larger but follows scene edges: the sub-floor haze is now removed everywhere,
as on a GPU, so spokes and edges are evenly a bit darker. `t41/hero_spoke_zoom8x.png`
shows old | mask-only | pixel floor | diff (mask-only: broken dashes) | diff
(pixel floor: continuous along the spoke). Verdict: no seams -> adopted.

## Adoption

`benchmarks/cameras_v2.json` bicycle contrib_floor 6.1035e-05 -> 1/255. Old golden
archived to `tests/fixtures/hero/archive/hero_golden_8bit_floor16384_pre-iter156.png`;
new golden = this render (hero 42.49 dB vs the old golden).

Speed (2 interleaved rounds each, base 93aa708): **107.75 -> 103.23 ms/view
(-4.5 ms, -4.2%)**, stage_blend 49.62 -> 44.80 ms. The pixel-floor compare costs
~0.5 ms/view (115.2 mask-only vs 115.7 on the pre-rebase base).
