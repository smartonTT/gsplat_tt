# Microblock-shape model (task #408)

Question: would a different 32-pixel blend microblock shape cut (record, microblock)
dispatches enough to matter? Today's shape is 4x8 (4 rows by 8 columns).
CPU model only, no device run.

**Verdict: gate FAILS. Keep 4x8; no device follow-up.**
4x8 has the fewest dispatches on all 30 views (the best other shape, 8x4, is
+8.3% on its best view, +10.0% on average). No other shape is bit-exact against 4x8.

## Result (bicycle, 30 bench views, 1024x1024)

| shape | dispatches/view | ratio vs 4x8 | pair ops/view | live records/view | lane fill | bit-exact (fp32) | u8 ch diffs/view | saving ms/view (spec 3.682 x ratio) | saving ms/view (t148 per-dispatch) | saving ms/view (t147 per-live-rec) |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 4x8 | 6,119,378 | 1.000 | 4,796,240 | 2,128,168 | 0.325 | yes | 0 | +0.000 | +0.000 | +0.000 |
| 8x4 | 6,729,774 | 1.100 | 4,633,951 | 2,169,698 | 0.304 | no | 4,519 | -0.367 | -0.267 | -0.074 |
| 2x16 | 7,479,250 | 1.222 | 6,494,729 | 2,130,403 | 0.268 | no | 4,523 | -0.818 | -0.595 | -0.004 |
| 16x2 | 9,784,481 | 1.599 | 5,954,673 | 2,251,518 | 0.221 | no | 6,465 | -2.205 | -1.604 | -0.221 |
| 1x32 | 11,404,439 | 1.864 | 6,647,720 | 2,160,537 | 0.180 | no | 6,383 | -3.180 | -2.313 | -0.058 |
| 32x1 | 16,719,544 | 2.732 | 9,322,571 | 2,326,683 | 0.136 | no | 7,217 | -6.378 | -4.640 | -0.356 |

records/view 2,702,239  cull-kept 2,457,861  4x8 disp without early stop 7,473,571
gate: FAIL (no shape is bit-exact and >= 0.09 ms/view)

Savings are 4x8 minus the shape, so negative means slower. Three cost conversions:
- spec: 3.682 ms/core x (1 - dispatch ratio)
- t148: 65 cycles per microblock dispatch at 1.35 GHz over 110 cores
- t147: 0.197 us per live record per tile over 110 cores

All three say every shape is slower than 4x8. The gate needs a shape that is bit-exact
and saves at least 0.09 ms/view (2.5% fewer dispatches). No shape meets either condition.

Why 4x8 wins: a record pays one dispatch per block its ellipse touches. The two
most compact shapes (4x8, 8x4, perimeter 24 px) are touched by the fewest
ellipses; long thin blocks (16x2, 1x32, 32x1) are crossed by many more, and
their lane fill drops from 0.325 to 0.14-0.22. 4x8 beats 8x4 on every view
(why exactly, e.g. footprint orientation in this scene, was not analysed).

One note: 8x4 has 3.4% fewer *pair ops* (the jump-walk works on bit pairs
2J, 2J+1), and on its best view 4.7% fewer. That would only help if device cost
were per pair rather than per dispatch, and 8x4 also has 10% more dispatches,
2% more live records and is not bit-exact. Not worth a build.

## Why no shape is bit-exact

The T early stop works per microblock: every 512 records the live mask drops
blocks whose pixels all have T < 1/256. The per-pixel floor applies to alpha, not
alpha*T, so pixels past the stop would still have received small contributions.
Which pixels stop when therefore depends on the block shape. Differences per
shape are small (max abs ~0.0038 = eps, 82-84 dB PSNR vs 4x8, ~4.5-7k u8
channel values per view out of 3.1 M) but not zero.

## Validation

- 4x8 lane fill 0.325 and 2.7 M records/view match t148 (0.325, ~2.95 M on its views).
- Hero 4x8 render from the model: 42.98 dB vs benchmarks/reference_v2/hero.png
  (device xvpin hero: 42.51 dB). This is the model's CPU image, not a device screenshot.

## Method

- project.py: numpy projection of the 30 views in benchmarks/cameras_v2.json
  (EWA +0.3 low-pass, near 0.2, 3-sigma radii, radius cap 512, opacity >= 1/255,
  PRECULL rect shrink with margin 0.25).
- mbsim.cpp: per tile, depth-sort the records. For each shape:
  - apply the band cull test (ellipse m2 <= 2 ln(op/floor) + 0.05 meets the block's
    pixel-centre box)
  - run the fp32 blend with the 512-period live mask (eps 1/256)
  - count dispatches, pairs and live lanes, and diff the image against 4x8.
- aggregate.py: averages over views and applies the cost models.
- out/*.txt: raw per-view output.

Caveats (none changes the ranking; the gaps are 10% or more):
- The cull runs in fp64 here; the device uses fp32 and the SFPU log.
- exp runs on the CPU.
- T is compared in fp32, not as bf16-packed T.
- The PRECULL rect is approximate.

## Re-run

    docs/mb-shape-model/run.sh [scenes/bicycle.ply] [/tmp/mb-shape-model]
