# Saturation-depth cull, step 0: CPU sizing (task #675, throughput PLAN D3)

Date: 2026-10-13 (PDT). CPU only, no device. Code: `docs/sat-depth-cull/{step0.py,satcull.c,aggregate.py}`.

## Verdict: **SHELVE (best net -0.001 ms/view, oracle 0.104; kill line 0.15)**

- Perfect-knowledge bound (oracle: every tile cut exactly at its own saturation depth, no
  fallback): **0.104 ms/view** on the 30-view orbit, 0.102 ms/view on the
  training tour. Both are below the 0.15 ms/view kill line before any prediction loss.
- Real predictor (previous view reprojected): best arm nets **-0.001 ms/view**
  (orbit, m0.0_s1). Orbit arms with negative net (fallback waste exceeds capture): m0.02_s0, m0.02_s1, m0.05_s0, m0.05_s1, m0.0_s0, m0.0_s1.
- Why it is small: only 14.9% of tiles ever reach T < eps on every pixel
  (41.4% of pixels never do: sky, thin foliage, tile edges against background), and
  in the tiles that do, few pairs lie past the saturating one (1.61% of all pairs).
  The blend's per-microblock live mask (blend-waste-t148) already skips most dispatch work for
  those pairs, so the true per-pair saving is below the cost model used here.

## Method

- Model: `scenes/bicycle.ply` (6.1 M Gaussians, SH0), projected with the same math as
  `gsplat.rasterization.project_gaussians` (eps2d 0.3, near 0.2, opacity >= 1/255, 3-sigma
  capped by the contribution floor). Pairs = bbox tiles kept by the per-pair peak-contribution
  test at floor 1/255 (`get_tile_assignments`), i.e. the K2 pair set; sorted by (tile, depth).
- Saturation: each tile is blended front to back on the CPU (alpha clamp 0.99, alpha < 1/255
  skipped) until every pixel has T < 1/256 (device BLEND_T_EPS). The saturating pair's depth is
  the tile's saturation depth; pairs behind it are the oracle share.
- Prediction for view i from view i-1: every saturated pixel of view i-1 is lifted to 3D
  (pixel ray at the depth of the Gaussian that saturated it), warped into view i, and carries
  that Gaussian's view-i depth. Per tile the cut is the maximum key over its 4x4 cells; a tile
  with any empty cell (disocclusion) or any warped unsaturated pixel gets no cut. Margin m
  multiplies the cut depth by (1+m) (0, 2, 5%); slack s=1 takes the max over the 3x3 tile
  neighbourhood (1 tile of screen slack).
- Scoring per tile with a cut: success if the true saturation depth <= cut (saves the pairs
  behind the cut); fallback if not (the predicted cut was too early). A fallback reruns the tile
  with its full list, so it wastes the pairs blended before the cut. Net = (captured - wasted)
  pairs x cost per pair. The first view of a pass has no prediction and renders in full.
- Cost per pair (generous upper bound): at best-iter-229 (1-chip b2b 8.063 ms/view) the
  pair-proportional stages are K2/sort 1.33 + mat 1.33 + blend 3.79 = 6.45 ms/view
  (docs/training-view/ANALYSIS.md, ttp/t668-training-view-analysis), divided by the orbit's mean
  CPU pair count: **2.74 ns/pair**. A culled pair is credited with all three stages.
- Camera sets: `benchmarks/cameras_v2.json` (30-view orbit, 1024x1024, the headline set; hero
  then orb_00..orb_28, steps ~2 deg after the first 18 deg jump) and
  `benchmarks/cameras_train_nn.json` (ttp/t667-training-bench: the dataset's 169 training views
  in greedy nearest-neighbour order, 1237x822, dataset intrinsics), first 43 views.
  The orbit is the ~2 deg tour; the NN tour's median step is 14.5 deg.

## Results

Raw output of `python3 docs/sat-depth-cull/aggregate.py <out>` (per-view JSON in `evidence/`):

```
cost per pair (upper bound): 6.45 ms / 2,351,151 pairs = 2.743 ns/pair

## orbit: 30 views, mean pairs 2,351,151, median step 3.20 deg
tiles fully saturated: 14.922%; pixels never below eps: 41.439%
oracle: pairs behind tile saturation 1.611% = 0.1039 ms/view (of which inside the 512-period live-mask lag 0.542%)
| margin, slack | tiles cut/view | fallback rate | captured pairs | wasted pairs | net ms/view |
|---|---|---|---|---|---|
| m0.02_s0 | 124.7 | 15.3% | 0.563% (of behind 34.9%) | 4.289% | -0.2395 |
| m0.02_s1 | 33.4 | 0.8% | 0.008% (of behind 0.5%) | 0.032% | -0.0016 |
| m0.05_s0 | 124.7 | 11.2% | 0.368% (of behind 22.8%) | 2.892% | -0.1622 |
| m0.05_s1 | 33.4 | 0.8% | 0.006% (of behind 0.4%) | 0.032% | -0.0017 |
| m0.0_s0 | 124.7 | 51.9% | 0.522% (of behind 32.4%) | 15.166% | -0.9411 |
| m0.0_s1 | 33.4 | 0.9% | 0.010% (of behind 0.6%) | 0.033% | -0.0015 |

## train: 43 views, mean pairs 1,849,170, median step 14.55 deg
tiles fully saturated: 13.529%; pixels never below eps: 43.510%
oracle: pairs behind tile saturation 2.006% = 0.1018 ms/view (of which inside the 512-period live-mask lag 0.445%)
| margin, slack | tiles cut/view | fallback rate | captured pairs | wasted pairs | net ms/view |
|---|---|---|---|---|---|
| m0.02_s0 | 50.2 | 53.8% | 0.070% (of behind 3.4%) | 3.144% | -0.1556 |
| m0.02_s1 | 14.9 | 36.6% | 0.003% (of behind 0.1%) | 0.753% | -0.0379 |
| m0.05_s0 | 50.2 | 50.5% | 0.049% (of behind 2.4%) | 2.928% | -0.1457 |
| m0.05_s1 | 14.9 | 36.6% | 0.002% (of behind 0.1%) | 0.753% | -0.0380 |
| m0.0_s0 | 50.2 | 66.4% | 0.063% (of behind 3.1%) | 3.758% | -0.1870 |
| m0.0_s1 | 14.9 | 37.8% | 0.004% (of behind 0.2%) | 0.781% | -0.0394 |

best net -0.0015 ms/view (orbit m0.0_s1); oracle max 0.1039; verdict shelve
```

Columns: tiles cut/view = tiles given a predicted cut; fallback rate = share of those whose
true saturation came after the cut; captured = pairs dropped correctly (share of all pairs, and
of the oracle's pairs); wasted = pairs blended before a failed cut; net = (captured - wasted) x
cost per pair. "512-period lag" = oracle pairs that fall inside the device's live-mask refresh
window after saturation (modelled, period 512 per tile); these already cost the device only
the record overhead.

## What would change the verdict

- Nothing in the predictor can exceed the oracle line. Even a perfect predictor with free
  fallbacks is about half the kill line, at a cost model that overstates the saving.
- A per-pixel or per-microblock cut (not per tile) is a different lever: the device already does
  it in the blend (live mask), and blend-waste-t148 sized the remaining waste at 0.19 ms of
  dispatches on an older build.
- Scenes with a solid background (no sky) saturate more tiles; bicycle is the reference scene.

## Kill line

PLAN D3: net below 0.15 ms/view b2b -> shelve. Step 1 (device A/B) is not queued.
