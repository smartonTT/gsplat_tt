# T-A (task #533): exact ellipse-tile pair test, CPU model

Written 2026-10-12 (PDT) for lever 2 / B2 of docs/structural-plan/PLAN.md (c94c13b0).
Host model only, no device runs. Pair counts are modelled; ms figures are bounds or
estimates derived from #528's measured slopes, **not measurements**.

## Answer

| | removed pairs per view, M (median / min / max) | share of P | bound at 2.23 ms/M (median / min / max) |
|---|---|---|---|
| **exact_m** (image-identical rule) | **0.183** / 0.158 / 0.206 | 6.8 % (6.2-7.8) | **0.408** / 0.352 / 0.460 ms |
| exact (no margin) | 0.188 / 0.162 / 0.212 | 7.0 % | 0.420 / 0.361 / 0.473 ms |
| exact_m, well-conditioned gaussians only | 0.137 / 0.115 / 0.161 | 5.0 % | 0.306 ms |
| row_dev (pfwc's loose t, not-ok gaussians keep all tiles) | 0.116 / 0.099 / 0.136 | 4.4 % | 0.259 / 0.221 / 0.303 ms |

**Go/no-go for B2 (bound >= 0.3 ms/view): GO, but only just, and only for a tight test.**
The rule as the spec states it clears the bar at 0.41 ms. Two catches:

1. **The test has to be tight.** About 25 % of the removed pairs belong to ill-conditioned
   gaussians (64·det < a·c, the needles that PRECULL already leaves at 3 sigma). Leave
   those out and the bound is 0.31 ms. Also use pfwc's looser t (log2 bound + 0.25) and it
   falls to 0.26 ms, which is below the bar. So B2 must cover needles and use the band
   cull's threshold, 2 ln(op/floor) + 0.05.
2. **2.23 ms/M overstates what a dead pair costs.** Every removed pair is dead: blend never
   draws it. mat's band cull already gives it a zero microblock mask, and blend skips
   mask-0 records (alpha_blend_compute_mb.cpp:1304, 1484). The pair still costs k2
   (0.255 ms/M), sort (0.455 ms/M) and mat per record (1.42 ms per ~24.7k records per
   core, about 0.52 ms/M). That comes to about 1.15 ms/M, so the **realistic gross gain
   is about 0.21 ms/view** (estimate). The heaviest tile shrinks only 3.4 %
   (22,976 -> 22,185 pairs) against 6.8 % on average, so the slowest mat+blend cores gain
   less than the mean.

Net of the test cost (below), expect roughly 0.1-0.2 ms/view, about 1-2 % of 9.13 ms. That
passes the plan's stated gate. I would still rank B2 after B1 and B3 (see Recommendation).

## Method

`count.py` reuses the t140/t157 host model (`docs/precull-t140/estimate.py` projection and
`docs/precull-t157/analyze.py` `t_dev`/PRECULL_PC rect). For each of the 30 views in
`benchmarks/cameras_v2.json` order (hero, orb_00..orb_28) over `scenes/bicycle.ply`
(6.13 M gaussians, 1024², floor = 1/255), it:

- expands each visible gaussian into the tiles of its **PRECULL=2 pixel-centre rect**
  (r' = min(r, max(sqrt(t·cov) − 3/8, 1/8)) when ok, else the 3-sigma rect). This is the
  current default (vis_mode.h; project_pfwc_compute.cpp:649 at ad647d8e);
- for every such pair, takes the minimum m² = [u v]·Σ⁻¹·[u v]ᵀ over the tile's continuous
  pixel-centre box [32k + 0.5, 32k + 31.5]² (`box_min_m2`, exact: centre-in-box, else the
  min along the 4 edges). Max alpha over the box = op·exp(−m²/2). The box is a superset of
  the 1024 pixel centres, so the test is conservative;
- keeps the pair under each arm: `exact` m² <= 2 ln(op/floor); `exact_m` the same + 0.05
  (the band cull's own live test, so no drawn pixel changes); `row_dev` t157's device-shaped
  "row" arm.

Each pair is one 32-B record through k2, sort and mat+blend, so **pairs = blend records**
(the records mat+blend reads). Live records, those that draw at least one pixel, are the
same under every conservative arm. Removing a dead record changes no pixel.

**Calibration against the device.** The model's P under PRECULL=2 is +0.30 % from the
device's P in #528 (`VIEW_COUNTS` from docs/profile-t224/out1/run-counts.log @377b0ab7,
copied to `out/device_counts_t528.txt`), with a range of +0.18 to +0.50 % over the 30
views. M is +0.25 %. The hero max tile is 22,976 in the model and 22,979 on the device.

Reproduce (repo root, ~35 s per view, ~4.5 min for 30 with 4 processes):

    python3 docs/ellipse-pairs-tA/count.py all > docs/ellipse-pairs-tA/out/views.jsonl   # then add M_dev/P_dev
    python3 docs/ellipse-pairs-tA/summarize.py

Per-view tables: `out/summary.txt`. Raw output: `out/views.jsonl`.

## Extra per-pair test cost

Cheaper exact form: for a convex quadratic whose minimum (the mean) lies outside the box, the
box minimum lies on a face that faces the mean. Let u_e = clamp(0, u_lo, u_hi) and
v_e = clamp(0, v_lo, v_hi). Then

    min m² = min( Q(u_e, clamp(−(b'/c')·u_e, v_lo, v_hi)),  Q(clamp(−(b'/a')·v_e, u_lo, u_hi), v_e) )

This is exact: both points lie in the box, and each minimizes Q on its face. When the mean is
inside the box it gives 0.

SFPU instruction count per 32 pairs (one row of lanes):

| part | instr |
|---|---|
| tile corner → u_lo, v_lo, u_hi, v_hi (int→fp cast, MAD, add) | 6 |
| u_e, v_e (2 min/max each) | 4 |
| two candidates: clamp (mul + 2 min/max) + Q (2 mul, 2 MAD, 1 mul) | 2 × 8 = 16 |
| min, subtract T, sign → keep flag | 3 |
| SFPLOAD of a', 2b', c', −b'/c', −b'/a', T, 0.5−mx, 0.5−my, tx, ty; one SFPSTORE | 11 |
| **total** | **~40, about 1.3 SFPU cycles per pair** |

Cross-check against a measured SFPU kernel: mat's band cull (band_batch) takes 1.077 ms per
core for ~24.7k records with a 32-microblock mask each. That is ~59 cycles per record,
~1.8 cycles per record-microblock test, so **plan on 2-4 SFPU cycles per pair**. Over all
2.73 M PRECULL=2 pairs on 110 cores at 1.35 GHz, the SFPU part is 0.04-0.07 ms/view.

The catch is where the test runs. The data movers and TRISCs have no FPU: soft float costs
~85-90 cycles per op (t157; docs/OPTIMIZATION-PLAN.md #20). k2 today walks pairs as scalar
code on the TRISCs (k2_trisc_compute.cpp, `emit_pairs_diet_from`). So B2 has two routes:

- **SFPU in k2.** MATH runs the test on pair pages that UNPACK/PACK have filled with
  per-gaussian coefficients, then compacts. The scalar marshalling (gathering ~10 words per
  pair) costs more than the SFPU math. At ~10-20 scalar cycles per pair that is
  0.18-0.37 ms if not overlapped, which can eat the whole gain.
- **Scalar fixed-point in k2,** only on pairs of multi-tile gaussians (1.58 M of 2.73 M
  pairs; single-tile pairs need no test). An integer version of the formula above is
  ~10 multiplies plus clamps, ~40 cycles per pair. Spread over 3 TRISCs × 110 cores, about
  0.14 ms/view (estimate). Pfwc would also have to ship the conic in fixed point; the
  record has to grow, or k2 has to read it.

The same test must also run identically in the sort count and emit tile walks, or pairs go
missing (PLAN B2). Running it once in k2 and having sort consume k2's kept list avoids a
second and third copy.

## Recommendation

- Under the plan's rule (2.23 ms/M bound >= 0.3 ms), B2 is **GO**: the bound is 0.41 ms/view
  (0.35-0.46).
- The realistic net gain is ~0.1-0.2 ms/view (estimate): dead pairs skip blend already,
  and the test has a cost. So B2 ranks below B1 (0.6-1.0) and B3 (0.2-0.4) and should
  build after them.
- Requirements for B2 if built: the band-cull threshold (+0.05, not pfwc's +0.25 log2
  bound); cover the ill-conditioned gaussians (25 % of the gain; needs an fp32/fixed-point
  error analysis like test_precull.cpp, with teeth); one test site whose result sort
  reuses; a host unit test showing 0 live pairs lost. Default on only if md5 906e0435
  holds.
- Cheaper first step, if a host-side drop hook is easy to add: a device A/B that drops the same pairs host-side
  would show whether removing dead pairs moves mat+blend's critical cores at all. That
  checks the 1.15 vs 2.23 ms/M question with real data.
