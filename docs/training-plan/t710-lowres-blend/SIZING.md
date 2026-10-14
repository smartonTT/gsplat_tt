# T11 low-res blend path: sizing (task #710)

Date 2026-10-14 (PDT). CPU only: no device, no IRD. Inputs: D1 measurements
(`../d1-scaling/RESULTS.md`, `out/summary.json`, ttp/t691-d1-n-res-sweep @b91290ef) and per-tile
counts from `tilecounts.py` (25 of the 169 bicycle training cameras, every 7th; N = 0.5M, 1M,
1.5M, 3M, 6.13M by opacity-ranked pruning, the same subsets as D1; s = 1, 2, 4 =
1237x822, 618x411, 309x205). Model and fits: `model.py` -> `out/model.json`, `out/model.txt`;
skip-cost and list-length check: `sens2.py` -> `out/sens2.json`. Large inputs (`out/scene.npz`,
`out/tiles.pkl`) are not committed; `tilecounts.py` regenerates them (about 20 min, 1 core).

All ms/view below are **modelled** b2b forward ms/view on one p150 (120 cores), not measured.
"M" marks D1 measurements.

## Why low res is slow today

Blend = list scheduling of per-tile work over 120 cores. At 1/4 res there are 70 tiles for 120
cores, and one tile can hold 60-180k pairs; that tile alone sets the blend time. Model of today's
path vs D1 (b2b ms/view): 1M s4 6.93 vs 6.47 M; 0.5M s4 4.77 vs 4.33 M; 1M s2 3.96 vs 3.71 M;
3M s2 6.97 vs 7.29 M; full res 6.13M 6.48 vs 7.92 M (rms error over all 11 D1 points 0.55 ms).
With perfect balance the same work would take 1.13 ms (1M s4), not 6.5. So the slowdown is load
imbalance, not per-pair cost.

The model also reproduces D1's failures exactly: the heaviest tile list is 76k (1.5M s4), 120k
(3M s4), 181k (6.13M s4) and 73k (6.13M s2), all over the 65,472-record bucket; 1M s4 (60k) and
3M s2 (49k) fit, as D1 measured.

Fit: tile cost = 39.4 us per non-empty tile + 97.6 ns per 4x8 microblock a pair touches (pairs
and microblocks are nearly collinear; the fit puts all of it on microblocks). The per-tile fixed
cost is poorly pinned: refits with it fixed at 1, 5 and 20 us change the option numbers below by
under 0.1 ms and do not change the ranking.

## Options (ranked)

| rank | option | 1/2 res, 1M | 1/2 res, 3M | 1/2 res, 6.13M | 1/4 res, 0.5M | 1/4 res, 1M | 1/4 res, 3M | fixes bucket overflow |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **B1 adaptive split: heavy 32x32 tiles become 16x16 / 8x8 sub-tiles with their own sorted lists** | **1.85** | **2.90** | **3.53** | **1.09** | **1.43** | **2.40** | yes (max list 25k s2, 35k s4) |
| 2 | B1' same split, sub-tile cores walk the parent's list and skip misses (2 ns/skip) | 1.83 | 2.89 | fails | 1.07 | 1.43 | fails | no |
| 3 | B2 depth split: heavy tile's sorted list cut into k chunks on k cores, (C,T) merge | 1.77 | 2.84 | fails | 1.02 | 1.42 | fails | no |
| 4 | A8: 8x8 screen tiles everywhere | 4.25 | 5.27 | 5.88 | 1.51 | 1.82 | 2.73 | yes (max 35k) |
| 5 | A16: 16x16 screen tiles everywhere | 2.30 | 3.48 | 4.19 | 2.01 | 2.85 | 5.19 | s2 yes; s4 6.13M no (70k) |
| 6 | C: larger or segmented sort bucket only | 3.96 | 6.97 | 8.62 | 4.77 | 6.93 | 12.76 | yes, but no speedup |
| - | today (O0) | 3.71 M | 7.29 M | fails | 4.33 M | 6.47 M | fails | - |
| - | bound: today's work, perfect balance | 1.70 | 2.71 | 3.33 | 0.83 | 1.13 | 1.92 | - |

1.5M: s2 B1 2.17 (today 4.72 M), s4 B1 1.72 (today fails). Full res (s=1): A16 and A8 are
much worse (6.13M: 7.9 and 15.9 vs 6.5 model), so uniform small tiles only suit one resolution.
B1 picks its split per tile, needs no setting per resolution, and the model gives it about
-0.5 ms at full res 6.13M too (6.48 -> 5.95; unverified, the model is 1.4 ms low on that point).

Sensitivity: B1' depends on the skip cost; at 4 ns/skip 1/4 res 6.13M goes 2.96 -> 3.32 ms, at
1/2 res under 0.06 ms. B1 with own lists pays instead for 250-340 extra sort lists (about +0.03
to +0.06 ms with D1's sort cost per list).

## Risks per option

- **B1 (own sub-tile lists).** tile_assign has to emit sub-tile keys for heavy tiles. That needs
  per-tile counts before the key pass (a count pass already exists for buckets) and a split rule
  (a tile goes when its estimated cost passes total/120). The blend kernel has to handle a 16x16
  or 8x8 tile: it already works per 4x8 microblock with a microblock cull, so this is a microblock
  mask plus output addressing, but the writer and backward have to follow it. The claim order
  needs mixed tile sizes. Sub-tile lists above 8,192 (kBucketFit) take the slower non-L1 path; D1
  showed no extra cost for it (fitted over-8192 term = 0), but that is untested at 35k. More lists
  mean more sort work (modelled as small). The output should be bit-identical: each pixel blends
  the same records in the same order.
- **B1' (walk parent list).** Same kernel work, no tile_assign change, but the parent list still
  has to fit one bucket. It fails exactly where D1 fails, unless C is also built. Each sub-tile
  core also reads the whole parent list.
- **B2 (depth split).** Slightly best modelled speed, but the parent list still overflows (needs
  C), it needs a (C,T) merge step between cores, and in training the backward needs each chunk's
  start transmittance, so it adds a sync and a scan between cores. Floating-point order changes
  (chunks are composited and then merged), so it is not bit-identical to today. It still matches
  the math exactly, but the T0 oracle tolerance has to cover it.
- **A8 / A16.** Simplest (a constant), but each costs a lot at the resolutions it does not fit,
  and the fixed per-tile cost grows 4x/16x. A DashGaussian schedule would need it switched with
  resolution. A16 still overflows at 1/4 res 6.13M.
- **C (bucket).** Only removes the failure; speed stays where today's numbers are (worse than full
  res). Keep it only as a guard for scenes denser than bicycle.

Lossless rule (user msgs #384/#387): every option here is exact. Nothing is culled, reused or
approximated; each pixel still composites its full depth-sorted list. Each one still has to show
it: same md5 as today where today runs (B1, B1', A8, A16), the T0 gradient oracle on every
training step, and test-view PSNR/SSIM/LPIPS against published 3DGS/gsplat or LiteGS bicycle
numbers (labeled published).

## What this means for T11 (DashGaussian)

DashGaussian trains early steps at low res with fewer Gaussians, and raises both together. With
today's blend this is backwards: low res is slower per view than full res (1M: 6.47 ms at 1/4 res
vs 3.25 ms at full res) and fails above about 1.2M Gaussians at 1/4 res. With B1 the forward
falls to about 1.1-1.4 ms/view at 1/4 res (0.5-1M) and 1.8-2.9 ms/view at 1/2 res (1-3M), a 2-4.5x
cut at those points. The backward blend uses the same tile lists and the same scheduling, so it
should gain about as much (not modelled here).

## Recommended next device task

Build B1 (adaptive heavy-tile split with own sub-tile lists) on best-iter-229 on the p150
measurement box. First, a bit-exact check against today's path (md5) at 1M s4, 3M s2 and full
res. Then time the D1 grid points s4 {0.5M, 1M, 1.5M, 3M}, s2 {1M, 3M, 6.13M} and s1 6.13M
(30-view b2b, `ab_jit_check.py` on both arms, hero screenshot and diff for s1). Pass bar: 1M s4
at or under 2.0 ms and 3M s4 running. Then add the backward under the T0 oracle.
