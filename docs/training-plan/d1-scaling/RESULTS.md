# D1 (#691): forward N x resolution sweep on the 169 training views (p150)

Measured 2026-10-14 00:37-00:57 PDT on bh-49 board 0 (Blackhole **p150a**, IRD 134115, the #687
measurement box), under `ttp lock p100`, synced with opt/sync_remote.sh from this worktree.
Code: best-iter-229 (3702fbb5). Cameras: benchmarks/cameras_train_nn.json (169 train views,
1237x822; 1/2 and 1/4 scale the intrinsics). Workload: importance-pruned proxies of bicycle.ply
(6,131,954 Gaussians), ranked by summed alpha*T over the 169 views (d1prep.py, makeproxy.py).
The proxies are a **timing workload only**, not a lever and not a quality claim.

M = measured here. E = estimate. b2b = render/run.py --back-to-back, 6 passes, median per pass;
two rounds per config, spread <= 0.002 ms. Latency and stage split are one round, latency mode.
Pairs are reference-CPU tile-Gaussian pairs per view (mean over the 169 views).

## Device sanity check
Hero view on device (opt/metal-screenshots/t691-229-p150/hero.png, diff hero_diff10.png):
**PSNR 42.85 dB vs benchmarks/reference_v2/hero.png** (bar 42.4). The x10 diff shows only edge
residue on the spokes, frame and leaves; no tile seams or blocks (checked by eye).
30-view sweep md5 84695855, 0 of 30 views differ between passes.

## Table (M)
| N | res | tiles | pairs/view | b2b ms/view (r1, r2) | latency | sort | blend | other |
|---:|---|---:|---:|---|---:|---:|---:|---:|
| 6.13M | 1237x822 | 1014 | 1.949M | **7.918** (7.917, 7.918) | 7.871 | 0.388 | 7.238 | 0.218 |
| 6.13M | 618x411 | 260 | 1.647M | fails (sort bucket overflow) | - | | | |
| 6.13M | 309x205 | 70 | 1.530M | fails (sort bucket overflow) | - | | | |
| 3.00M | 1237x822 | 1014 | 1.376M | 5.758 | 5.677 | 0.393 | 5.069 | 0.189 |
| 3.00M | 618x411 | 260 | 1.121M | 7.288 | 7.237 | 0.124 | 6.886 | 0.204 |
| 3.00M | 309x205 | 70 | 1.022M | fails (sort bucket overflow) | - | | | |
| 1.50M | 1237x822 | 1014 | 0.877M | 3.934 | 3.856 | 0.379 | 3.270 | 0.182 |
| 1.50M | 618x411 | 260 | 0.689M | 4.723 | 4.659 | 0.129 | 4.315 | 0.191 |
| 1.50M | 309x205 | 70 | 0.615M | fails (sort bucket overflow) | - | | | |
| 1.00M | 1237x822 | 1014 | 0.668M | 3.245 | 3.166 | 0.378 | 2.584 | 0.180 |
| 1.00M | 618x411 | 260 | 0.510M | 3.708 | 3.630 | 0.118 | 3.309 | 0.179 |
| 1.00M | 309x205 | 70 | 0.447M | 6.470 | 6.403 | 0.063 | 6.134 | 0.182 |
| 0.50M | 1237x822 | 1014 | 0.418M | 2.437 | 2.375 | 0.388 | 1.778 | 0.183 |
| 0.50M | 618x411 | 260 | 0.300M | 2.560 | 2.493 | 0.126 | 2.168 | 0.176 |
| 0.50M | 309x205 | 70 | 0.253M | 4.331 | 4.264 | 0.065 | 3.996 | 0.180 |

"other" = view total minus sort and blend (project, tile_assign, cross-view and the rest; pfwc runs inside project/xview).
Full table with all stage columns: out/analyze.md; raw numbers out/summary.json; per-run logs out/logs/.

## Fits (M, least squares over the rows above)
| fit | floor ms | ns per pair | ns per Gaussian | rms / max residual ms |
|---|---:|---:|---:|---|
| full res: floor + slope*pairs | 0.86 | 3.60 | - | 0.058 / 0.079 |
| full res: a + b*pairs + c*N | 1.04 | 3.12 | 0.13 | 0.026 / 0.039 |
| 1/2 res: floor + slope*pairs | 0.79 | 5.78 | - | 0.034 / 0.041 |
| 1/2 res: a + b*pairs + c*N | 1.00 | 4.50 | 0.41 | 0.002 / 0.003 |
| 1/4 res (2 points, exact line) | 1.54 | 11.0 | - | - |
| all resolutions pooled | 2.70 | 2.63 | - | 1.21 / 2.59 (does not fit) |

## Findings
1. **Fixed floor ~0.9-1.0 ms/view at full res** (sort ~0.38, other ~0.18, the rest is blend's
   per-view fixed cost). Everything above it is per-pair blend work at ~3.1-3.6 ns per pair.
   Blend is 92% of forward latency on the full model (7.24 of 7.87 ms).
2. **Pair ratio of a pruned proxy at N = 1.2M is ~0.39, not the 0.3 that PLAN-single-chip §4
   assumed** (pairs interpolated between 1M and 1.5M: 0.752M vs 1.949M). Forward b2b at 1.2M is
   **~3.5 ms** (E, linear between the 1M and 1.5M rows), not ~2.4. Pruned proxies keep the
   original splat sizes; a model trained to 1.2M usually has larger splats, so its pair ratio is
   more likely higher than lower (E).
3. **Lower resolution is slower per view, not faster.** At the same N, 1/2 res costs 5-27% more
   and 1/4 res 78-100% more than full res, though pairs fall 15-28% (1/2) and 33-39% (1/4). Sort gets cheaper
   (0.39 -> 0.12 -> 0.06 ms, it scales with tiles) but blend's per-pair cost rises 3.6 -> 5.8 ->
   11 ns. The work is split per 32x32 screen tile: at 1/2 res there are 260 tiles (about 2 per
   core) and at 1/4 res 70 tiles for ~110+ cores, so cores sit idle and each dense tile runs
   serially. A simple per-core load model (pairs / min(tiles, cores)) does not fit either
   (rms ~1 ms), so the slowdown is more than the core count alone (deeper per-tile lists,
   uneven tiles; not separated here).
4. **Hard limit: a tile's pair list must fit the 65,472-record sort bucket.** The full model
   fails at 1/2 and 1/4 res, and 3M and 1.5M fail at 1/4 res (tile 25 holds 67-73k records; the
   contrib-floor retry ladder runs out and the view errors out). Training at reduced resolution
   on this pipeline needs larger buckets or a split of heavy tiles, whatever the speed.

## What this changes in the plan
- §4 row "forward at 1.2M": ~2.4 -> **~3.5 ms (p150, M-interpolated)**; blend backward scaled by
  the measured blend ratio 0.395 (2.86 / 7.24 ms): ~3.4-4.0 ms, not 2.6-3.1. Final-N step at
  1.2M: **~9-10 ms (E)**, not 7-8.
- T11 (DashGaussian resolution schedule) cannot bank on cheaper low-res steps with today's
  tile split: low res is slower and overflows. It needs a low-res blend path first (smaller
  screen tiles, e.g. 16x16, or heavy tiles split across cores), sized as a separate lever.
- The per-pair blend cost (3.1-3.6 ns, forward) is still the lever that decides 1-2 ms; it
  counts double because the backward walks the same pairs.

## Not done / caveats
- The JIT kernel listing for this run came back empty (drive.sh lists
  /localdev/smarton/.cache/ttmc-t691 but remote.sh put the cache under its own $C path), so it
  was not compared to #687's listing. This sweep is a single build at one commit, not an A/B, so
  opt/ab_jit_check.py's A/B gate does not apply; all 11 good configs ran the same binary.
- out/counts.npz and out/rank.npy (106 MB, regenerable with d1prep.py) are not committed.
