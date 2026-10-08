# t439: weighted (LPT) pfwc tile deal at ETH 12x10 on p150: shelved

**Verdict: shelved, off by default.** `GSPLAT_TT_PFWC_DEAL=lpt` evens out the pfwc cores
(max/mean core end 1.144 -> 1.096) but the mean core gets slower by about as much, so the pfwc
window moves only -0.018 ms traced and the untraced total does not move (+0.011 ms/view, noise).
The gate was -0.1 ms/view untraced. This was the last lever before the project stop line.

## Setup

- Box: bh-30 (p150b, aiclk 1350, tt_aus), under the viewer exception (no free measurement p150):
  viewer stopped 04:22:00Z, back at 04:25:30Z (SELFTEST 10.84 ms, localhost:8091 200).
- Tree: 31bf48c4 (this branch). ETH dispatch, compute grid 12x10, bicycle 1024x1024, 30 views.
- Arms: **S** = strided deal (default, `GSPLAT_TT_PFWC_DEAL=0`), **L** = `GSPLAT_TT_PFWC_DEAL=lpt`
  with `GSPLAT_TT_PFWC_DEAL_W=docs/pfwc-deal-t439/bicycle-hero-w.txt` (t437 CPU-model hero weights,
  `make_weights.py`; device per-tile counts were not dumped). Static equal-count LPT table, fed
  through the #433 `PFWC_TILE_LIST` kernel path; seg_base unchanged.
- 3 alternating untraced rounds (S L, L S, S L), then one Tracy capture per arm (views 0:10 + warmup,
  36 KB profiler overlay as #427). Scripts: `drive439.sh`, `bench439.sh`, `sync_bh30.sh`,
  `vstart439.sh`; analysis `ana439.py`. Raw logs and device CSVs in `out/`.

## Untraced (avg_frame_ms, 30 views)

| round | S | L | S blend | L blend |
|---|---|---|---|---|
| r1 | 7.763 | 7.759 | 5.264 | 5.252 |
| r2 | 7.789 | 7.752 | 5.248 | 5.203 |
| r3 | 7.835 | 7.911 | 5.357 | 5.248 |
| **mean** | **7.796** | **7.807** | 5.290 | 5.234 |

L - S = +0.011 ms/view (gate: -0.1). Project stage (holds pfwc): S 1.084, L 1.118 (r3-L 1.196
outlier; r1/r2 equal at 1.08). Blend is not slower (-0.056, within round-to-round noise of 0.1).

## Traced per-program windows (`out/ana439.txt`, 10 views)

| program | S window | L window | delta | S mean core end | L mean core end | S max/mean | L max/mean |
|---|---|---|---|---|---|---|---|
| pfwc | 1.874 | 1.856 | -0.018 (per view -0.162..+0.110) | 1.638 | 1.693 | 1.144 | 1.096 |
| blend (tile_blend_sfpu) | 5.195 | 5.189 | -0.006 | 5.062 | 5.060 | 1.026 | 1.026 |

S pfwc 1.874 matches #437's 1.898 for the same config. The LPT deal removed about a third of the
imbalance (max/mean 1.144 -> 1.096), as #437 modeled, but the average core finished 0.055 ms
later. Likely cause (not measured): under the strided deal all 120 cores read neighbouring tiles
at each step, while the LPT order scatters each step's reads across DRAM; the tile-list read
itself is 50 u32 per core. Whatever the cause, it ate the modeled +0.10 to 0.19 gain.

## Correctness

- S: list md5 39d84b28 in all 3 rounds, `MD5_GOLDEN_OK grid 12x10 (30/30 views)`.
- L: list md5 e1ecd879 in all 3 rounds (stable). It differs from S because the compacted gid order
  changes, which flips depth-tie order in blend: hero L vs S differs in 280 pixels, max 7 LSB
  (PSNR 83.7 dB, `out/hero-L-vs-S_diff10.png`). No per-grid golden was added since L stays off.
- Hero (device, bh-30, 31bf48c4, L config, `out/hero-L.png`): PSNR vs the reference render
  benchmarks/reference_v2/hero.png = 42.51 dB (S: 42.51 dB). Diff `out/hero-L_diff10.png`
  (|hero - ref| x 10). Looked at both: no tile seams, no blocky or empty tiles.
