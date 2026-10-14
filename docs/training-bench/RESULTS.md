# Training-view bench (task #667) — results

Measured 2026-10-13 20:13–20:50 PDT on bh-33 (p100 measurement box, `MESH_DEVICE=P100`, one
`ttp lock p100` session), code = best-iter-229 + camera files + run.py intrinsics option
(`0f372f52`; kernels and host pipeline unchanged from b9171bf9). Remote tree synced by
`opt/sync_remote.sh` (repo-scoped, one-way), stamp `sha=0f372f52 so_md5=9785034f`
(evidence/sync.log), so the run did not rely on devsync.

## Camera sets (all from the public Mip-NeRF360 bicycle COLMAP model, `scripts/training_cameras.py --seed 0`)

| file | views | size | intrinsics |
|---|---|---|---|
| `benchmarks/cameras_v2.json` (orbit, headline control) | 30 | 1024x1024 | fov 50 |
| `benchmarks/cameras_orbit_1237.json` (same orbit) | 30 | 1237x822 | fov 50 on the long side (fx 1326) |
| `benchmarks/cameras_train_shuffled.json` (train, seed-0 random order; reference) | 169 | 1237x822 | dataset fx 1163 / fy 1158 |
| `benchmarks/cameras_train_nn.json` (same train views, greedy NN tour over position + view dir) | 169 | 1237x822 | dataset |
| `benchmarks/cameras_test.json` (every 8th image, Mip-NeRF360 test split) | 25 | 1237x822 | dataset |

Hero check: `_DSC8679` from COLMAP vs the cameras_v2 hero c2w: max abs diff 8.7e-08.
`render/run.py`: a camera file's optional `"intrinsics"` {fx, fy, cx, cy} overrides the fov-derived K.

## b2b ms/view (headline mode: `render/run.py --back-to-back --no-ref`)

4 rounds, arm order rotated each round (plus an unscored round 0). Orbit arms: 20 passes per
round; train arms: 6 passes of 169 views. Value = median of the 4 round medians.

| arm | views | b2b ms/view | FPS | round min..max | raw md5 (all rounds) |
|---|---|---|---|---|---|
| orbit 1024x1024 (headline control) | 30 | **8.059** | 124.1 | 8.056..8.063 | 1d372b94 |
| orbit 1237x822 | 30 | **9.776** | 102.3 | 9.773..9.777 | abffa6f4 |
| train, shuffled (seed 0) | 169 | **8.575** | 116.6 | 8.572..8.576 | 954bb1cc |
| train, NN-ordered | 169 | **8.565** | 116.8 | 8.564..8.565 | ea105357 |
| test split (every 8th) | 25 | **9.208** | 108.6 | 9.207..9.211 | fce3d30c |

Published reference (not measured): Faster-GS inference renderer, RTX 4090, bicycle 1237x822,
1.83 ms/view (547 FPS). At the same size our train views are 4.7x and the test split 5.0x slower.

The rounds span 0.001–0.004 ms per arm, far below the ~0.1 ms noise floor, so every gap
between arms above that floor is real.

### Findings
- **View order costs nothing today.** Shuffled vs NN-ordered: 8.575 vs 8.565 ms (0.010 ms,
  noise). The pipeline has no cross-view reuse that benefits from coherence yet: every view
  recomputes pfwc/sort/blend from scratch. That makes the NN tour the natural place for
  coherence levers (reused sort order, incremental culling). Today it gives no gain.
- **Training views cost 0.5 ms/view more than the headline orbit** (8.57 vs 8.06), but the orbit is
  a different size and FOV. At the same 1237x822 size the dataset's training views are 1.2 ms
  *cheaper* than the orbit (8.57 vs 9.78). Likely cause: the 1237 orbit uses fov 50 on the long
  side (fx 1326, vertical FOV 34°), which is narrower than the dataset's (fx 1163). The same
  Gaussians therefore cover more pixels and more tiles per Gaussian. The headline orbit is a
  conservative stand-in for training throughput.
- **The test split is 0.6 ms slower than train** (9.21 vs 8.57): of 25 views, several are close-ups
  of the bike/bench (_DSC8679 hero, _DSC8687, _DSC8695), and the blend stage dominates them.
- **One tile overflow per process on the dataset cameras.** One view in every train/test run
  (e.g. _DSC8850, _DSC8830, _DSC8695: a tile with ~34k padded records > 32768 bucket) makes
  the sort grow its bucket to 65472 once and re-render that view at the same floor (1/255). That
  costs about 4–5 ms (`project_gather_wait`) once per process. The warm-up pass absorbs it in
  b2b (all passes give identical md5). Every later view then runs with the larger bucket. Whether
  that bucket size changes per-view cost was not isolated here (see follow-up). The orbit never
  overflows.

## Per-view stage spread (latency mode, `GSPLAT_PER_VIEW_STAGES=1`, first view dropped)

Latency-mode numbers are for spread and attribution only, not for conclusions (charter).

| arm | wall min / med / max ms | blend min / med / max | sort min / med / max |
|---|---|---|---|
| orbit (29) | 5.42 / 7.97 / 9.07 | 4.34 / 6.72 / 7.79 | 0.78 / 0.80 / 1.08 |
| train shuffled (168) | 5.96 / 8.37 / 14.73* | 4.51 / 7.11 / 10.20 | 0.67 / 0.81 / 1.25 |
| train NN (168) | 5.95 / 8.34 / 14.96* | 4.70 / 7.08 / 10.19 | 0.79 / 0.82 / 1.29 |
| test (24) | 6.95 / 8.98 / 16.56* | 5.86 / 7.68 / 9.37 | 0.78 / 0.81 / 1.53 |

\* the max is the one-off bucket-grow view above (gather wait 4–5 ms). Without it, the slowest
train views are _DSC8693/8681/8685/8683 at 11.0–11.5 ms. Blend is ~84% of the wall time in
every arm; sort is flat at ~0.8 ms.

Slowest vs fastest views and their in-frustum Gaussian count (CPU count over bicycle.ply,
6.1M Gaussians, ±15% guard band): slow train views hold 2.1–2.4M Gaussians in frustum, fast
ones 1.48–1.58M. For the orbit it is 2.47–2.52M (slow) vs 1.88–1.91M (fast). Per-view cost
follows the in-frustum count (more Gaussians means more tile pairs to blend). The exceptions are
the bucket-grow outliers (_DSC8850: 1.48M), which come from one dense tile, not the total.
Tile-pair counts per view are not logged by the per-view profiler; the in-frustum count stands
in for them. Full lists: evidence/analyze.txt.

## Correctness and screenshots

- Orbit golden check (opt/ttw/screenshot.sh, device hero + 30-view sweep at 0f372f52):
  `golden_match=true max_lsb_vs_golden=0`, sweep md5 077caaba, 0 of 30 views differ. Hero PSNR vs
  the reference render (benchmarks/reference_v2/hero.png) = 42.85 dB.
  Screenshot: opt/metal-screenshots/t667-229/hero.png (+ hero_diff10.png); looked at: no tile
  artifacts.
- 3 random training views rendered on device (first 3 of the seed-0 shuffle), saved in
  evidence/. Each was looked at full size: no tile seams, no blocky or missing tiles.
  PSNR **vs dataset photo** (images_4 JPG, 1237x822; SH degree 0 render, so view-dependent colour
  is missing. This is not a reference-render comparison):
  _DSC8815 21.37 dB, _DSC8843 25.06 dB, _DSC8837 23.37 dB.
- JIT hashes (opt/ab_jit_check.py, t656 tool): all 15 kernels have the same hash set in all 5 arms.
  The tool reports `A/A` / JIT-HASH FAIL, which is the correct state here. The arms differ only
  in camera files, not in code, so this confirms they ran identical binaries.

## Reproduce

    ttp detach t667 -- docs/training-bench/drive.sh <rev>     # one ttp lock p100 session
    python3 docs/training-bench/analyze.py --images <mipnerf360/bicycle/images_4> [--ply scenes/bicycle.ply]

The dataset images and COLMAP files are not in git. Only the derived camera JSON files are committed.
