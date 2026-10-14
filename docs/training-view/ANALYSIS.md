# Training-view analysis (task #668, answers user msg #353)

Analysis only: no device runs. Labels: **M** = measured on our hardware (cited), **E** = estimate
from measured numbers (arithmetic shown), **P** = published (cited; not measured by us),
**P?** = seen only in a search snippet, not checked against the full paper.

Baseline: iter 229, p100 bh-33, 110 cores, bicycle 30 views, b2b **8.063 ms/view (M)**.
Stage split (E, structural plan 4 §1, origin/ttp/t663-structural-plan-4 @149af7ac):
pfwc ~1.38, sort (K2) ~1.33, mat ~1.33, blend ~3.79, gaps ~0.2 ms/view.

## 1. What depends on consecutive views being similar

Short answer: **nothing that is kept reuses the previous frame's results.** The levers that did
(sort reuse, mat depth-order reuse) were shelved or kept viewer-only. Random views stay correct.
The b2b overlap needs the *next pose known in advance* and *fixed parameters between the
overlapped views*; that holds for a training batch and does not hold across optimizer steps.
Some orbit-tuned sizes cost speed (rarely), one fallback changes the image, and three caches
are **correctness bugs for training** (not for random views).

| item | where | random views | training (params change) |
|---|---|---|---|
| xview: pfwc of view N+1 queued behind blend N | render/host/xview.h:1-15, render.cpp:214-225, :282-300, run.py:755-758 | correct, same speed if the next pose is given; exact-key check (pose, scene ptrs, N, size, floor), a miss reruns pfwc (one wasted pfwc, never a wrong image) | within a batch at fixed params: fine. Across a step: the key holds host pointers, not contents, so an in-place param update **hits with stale outputs** unless the trainer drops the hint or bumps a version |
| B4a fold #552 (next view's K2 inside this view's blend) and L-A prefetch #628 | render.cpp:410-450, tile_assign_device.cpp:1239, env_config.h:234 | pose-keyed with cancel on miss (tile_assign_fold_cancel, render.cpp:214-225): correct, speed kept if the next pose is known | same as xview |
| sort reuse across views | docs/b4-overlap/PLAN.md:32 (#529) | shelved: 0% identical tiles even on the orbit | n/a |
| mat depth-order reuse | docs/b4-overlap/PLAN.md:32 (#539) | viewer-only, 0.2-0.4 ms at slow motion, 0 at bench motion; not in the bench path | n/a |
| per-frame claim order (blend) | render/host/blend_claim_order.h:5-10 | uses this view's counts only: not coherence-dependent | fine |
| LPT pfwc deal | render/host/pfwc_deal.h:5-15 | default off | n/a |
| tile capacity kTileCap 32768 -> kTileCapBig 65472 | sort_onelaunch_layout.h:32,37; sort_device.cpp:375-377, 3637 | speed only: one grow per process | fine |
| kBucketFit 8192, kOverflowL1Cap 16384 (max_tile_n ~26.5k on the orbit) | render/host/config.h:33, ~47 | speed only: a denser tile takes the slower overflow path | fine |
| pair ceiling 4,718,592 | env_config.h:63; growth tile_assign_device.cpp:551-563, 1633-1640 | speed only: regrow costs 9-12 ms once (task #85: 4 of 30 orbit views) | fine; size from the dataset cameras once |
| overflow retry with a coarser contrib floor | render.cpp:493-517, overflow_retry.h (8 retries, floor up to 0.5) | **changes that view's image**; fine for viewing | **biases gradients**: training must grow the cap instead |
| scene upload cache keyed by host pointer + N | pfwc_device.cpp:769-805; project_device.cpp:358-376; gather_visible_device.cpp:488-492 | fine (scene is fixed) | **correctness bug**: in-place param updates keep the pointer, so the device keeps the old means/cov/colour/opacity. Re-uploading costs ~58 ms per step (49.8 pack + 7.8 upload, frame 0, M) |
| static opacity prune at load (572k of 6.13M dropped, remapped order) | render/opacity_prune.py:1-20 (default on) | fine | **correctness bug**: opacities train, so the prune set goes stale; must be recomputed per densify step or done on device |
| b2b keep ring n_views+2 | render/run.py:352-363 | bench-only | n/a |
| md5 golden / tie order | hero_golden_8bit badge | random order does not change per-view output | training only needs PSNR-level agreement |

How smooth is the orbit (E, benchmarks/cameras_v2.json): mean step 0.061 units on a radius of
0.232, mean 3.5° per view, max 18°. Dataset cameras jump much more, but since nothing reuses
frame N-1, the expected random-order cost is only (a) bigger worst-case tiles or pair counts
from close-up views (one-time regrow, or the overflow path), and (b) a different survivor mix
per core. Not measured: no dataset cameras are tracked in the repo (no COLMAP or cameras.json
for bicycle). Follow-up F1 measures it.

## 3. Batched multi-view rendering on one chip (K = 2/4/8)

### What could be shared

| stage | per view today (E) | shared across K views? |
|---|---:|---|
| scene read (means 12 B, cov3d 24 B, opacity 4 B ≈ 40 B x 5.56M ≈ 222 MB, plus colour) | inside pfwc | **yes**: read once, project K times |
| projection, cov_cam, cov2d, conic, radii, visibility (pfwc TRISC) | ~1.3-1.7 | no: depends on the pose |
| record writes (~69 MB/view, docs/pfwc-rec32-t467) | inside pfwc | no |
| K2 counting scatter, mat, blend | ~6.5 | no: per view, per tile |
| launch / program boundaries | ≤0.38 left (#591) | yes: one launch drives K views |
| SH colour (degree 3, training) | 0 today (SH0 only, gsplat/loading_gaussians.py:18-56) | coefficients read once; evaluation per view |

pfwc does not wait for DRAM: reader input wait is ~0 (0.034 ms of 1.83, docs/pfwc-profile-t484,
M at iter 222). At 229 its end is set by the movers with TRISC close behind (t614). So reading
the scene once saves bandwidth that is mostly hidden today.

### Bound per K (E, upper bound)

saving per view ≤ (0.38 boundary + ~0.10 exposed read) x (1 - 1/K)

| K | saving ≤ (ms/view) | ms/view ≥ (from 8.063) | clears the 0.15 bar? |
|---|---:|---:|---|
| 2 | 0.24 | 7.82 | upper bound only; likely under once L-A's existing gain is counted |
| 4 | 0.36 | 7.70 | maybe |
| 8 | 0.42 | 7.64 | maybe |

Most of the boundary idle is already taken by L-A and the fold, so the real gain is likely
below these bounds. **For SH0 rendering, batching is not a speed lever** (under or near the bar).
It also costs L1: static CBs fill L1 today, and per-view state (records, counts) grows with K
while only the scene chunk is shared. DRAM capacity is fine (32 GB on p150, P).

### Where batching does pay: SH3 (training's colour model)

Training uses SH degree 3: 48 colour coefficients, 192 B fp32 per Gaussian (bicycle.ply has
f_rest_0..44). Reading them for the ~1.7M visible Gaussians (t484 survivors, M) is
1.7M x 192 B ≈ 326 MB ≈ **0.78 ms/view at 420 GB/s (E)**, on top of today's pipeline. The
evaluation math is small (~0.025 ms on SFPU, E). Batched, the read is paid once per K views:

| K | SH3 read per view (E) | saving vs K=1 |
|---|---:|---:|
| 1 | 0.78 | - |
| 2 | 0.39 | 0.39 |
| 4 | 0.20 | 0.58 |
| 8 | 0.10 | 0.68 |

So batching clears the 0.15 bar **for SH3 training**, not for today's SH0 bench. (bf16
coefficients would halve both rows.)

### N-chip data parallel

- Rendering (AFR, M): 2 chips 4.226 vs 8.395 ms/view b2b = **1.99x** (docs/l2-2chip/RESULT.md,
  #582, bh-50). Split-frame (#627, M) is worse for throughput (9.63 vs AFR 6.64 on mismatched
  cards). Rendering scales almost linearly.
- Training adds a gradient exchange per step. Dense fp32: 5.56M x 59 params x 4 B ≈ 1.3 GB (E).
  Sparse bf16 over the ~31% visible ≈ 200 MB (E): ~4 ms over PCIe at ~50 GB/s (E, link speed
  unverified), less over the p150's 4x QSFP-DD 800G card links (P, Tenstorrent p150 page).
  With K views per chip per step this is amortised over K views.
- Optimizer (Adam), dense: 328M params x ~28 B traffic ≈ 9.2 GB ≈ 22 ms per step at 420 GB/s
  (E). Per view that is 22/K; sparse Adam on visible Gaussians (Taming 3DGS, P) and bf16 states
  cut it 3-6x (E).

## 4. Forward rendering inside a full training step

Published ratios (all GPU):

| source | number | label |
|---|---|---|
| Faster-GS, arXiv 2602.09999 Tab. 4 | 4m10s for 30k iterations on RTX 4090 (3DGS 17m46s); 5090 2m43s | P |
| derived | 250 s / 30k ≈ **8.3 ms per training iteration** (all stages) | E from P |
| Faster-GS Tab. 6 | bicycle render 547.1 FPS (1.83 ms) | P |
| Faster-GS Tab. 3 | fusing optimizer into backward: 5m04s vs 5m31s; "parameter updates remain the main bottleneck" | P |
| Taming 3DGS, arXiv 2406.15643 | backward dominates, Adam close behind; SH is 48 of 59 attributes and most of the Adam cost; per-splat backward 4-5x; sparse Adam | P |
| Laplacian Frequency Hierarchies, arXiv 2609.03334 (FastGS base, garden) | 12.06 ms/iter: backward 56.5%, forward 18.2%, optimizer 14.3%, loss 3.5% | P? |
| DISTWAR, arXiv 2401.05345 | gradient computation 30% of training on a 4090 on average (up to 66%) | P? |
| 3DGRT particle backward, arXiv 2608.17298 | per-pixel backward 81-86% of the iteration | P? |

Rule of thumb (P?): one step ≈ 5-6x the forward, backward ≈ 2-3x the forward, optimizer
roughly equal to the forward. Our forward is 8.06 ms on one chip (M) vs 1.83 (P), so a TT
training step today would be roughly 40-50 ms per view (E), about 5x Faster-GS per iteration.
Densification and pruning change N during training, which breaks the static prune and any
N-keyed cache.

### What carries over to a TT backward

- **Carries over:** pfwc projection (the backward recomputes it), K2 tile lists and per-tile
  sorted order (the backward walks the same lists in reverse), microblock live masks (skip dead
  pairs), the blend claim order and LPT balance, the 32 B record layout, the device-resident
  scene and grow-only buffers.
- **Does not exist on TT:** float atomics through the NoC. GPU backwards use atomic adds per
  Gaussian. A TT backward should instead (a) write per-pair gradients in tile order,
  (b) scatter them to Gaussian order using the sort permutation (K2 already knows each Gaussian's
  pairs), (c) do a segmented reduce. This is deterministic, which also helps training repeat.
- **Memory (E):** params + grads + 2 Adam moments ≈ 59 x 4 x 4 = 944 B per Gaussian ≈ 5.7 GB at
  6M, which fits in 28-32 GB of device DRAM.
- **New work:** SH3 forward and backward, the loss (L1 + SSIM) on device, Adam on device.

## 5. Ranked training-oriented levers

Checked against structural plan 4 §6 exclusions and docs/b4-overlap/PLAN.md. None re-queues a
shelved or landed lever. Cross-view sort reuse (#529) and mat-order reuse on the bench (#539)
stay excluded. The scene prefilter fitted to the bench views stays excluded.

| # | lever | size | kind | check vs ledger |
|---|---|---|---|---|
| 1 | **Training-view bench**: bicycle dataset cameras, 1237x822, random order, b2b and latency, plus a pass that grows caps instead of using the floor ladder | measurement (needs the COLMAP/cameras.json, not in the repo) | enabler, required by msg #353 before the headline changes | new |
| 2 | **Device-resident params with a version-keyed cache** (replace the pointer key in pfwc/project/gather and xview.h) | correctness; avoids ~58 ms per step re-upload (M parts) | enabler | new |
| 3 | **SH3 forward with batched coefficient read** | adds ~0.78 ms/view at K=1, saves 0.39/0.58/0.68 at K=2/4/8 (E) | training | new (SH3 was out of scope for SH0 rendering) |
| 4 | **On-device Adam (sparse, bf16 moments), gradients summed over K views** | ~22 ms/step dense -> ~4-7 ms (E); per view /K | training | new |
| 5 | **TT backward prototype** with deterministic pair-gradient reduce | about 2-3x forward (P?) ≈ 16-24 ms/view today (E) | training | new |
| 6 | **2-chip data-parallel training**: gradient exchange over PCIe vs QSFP | ≤4 ms/step at 200 MB sparse bf16 over PCIe (E) | training | builds on AFR #582 (M 1.99x) |
| 7 | Remove the coarser-floor retry in training mode (grow caps only) | correctness of gradients; speed on rare views | training | new |
| 8 | K-batched pfwc for SH0 rendering | ≤0.24/0.36/0.42 ms/view at K=2/4/8 (E, upper bound) | rendering | new; likely under the bar, do only with #3 |

Rendering levers R1-R5 of structural plan 4 (blend body, mat, B2 ellipse, stealing) also cut
the training forward and carry into the backward's per-tile walk; they stay first in line for
the render headline.

## Sources in this repo

docs/structural-plan-4/PLAN.md (on origin/ttp/t663-structural-plan-4), docs/pfwc-profile-t484/README.md,
docs/l2-2chip/RESULT.md, docs/split-frame-2chip/device/RESULTS.md (on origin/ttp/t627-*),
docs/b4-overlap/PLAN.md, docs/pfwc-rec32-t467, benchmarks/cameras_v2.json, and the code lines
cited in §1.
