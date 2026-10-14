# Training quality bar — Mip-NeRF360 bicycle (task #692)

All numbers are **published**, not measured. Rule (user msgs #384/#387): match 3DGS/gsplat OR LiteGS
at the same resolution and iteration count; either one is enough.

## The bar

| Reference | Scope | PSNR | SSIM | LPIPS | Res | Iter | Gaussians | Train time / GPU | Source |
|---|---|---|---|---|---|---|---|---|---|
| 3DGS (Kerbl 2023) | bicycle | 25.246 | 0.771 | 0.205 (VGG) | images_4 (1237x822) | 30k | not published per scene | not published per scene | appendix Tables 4-6 (SSIM/PSNR/LPIPS), https://arxiv.org/html/2308.04079v1 (verified #695) |
| 3DGS (Kerbl 2023) | bicycle | 23.604 | 0.675 | 0.318 (VGG) | images_4 | 7k | — | — | same tables |
| 3DGS (Kerbl 2023) | M360 avg (9 scenes) | 27.21 | 0.815 | 0.214 (VGG) | outdoor /4, indoor /2 | 30k | — | 41m33s, A6000 | paper Table 1 |
| gsplat | M360 avg (7 scenes) | 28.95 | 0.870 | 0.135 (torchmetrics, net not stated on page) | not stated on page | 30k | not published | 35m49s, TITAN RTX | https://docs.gsplat.studio/main/tests/eval.html |
| LiteGS | bicycle | 25.10 | 0.7467 | 0.2531 (VGG, torchmetrics; example_metrics.py) | images_4 (full_eval.py) | 30k (arguments.py `iterations=30000`, run as int(30000/len(train)) epochs) | 1,000,000 (capped) | 33 s RTX 4090 / 50 s RTX 3090 | https://github.com/MooreThreads/LiteGS README |
| LiteGS | M360 avg | 27.59 | 0.8131 | 0.2296 | — | — | — | 40 s 4090 / 64 s 3090 | same |

Split: every 8th image held out as test (3DGS/gsplat `--eval`). LiteGS verified (#695): example_metrics.py uses `idx % 8 != 0` for train unless train_test_split.json exists; full_eval.py passes `--eval --sh_degree 3`.
gsplat publishes no bicycle-only number; its 7-scene average excludes flowers/treehill, so it is not comparable to the 9-scene 3DGS average.
Verified 2026-10-13 (#695): 3DGS bicycle values match appendix Tables 4-6 of arXiv 2308.04079 (PSNR 25.246). LiteGS settings read from github.com/MooreThreads/LiteGS @004b9521 (full_eval.py, litegs/arguments.py, example_metrics.py, litegs/training/trainer.py). Caveats: LiteGS's README table is assumed to come from full_eval.py at target_primitives=1M; full_eval.py also sweeps up to 5,987,095 for bicycle. LiteGS rounds 30k iterations down to whole epochs (int(30000/N_train)*N_train, so slightly under 30k).

**Target for us:** bicycle, images_4 1237x822, 30k iter, test views = every 8th:
PSNR >= 25.10, SSIM >= 0.747, LPIPS <= 0.253 (LiteGS bar). Stretch: 25.25 / 0.771 / 0.205 (3DGS).

**Easier reference: LiteGS.** Lower on all three metrics and capped at 1M Gaussians (3DGS grows ~6M on bicycle),
which also keeps memory within reach of p150 DRAM/SRAM. Settings confirmed from its scripts: images_4, ~30k iter, VGG LPIPS, every-8th test split, SH degree 3.

**Training time to beat (published):** LiteGS 33 s on RTX 4090 for bicycle (~1.1 ms/iter if 30k iter).
Our 1-2 ms/view target sits in the same range.

## LiteGS ideas for PLAN-single-chip.md

1. **Spatial clustering into 1024-point chunks + chunk frustum cull** — lossless only if the chunk cull is
   conservative (a chunk is dropped only when its bounding volume is fully outside); per-Gaussian cull stays exact.
2. **Visible-primitive compaction** (gather visible params into contiguous memory before raster/backward) — lossless (pure data movement).
3. **Sparse gradient / sparse Adam on visible primitives only** — lossless for the gradient; the optimizer
   step differs from dense Adam (no moment decay on invisible points), so it changes the training trajectory.
   Counts only if the quality bar still holds; flag as not bit-equal to 3DGS.
4. **Fused, modular kernels** (project+cull, raster fwd, raster bwd as separate fused steps) — lossless.
5. **Gaussian count cap (1M) with LiteGS densification** — not lossless vs 3DGS, but it is the LiteGS reference
   config itself, so it is allowed when matching the LiteGS bar.

No LiteGS stale-sort or approximate-cull trick was found; any sort reuse on our side must stay exact (#384).
