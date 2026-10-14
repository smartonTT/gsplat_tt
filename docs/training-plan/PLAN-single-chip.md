# Single-p150 trainer: plan to 1-2 ms per training view (task #686, user msgs #380, #382, #384)

2026-10-13 PDT. A plan only: no device runs, no reservation, no lock. It replaces the multi-chip
parts of PLAN.md (#672) and keeps its step-time model where still valid. Multi-chip training
(T9) is parked (msg #380).

Labels: **M** measured on our hardware (cited), **E** estimate from measured numbers (basis
shown), **P** published GPU number (not measured by us), **P?** from a secondary table, README or
search snippet, not checked against the full paper. Every lever is marked **exact** (same
result as the reference computation, checked against the T0 oracle), **algorithmic** (changes the
training recipe the way published trainers do; allowed only behind the quality-parity gate Q1),
or **lossy** (an approximation of the computation; dropped, msg #384).

Timing box (msg #382): all training timing is taken on the p150 measurement box that the p1
task is reserving now (single free p150 board, not bh-30). Numbers taken on bh-33 are labelled
**p100 (provisional)**. Every number from earlier tasks quoted below is p100.

## 1. Resolution: what the other trainers use for bicycle

| trainer | bicycle training/eval images | source |
|---|---|---|
| 3DGS (INRIA) | `images_4` for the Mip-NeRF360 outdoor scenes (bicycle, flowers, garden, stump, treehill), `images_2` indoor; every 8th image held out as test; 30k iterations | graphdeco-inria/gaussian-splatting `full_eval.py` (P); the same rule is stated by Mini-Splatting's README (`-i images_4` for 360 outdoor) and by later papers (P?) |
| gsplat (nerfstudio) | `--data_factor 4` outdoor, 2 indoor (benchmark scripts) | gsplat examples/benchmarks (P?, quoted via Matryoshka GS) |
| Faster-GS, Taming 3DGS, DashGaussian, FastGS, LiteGS | the 3DGS protocol (they compare to 3DGS per scene on the same split) | their papers/tables (P?) |

`images_4` for bicycle is 1237x822 (194 images; scenes/mipnerf360/bicycle/images_4, msg #364).
**Our training-view bench already matches**: benchmarks/cameras_train_nn.json and
cameras_train_shuffled.json are the 169 non-test views at 1237x822 with the dataset intrinsics
(fx 1163, fy 1158), cameras_test.json is the every-8th split (25 views) (#667 RESULTS.md, M).
Its forward b2b is 8.565 ms/view NN-ordered, 8.575 shuffled (p100 provisional, M).

## 2. Published GPU training references, bicycle, 1237x822, 30k iterations

ms/iteration = run time / 30k; one iteration = one training view (batch 1), so ms/iter is
directly comparable to our "ms per training view". All labelled published, not measured.

| trainer | GPU | bicycle run | ms/iter | final N | test PSNR | label |
|---|---|---|---:|---:|---:|---|
| 3DGS (INRIA) | 4090 | 27.97 min | 55.9 | 4.71M | 25.14 | P (FastGS Tab. 17) |
| DashGaussian (+Taming) | 4090 | 9.93 min | 19.9 | 4.70M | 25.31 | P (FastGS Tab. 17) |
| Taming 3DGS, budget mode | ? | 485 s | 16.2 | 0.81M | 24.78 | P? (3DGS-LM table) |
| Faster-GS | 4090 | 4m10s, Mip-NeRF360 average | 8.3 | - | - | P (arXiv 2602.09999 Tab. 4) |
| FastGS | 4090 | 1.92 min | 3.84 | 0.54M | 24.84 | P (FastGS Tab. 17) |
| LiteGS | 4090 | 33 s (3090: 50 s) | 1.1 | ~1M | 25.10 | P? (MooreThreads/LiteGS README; iteration count not checked) |

The 1-2 ms/view target sits at LiteGS on a 4090 and 2-4x under FastGS and Faster-GS. The fast
trainers get there mostly by training **fewer Gaussians** (0.5-1M instead of ~5M) and fewer
pixels early, not only by faster kernels. Note the quality column: FastGS and Taming's budget
mode land 0.3 dB under 3DGS on bicycle; LiteGS holds 25.10.

## 3. Quality-parity gate Q1 (msg #384: required before any training ms/view is called on target)

An end-to-end training run on the device, bicycle `images_4`, 169 train views, 30k iterations,
scored on the 25 test views (every 8th) with the 3DGS evaluation code's metrics (PSNR, SSIM,
LPIPS-VGG computed on the Mac from the device renders). Pass: PSNR >= 25.04 dB (3DGS 25.14 on a
4090, P, minus 0.1), SSIM and LPIPS no worse than the 3DGS bicycle row of the INRIA paper's
per-scene table (the gate task copies the exact values and cites the table; P). Plus the T0
oracle check on a sampled crop every N steps of the run (grads rel err <= 1e-2, Adam rel 1e-3).
Until Q1 passes, a training ms/view on the dashboard is shown as "step time, quality not yet
checked". Q1 decides every **algorithmic** lever below; a lever that fails it is dropped.

## 4. Where a training step's time goes (E, one view, K = 1)

Corrections to PLAN.md §1 from data that came in after it:
- DRAM streaming on 110 cores is **215 GB/s read, 260 GB/s write** (docs/hw-ceilings.md, M,
  p100), not the 420 GB/s PLAN.md used: param and Adam passes cost about twice its estimate.
- The final image goes to the backward as **fp32** (12 B/px, ~12 MB at 1237x822) and pair records
  stay fp32: bf16/u8 failed the oracle by orders of magnitude (T1 doc §3, M on the Mac).
- Blend is ~84% of forward wall time on the training views (#667, latency mode, M p100).

| stage | at N = 5.56M (today's bicycle.ply) | at N = 1.2M (parity-sized model, see note) | basis |
|---|---:|---:|---|
| forward (pfwc, sort, mat, blend) | 8.57 | ~2.4 | M p100 at 5.56M; at 1.2M: pfwc x N ratio, the rest x pair ratio 0.3 (E, Speedy-Splat's 10.6x fewer Gaussians gave 6.7x faster render, P) |
| loss L1 + D-SSIM fwd+bwd, fused | 0.2-0.4 | 0.2-0.4 | PLAN.md row d, resolution-bound not N-bound |
| blend backward | 8.7-10.2 | 2.6-3.1 | PLAN.md row e: 2.3-2.7x the blend part |
| pair-grad scatter + segmented reduce | 0.3-0.9 | 0.1-0.3 | PLAN.md rows g, h at 215 GB/s |
| projection + SH3 backward + Adam (visible rows, fp32 params, bf16 moments) | 7.6 | ~1.6 | 1.7M (0.36M) visible x 59 x 16 B / 215 GB/s |
| exposed pfwc at the step boundary | 1.4 | 0.3 | PLAN.md row k |
| **step at final N** | **~27-29** | **~7-8** | E |

Note on N: parity on bicycle needs about 1-1.5M Gaussians with a good densifier (LiteGS 25.10 dB
at ~1M, P?; Taming's budget curve flattens above ~1.4M on Mip-NeRF360, P). The pair ratio at
that N is the biggest unknown in this table; task D1 measures it.

Run average, not final N: a 30k run starts from SfM points (bicycle ~54k, E from the COLMAP
model) and reaches final N around iteration 15k. Averaged over the run the step is ~0.6x the
final-N step (E: linear growth to 15k, flat after), so **~4.5 ms/iter**; with a DashGaussian
resolution schedule (-45.7% average run time on 3DGS backbones, P) **~2.5-3 ms/iter** (E).

**Verdict on paper: 2.5-3 ms/iter run average at parity quality (E), on a p100-calibrated
model.** 1-2 ms needs the per-pair blend cost (forward and backward share it) to fall by about
2x more. That is the same per-pair work as the render headline's gap (8.06 vs 1.83 ms), so the
render levers R1-R5 of structural plan 4 count double in training. The p150's extra cores (vs
110 on the p100) help by an amount the p1 re-baseline will show.

## 5. Ideas from other trainers and renderers, sized for our pipeline

| idea (source) | what it does | class | size on our step (E unless marked) | verdict |
|---|---|---|---|---|
| Budgeted densification (Taming 3DGS, FastGS, LiteGS opacity-gradient-variance criterion) | grow to a target N with score-based sampling | algorithmic | the biggest single lever: 27-29 -> 7-8 ms at final N (§4) | **T8b**, behind Q1; size the budget for parity (~1-1.5M), not for speed |
| Resolution + primitive scheduling (DashGaussian) | low-res early iterations, N grows with resolution | algorithmic | -45.7% run time on GPU backbones (P); our cost per pixel is pair-bound, D1 measures the resolution slope | **T11**, behind Q1 |
| Sparse/visible-only Adam (Taming; INRIA repo `sparse_adam`) | update only visible rows | algorithmic (moments of invisible rows do not decay) | Adam on 31% of rows: 7.6 vs ~24 ms dense at 5.56M | **T2** with a dense-Adam flag for A/B on Q1 |
| Optimizer fused into the backward (Faster-GS Tab. 3) | one pass: segment reduce, projection/SH backward, Adam, write pfwc inputs | exact | removes grad write/read (~0.5 ms at 5.56M) and the 58 ms host re-pack | **T2** |
| Fused SSIM (Taming, gsplat) | one kernel for SSIM fwd+bwd | exact | 0.2-0.4 vs ~0.6 unfused | **T7** |
| Per-splat backward parallelism (Taming), warp-level reduce (LiteGS) | cut atomics contention | exact | our analogue is the 32-lane reduce of 9 terms per pair: FPU row-reduce, 1-2 ms at 5.56M | **T6** |
| Tight opacity-aware tile bounds (Speedy-Splat SnugBox/AccuTile; gsplat's opacity-aware radius) | fewer pairs, same pixels | exact (pairs dropped have alpha < 1/255 everywhere) | plan 4's B2 ellipse lever; counts twice in training | render lever R-B2 first; carries over |
| Morton-sorted cluster-cull-compact (LiteGS) | spatial order for culling | exact | **excluded**: Morton reorder + chunk cull was a net loss on our pipeline (+0.18..+0.41, #470/#479/#493, M) | not re-proposed |
| absgrad densification (gsplat), opacity-gradient variance (LiteGS) | better densify criterion: fewer Gaussians for equal PSNR | algorithmic | lowers the N needed for parity | option inside T8b |
| Primitive pruning (Speedy-Splat, Mini-Splatting) | prune 90% with fine-tuning | algorithmic | 10x fewer prims -> 6.7x render (P); quality drops on bicycle | option inside T8b, judged by Q1 |
| Levenberg-Marquardt (3DGS-LM) | fewer iterations, heavier steps | algorithmic | ~20% fewer GPU minutes (P?); large new solver | not ranked |
| Batch of K views per step (Grendel) | grads summed over K, sqrt(K) lr | algorithmic | -1.2 to -1.6 ms/view at K=4 at parity N (Adam, SH3 read, boundary / K) | **T5**, behind Q1 |
| StopThePop per-pixel sort | quality (popping), more sort work | exact but slower | no speed gain | not ranked |
| gsplat packed mode | memory layout for sparse visibility | exact | our Gaussian-major slots already do this | n/a |
| tt-splat order-independent matrix blend (community TT trainer) | not alpha compositing | lossy vs 3DGS (-3 dB on lego, its README) | - | dropped |
| bf16 pair records / bf16-u8 forward image into backward | halve bytes | lossy | failed the oracle (T1 doc §3, M) | dropped |
| Precision cuts, contrib floor > 1/255, coarser-floor overflow retry | fewer pairs | lossy | PSNR < 42.4 (#580) / biased grads | dropped; training grows caps (T3) |

## 6. Cross-view and cross-step reuse (all must be exact, msg #384)

Training views can be put in any order (msg #358); the NN tour (#667) is the natural order.
What can and cannot be shared, with params changing every step:

| lever | exact? | size | data needed |
|---|---|---|---|
| **Sort warm start**: the previous view's per-tile order as the start of an exact sort that is fast on nearly sorted input (run-aware merge, insertion fix-up of moved keys, or radix that skips sorted runs); output checked key for key against a full sort on every view | exact by construction | sort is 1.33 ms at 5.56M (~0.4 at parity N); upper bound ~2/3 of it if most keys stay in order. #529's "0% identical tiles" was about reusing the order as is, on the 12-degree orbit; a warm start needs near-sorted, not identical, input, so it is new evidence to collect, not a re-queue | **A1** (Mac): per-tile inversion count and run count between consecutive NN-tour views |
| **Batch-union pre-cull**: one compaction of the Gaussians that can be visible in the next K NN-tour views; each view's pfwc then walks only that subset and still culls exactly | exact with sparse Adam (rows outside the set get no gradient, so they do not move inside the batch); with dense Adam, exact if the set is grown by the bound on Adam's step (lr x step count) | pfwc 1.38 ms walks all 5.56M; in-frustum is 1.5-2.4M (#667, M CPU count). With |union of K| ~1.2x one view: ~0.6-0.8 ms/view at 5.56M, ~0.15 at parity N. Different from the killed per-view compaction (#596/#611): this one is paid once per K views | **A1**: union/single ratio for K = 2, 4, 8 |
| cov3d (from quat, scale) and SH3 coefficient read once per step, shared by K views | exact | K = 1: nothing to share. With T5: SH3 read /K (0.78 -> 0.20 at K=4, PLAN.md) | after T5 |
| Gaussians + Adam state resident in DRAM, never through the host | exact | removes 58 ms per step re-pack + upload (M parts) | T2/T3 |
| Ground-truth images resident in DRAM (169 x 1237 x 822 x 3 B = 516 MB u8) | exact | removes per-step H2D (~3 MB, ~0.1 ms, plus host latency) | M1 |
| Per-pixel forward state kept in L1 for the backward | exact | DRAM round trip is the fp32 image + dL/dimage, ~24 MB, ~0.1 ms: **under the bar** | not ranked |
| Params and moments in aggregate L1 across the run | exact | the param pass is ~1.6 ms at parity N and K=1, 0.4 at K=4; L1 is full of pipeline CBs | not ranked; revisit after M1 if the Adam pass dominates |
| Fused forward+backward per tile (records read once) | exact only with a pixel-local loss (L1); D-SSIM needs a 5-px halo from neighbour tiles | ~0.2-0.3 ms at 5.56M (re-read of 2.6M records); blend is SFPU-bound, not read-bound | not ranked |
| Reusing a view's sort order as is, approximate cull, stale caches | **lossy** | - | dropped (msg #384) |

Also needed for correctness, not speed (PLAN.md §2): the scene upload cache, the xview/fold/L-A
hint and the static opacity prune all go stale once params train. T3 fixes them.

## 7. Shortest path to a running single-chip trainer (M1)

Goal: a concrete workload to measure, as early as possible. Fine-tune from the trained
bicycle.ply with densification off, SH0, L1 loss first (D-SSIM when T7 lands), K = 1, NN tour.

1. **#685** (running): T1 device blend backward. Not duplicated here.
2. **T3** training mode: device buffers are the master copy (version-keyed caches, #671), param
   version in the xview/fold/L-A key, static opacity prune off, coarser-floor retry replaced by
   cap growth, and a forward flag that writes the fp32 RGB image (T1 finding 1).
3. **T2** fused segment reduce + projection/SH0 backward + Adam (sparse, dense flag), whose
   epilogue writes pfwc's input buffers. Can start now against T1's slot layout (T1 doc §5) with
   synthetic slots from the oracle.
4. **T7** loss on device: L1 and dL/dimage first (pixel-local, trivial), then fused D-SSIM with the
   11-tap separable Gaussian as a banded matmul on the FPU.
5. **M1** driver `render/train.py`: uploads the 169 GT images once, loops forward -> loss ->
   backward -> Adam over the NN tour, no host round trip inside a step. Oracle check: one step on
   a 64x64 crop and on the T0 dumps (rel 1e-2 grads, 1e-3 Adam); PSNR to the GT must rise over
   300 steps.

**Training metric (dashboard).** `train step ms/view (p150)`: wall time of steady-state training
steps (forward + loss + backward + optimizer, densify off) over the 169 NN-ordered views at
1237x822, b2b, divided by views processed; 4 rotated rounds, median of round medians, the #656
A/B rules and JIT-hash check. Shown with its N, K and SH degree, next to the render headline and
#683's forward-only `train NN b2b ms/view` column, tagged "quality not checked" until Q1. After
Q1: `train run ms/iter (p150)` = full 30k run time / 30k with the test PSNR/SSIM/LPIPS beside
it and the published rows of §2 labelled published. The render headline stays the dashboard
headline until the training metric exists (charter).

## 8. Ranked tasks

Order: things that unblock a measured trainer first, then by expected ms/view at parity N.
Device tasks run on the p150 box from the p1 task (msg #382) under its lock, never reserve or
release; bh-33 numbers are p100 (provisional). Mac tasks need no device.

| id | task | class | expected effect (E) | device | depends on | parallel with |
|---|---|---|---|---|---|---|
| D1 | **N x resolution scaling sweep** of today's forward on the training views: importance-pruned proxies of bicycle.ply (N = 6.1M, 3M, 1.5M, 1M, 0.5M; pruning by summed alpha*T over the train views, a timing workload only) x resolution (1237x822, 1/2, 1/4), b2b + per-stage split + pair counts | measurement | fixes the pair ratio and fixed-cost floor in §4 that every number in this plan rests on | yes (keeps the box busy now) | p1 p150 box | A1, T2, T7 |
| A1 | **Cross-view coherence on the NN tour** (CPU): per-tile inversion and run counts between consecutive views (sort warm start), union/single in-frustum ratio for K = 2, 4, 8 (pre-cull), on bicycle.ply and one D1 proxy | analysis | decides L1/L2 | no | - | all |
| T3 | training mode for caches + fp32 image output + cap growth instead of the floor ladder | exact | removes 58 ms/step re-upload and three stale-data bugs | yes, short | #671 | T2, T7, D1 |
| T2 | fused segment reduce + projection/SH0 backward + Adam (sparse, dense flag), epilogue writes pfwc inputs | exact (dense) / algorithmic (sparse) | ~1.6 ms at parity N, K=1 (vs seconds on host) | yes | T1 layout (#685); can start on oracle slots now | T3, T7 |
| T7 | loss on device: L1 first, then fused D-SSIM (FPU banded matmul) | exact | 0.2-0.4 | yes | T0 | T2, T3 |
| M1 | trainer driver, first measured `train step ms/view (p150)` at N = 5.56M and on a D1 proxy; dashboard column | milestone | first M number | yes | #685, T2, T3, T7 (L1 ok) | - |
| T4 | SH3 forward + backward (in T2's pass) | exact | +0.2-0.8 per view (K-dependent) | yes | T2 | T6 |
| T8b | on-device densify/prune: the 3DGS default schedule first (exact replica of the reference recipe), then a budget mode (Taming/LiteGS/absgrad criterion), stream compaction via gather_scan_bases | exact (default) / algorithmic (budget) | budget mode: final-N step 27-29 -> 7-8 ms | yes | T2, T3 | T6, L1 |
| Q1 | **quality-parity run**: 30k iterations from SfM, test metrics vs published 3DGS; T0 check every N steps | gate | required before any target claim | yes (long) | M1, T4, T8b, T7 with D-SSIM | - |
| T6 | blend-backward diet: FPU row-reduce of per-pair partials, slot count on paper first | exact | 1-2 ms at 5.56M, ~0.3-0.6 at parity N | yes after paper step | #685 timing | T4, T8b |
| L1 | exact sort warm start (run-aware merge / insertion fix-up), checked key for key every view | exact | up to ~0.25 at parity N, ~0.9 at 5.56M | yes | A1 positive | T6 |
| L2 | batch-union pre-cull over the next K NN views | exact (sparse Adam; dense with step bound) | ~0.15 at parity N, 0.6-0.8 at 5.56M | yes | A1 positive, T2 | T6 |
| T11 | resolution + primitive schedule (DashGaussian) | algorithmic | run average ~4.5 -> ~2.5-3 ms/iter | yes | Q1 baseline, D1 resolution slope | - |
| T5 | K views per step | algorithmic | -1.2 to -1.6 ms/view at K=4 | yes | M1, T4, Q1 | T11 |

Render levers R1-R5 of structural plan 4 (blend body, mat, B2 ellipse, work stealing) stay in
the render queue; each slot saved in the shared alpha recompute is worth it twice in training.

## 9. Not proposed (ledger check)

Cross-view sort reuse as is (#529; msg #384 forbids it), FPU dense-GEMM blend and its
log-domain variant (t111/t511/t148), Morton chunk cull and u64 gather (#470-#493, #631-#644),
per-view pfwc compaction (#596/#611), persistent kernel (#579), precision cuts (#580), metal
trace replay (#526, <= 0.083 ms on render; re-size only if M1 shows many short programs per
step), 2-chip data parallel (T9, parked by msg #380).

## 10. Sources

docs/training-plan/PLAN.md (#672, @08b39226), docs/training-plan/T1-blend-backward.md (#677,
on ttp/t685 @c472d9b3), benchmarks/train_oracle/README.md (#674), docs/training-bench/RESULTS.md
(#667), docs/training-view/ANALYSIS.md (#668), docs/structural-plan-4/PLAN.md §6 (#663),
docs/hw-ceilings.md (smarton/tt-project-opt). Public: graphdeco-inria/gaussian-splatting
full_eval.py; FastGS arXiv 2511.04283 Tab. 17; Faster-GS arXiv 2602.09999; Taming 3DGS arXiv
2406.15643; DashGaussian arXiv 2503.18402; 3DGS-LM arXiv 2409.12892; LiteGS arXiv 2503.01199 and
github.com/MooreThreads/LiteGS; Speedy-Splat arXiv 2412.00578; Mini-Splatting (github.com/fatPeter/mini-splatting);
Grendel arXiv 2406.18533. Glean/Slack not searched again in this task (#672 covered tt-splat).
