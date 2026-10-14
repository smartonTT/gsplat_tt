# On-device Gaussian-splatting training: ranked plan (task #672, user msg #353 / #357)

2026-10-13 PDT. A plan only: no device runs, no reservation, no p100 lock. To be sent to the user
before any device time goes to the training path (msg #357).

Labels: **M** = measured on our hardware (cited), **E** = estimate from measured numbers (basis
shown), **P** = published GPU number (not measured by us), **P?** = seen in a search snippet or a
secondary table, not checked against the full paper.

Inputs: docs/training-view/ANALYSIS.md (#668, origin/ttp/t668 @91bd2f55; this branch is cut from
it), docs/structural-plan-4/PLAN.md (#663 @149af7ac, incl. its §6 exclusions),
docs/training-bench/RESULTS.md (#667 @76c6c7d7: blocked, no bicycle dataset poses or images
reachable), docs/hw-ceilings.md, render/host/pfwc_device.cpp:769-805 (upload cache), memory
(shelved/landed levers, #656 A/B rules).

## 0. Starting point

| item | value | label |
|---|---|---|
| Forward, best-iter-229, p100 bh-33, 110 cores, 1024x1024, 5.56M Gaussians, b2b | 8.063 ms/view | M |
| Forward split: pfwc 1.38, sort 1.33, mat 1.33, blend 3.79, gaps 0.2 | ms/view | E (plan 4 §1) |
| Backward, optimizer, loss on device | none exist | - |
| CPU reference (gsplat/rasterization.py) | numpy, SH0, no autograd | M (code) |
| H2D scene upload, frame 0: pack 49.8 + upload 7.8 ms (means + cov3d SoA, ~200 MB) | 58 ms | M (ANALYSIS §1) |
| 2 chips AFR: bench b2b 4.226 vs 8.395 ms/view (1.99x, #582); viewer page FPS 113.3 vs 71.2 (1.59x, #615, host-bound) | | M |
| Dataset poses and GT images for bicycle | not reachable (#667) | M |

GPU training references (bicycle unless noted, 30k iterations):

| source | total | per iteration | label |
|---|---|---|---|
| Vanilla 3DGS on RTX 4090, bicycle (FastGS arXiv 2511.04283 per-scene table) | 27.97 min, 4.71M Gaussians at the end | ~56 ms (E from P?) | P? |
| Taming-3DGS budget mode on RTX 4090, bicycle (same table) | 5.65 min, 0.81M Gaussians | ~11 ms (E from P?) | P? |
| Faster-GS on RTX 4090, Mip-NeRF360 average (arXiv 2602.09999 Tab. 4) | 4m10s | ~8.3 ms (E from P) | P |
| Faster-GS bicycle render, same model | 1.83 ms/view | | P |

Caveat: GPU per-iteration times average over a run in which N grows from ~50-100k SfM points to
the final count (densification stops at 15k). Our estimates below are at the **final** N = 5.56M,
i.e. the slowest step of a run. A fair comparison is either the full-run time (needs T10) or the
GPU step at final N (not published).

## 1. Where a training step's time would go on TT (E, final N = 5.56M, one view)

Per-view costs; "/K" marks costs paid once per optimizer step and shared by K views.

| # | stage | ms per view, K=1 | K=4 | basis (E unless marked) |
|---|---|---:|---:|---|
| a | forward SH0 as today | 8.06 | 8.06 | M |
| b | SH3 colour read in forward | +0.78 | +0.20 | ANALYSIS §3: 1.7M visible x 192 B / 420 GB/s |
| c | static opacity prune off (6.13M instead of 5.56M) | +0.14 | +0.14 | pfwc 1.38 x 572k/5.56M; pairs unchanged (pruned ones fall under the 1/255 floor) |
| d | loss L1 + D-SSIM, forward and backward, on device | 0.2-0.6 | 0.2-0.6 | ~1 Mpx x 3 ch x ~20 maps x 4 B = 240 MB if unfused (0.6); fused per tile with an 11-px halo ~0.2; the 11-tap separable filter is a banded matmul (FPU) |
| e | blend backward (recompute alpha front to back, per-pair grads) | 8.7-10.2 | same | forward body 54 slots -> ~100-120 slots per microblock (recompute ~20, reciprocal of (1-a) ~8, gradient math ~25, 9 per-lane accumulators spilled past 8 LREGs ~18) plus a 32-lane reduce of 9 values per pair (~90 slots / 3.7 live microblocks); 0.041 ms per slot (SWAP_FILL, M) -> 2.3-2.7x of 3.79 |
| f | sort, mat in backward | 0 | 0 | reuse the forward's per-tile sorted lists and live masks (21 MB pairs + ~10 MB masks per view stay in DRAM) |
| g | per-pair grad scatter to Gaussian-major slots (9 x fp32 = 36 B per pair) | 0-0.7 | same | 2.614M writes x 78 cycles (hw-ceilings p4, M) / 220 movers / 1.35 GHz = 0.68; the movers are partly idle during blend (t630: reader 4.47, writer ~2.1 ms idle, M), so mostly hidden |
| h | segmented reduce of pair grads (contiguous, deterministic) | 0.22 | 0.22 | 94 MB read / 420 GB/s, fused into (i) |
| i | projection + SH backward and sparse Adam, fused, fp32 params, bf16 moments | 3.8 | ~1.7 | 1.7M visible x 59 params x 16 B (param r/w 8 + moments r/w 8) = 1.6 GB / 420 GB/s; K=4: union of visible ~3M (guess, needs T10) / 4 |
| j | Adam epilogue writes pfwc's inputs (means SoA, cov3d, opacity, SH) | 0.16 | 0.04 | 1.7M x 40 B / 420 GB/s; replaces the 58 ms host re-pack |
| k | step boundary: next step's pfwc cannot overlap the previous blend | 1.4 | 0.35 | one exposed pfwc per step (b2b overlap lost once) |
| l | densify/prune every 100 steps (to iter 15k), on device | 0.12 | 0.03 | ~12 ms per densify (2.5 GB of params + moments read and written, incl. compaction, at 420 GB/s) / 100 steps |
| | **total per view** | **~23.6-26.2** | **~19.6-22.0** | E (sum of rows) |
| | dense host-side alternative for (e)-(j) | seconds | | numpy backward + host Adam over 328M params + 58 ms re-upload |

Shares at K=1 (mid estimate 24.9 ms): blend forward+backward ~13.2 ms (53%), forward incl. SH3 ~9.0 ms (36%), param/Adam pass ~4.2 ms (17%). Forward is ~36% of the step here
(not 18%) because our forward is slow relative to our estimated backward; GPU data put the
forward at ~18% (Laplacian Hierarchies, P?). If the backward comes in at the GPU ratio (2-3x
forward = 16-24 ms), the step is ~28-36 ms (E, upper band).

On these estimates one p100 at final N (~20-26 ms/view, K=4 to K=1) would be ~2x faster per iteration than
vanilla 3DGS on a 4090 averaged over its run (~56 ms, P?) and ~2.5-3x slower than Faster-GS
(~8.3 ms average, P). 2 chips (T9) would be ~11-12 ms/view at K=4-8 (E).

Render levers R1 (blend body), R3 (mat), R5 (stealing) of plan 4 cut (a). R1's slot work carries
into (e): the backward recomputes the same alpha, so each slot saved in the shared part is worth
~0.04 ms twice.

## 2. The three caches that break once parameters train (names as in ANALYSIS §1)

| cache | breaks how | fix in training | per-step cost |
|---|---|---|---|
| **Scene upload cache** keyed by host pointer + N (pfwc_device.cpp:769-805, project_device.cpp:358-376, gather_visible_device.cpp:488-492) | in-place host updates keep the pointer: device keeps old means/cov/colour/opacity | #671 version-keys it (correctness for any host-side update). For on-device training the device buffers become the master copy: Adam (T2) writes pfwc's input layout directly, the host never uploads | 0 on device (j: 0.16 ms); 58 ms per step if updates stay on the host (M parts) |
| **xview / B4a fold / L-A prefetch hint** keyed by pose + scene pointers (xview.h, render.cpp:214-225, 410-450) | an update between steps hits with stale pfwc/K2 outputs | add the param version to the key; within one step's K views overlap stays valid; across a step the hint is dropped | ~1.4 ms per step (k), /K per view |
| **Static opacity prune at load** (render/opacity_prune.py, 572k of 6.13M dropped, order remapped) | opacities train, so the prune set and the remap go stale | off in training; 3DGS's own prune (opacity < 0.005 at each densify step) does it on device with the compaction in T8 | +0.14 ms per view (c) |

Not a cache but also wrong for training: the **overflow retry with a coarser contrib floor**
(render.cpp:493-517) changes the image and biases gradients. Training grows the caps instead
(one 9-12 ms regrow, M #85). Folded into T3.

## 3. Design notes per area

**Backward (T1, T2).** No float atomics over the NoC (ANALYSIS §4). Deterministic path:
1. The forward keeps per view: per-tile sorted pair lists and live microblock masks (already
   built), and per pixel the final T and the final colour (8 MB). No last-contributor index is
   needed when the backward runs front to back.
2. Blend backward per tile on the core that owns it in the forward (same claim order, LPT),
   front to back. Per pixel it carries T and the colour accumulated so far; the colour behind
   Gaussian i is (C_final - C_<=i) / T_{i+1}, so dL/dalpha_i needs one reciprocal of (1 - a_i)
   per live pixel (SFPU). Per pair it sums 9 gradient terms (dmean2D 2, dconic 3, dopacity 1,
   dcolour 3) over its live microblocks.
3. FPU option for the 32-lane reduce: write each pair's 9 per-lane partials as rows of a Dst
   tile and reduce rows on the FPU (matmul with a ones column, or reduce_tile), instead of SFPU
   shuffles. Saves up to ~90 SFPU slots per pair (~1-1.5 ms of (e), E). T6 sizes it.
4. Each pair writes its 36 B gradient to slot P[g] + j in Gaussian-major order. P[g] is the
   exclusive prefix sum of each visible Gaussian's tile count, which pfwc/gather_scan_bases
   already have; j is the pair's index within that Gaussian's tile rectangle. No two pairs share
   a slot, so the scatter needs no atomics and its sum order is fixed.
5. One pass over visible Gaussians reads its contiguous pair-grad segment, sums it, runs the
   projection backward (cov2d, EWA Jacobian, quaternion and scale, means) and the SH backward,
   and applies Adam in the same pass (T2). Gradients for params never hit DRAM at K=1.

**Optimizer (T2).** Sparse Adam over visible Gaussians only (Taming 3DGS, P) fused with the
backward (Faster-GS Tab. 3: fusing the optimizer into the backward 5m31s -> 5m04s, P). fp32
params, bf16 moments: 16 B per param instead of the dense 28 B x 328M (= 22 ms, ANALYSIS).
Sparse Adam skips moment decay for invisible Gaussians: a known small change to Adam's
behaviour (Taming's choice, P); T2 checks PSNR against dense Adam on the CPU oracle. Host Adam
is not viable: D2H of grads + host update + 58 ms re-upload per step.

**SH3 (T4).** Forward: per visible Gaussian, read 48 coefficients and evaluate colour for the
view direction (~0.025 ms math, E); with K views per step, read once per K (+0.78 / 0.39 / 0.20
/ 0.10 ms at K = 1/2/4/8, E). Backward: dL/dSH_k = Y_k(dir) x dL/dcolour, inside T2's per-Gaussian
pass; the coefficient read is shared with the forward's if the K forwards and the backward pass
are organised per Gaussian chunk, otherwise it is read twice. bf16 coefficients halve the read
(PSNR check needed). bicycle.ply already has f_rest_0..44 (ANALYSIS §3).

**Batching and multi-chip (T5, T9).** K views per optimizer step. Per view it amortises SH3 reads,
the param/Adam pass, the step boundary and the densify pass: ~4 ms per view saved at K=4 (§1
table rows b, i, j, k, l, E). Changes the optimisation: gradients summed over K views, learning rates scaled
(Grendel, arXiv 2406.18533, trains with batches up to 32 using sqrt(batch) lr scaling, P?). Quality
must be checked against K=1 at equal views seen. 2 chips: each chip renders K/2 views; gradients
for the visible union are exchanged before Adam. With each chip owning half the Gaussians
(Grendel-style), each sends ~100 MB bf16 and receives ~100 MB per step (E from ANALYSIS's 200 MB).
Over host PCIe at ~26 GB/s (E from the M frame-0 upload: ~200 MB in 7.8 ms) that is ~8 ms per
step if not overlapped, 1 ms/view at K=8; over the card's QSFP links it would be less, if the two
boards of the 2-board host are cabled (unknown). Expected 2-chip speedup ~1.6x at K=2, ~1.85x
at K=8 (E, vs 1.99x for rendering alone, M).

## 4. Ranked tasks

Ranked by effect on the training step per view; the first three are the minimum for any
measured on-device step. Each device task: `ttp lock p100 -- ...` on the existing measurement
reservation (bh-33), never reserve/release; JIT-hash check (`python3 opt/ab_jit_check.py`) on both
arms of any A/B; a b2b delta under ~0.1 ms/view from 4 rounds is noise without a tighter same-day
A/A; forward-path changes keep the render gates (md5 077caaba, hero on device, PSNR >= 42.4 dB vs
benchmarks/reference_v2/hero.png, visual tile-artifact check).

| rank | task | effect on step time per view | first measurable milestone | device | device-hours | depends on |
|---|---|---|---|---|---|---|
| T0 | **Gradient oracle on the Mac**: torch-autograd port of gsplat/rasterization.py (SH0 and SH3, L1+D-SSIM), finite-difference checked; a self-supervised target (render of the unperturbed scene vs a perturbed scene) so no dataset is needed | enabler (0 ms) | per-param grads of a 64x64, 50-Gaussian scene match finite differences to rel 1e-3 | no | 0 | - |
| T1 | **Blend backward on device** (§3 steps 1-4): per-pair grads, Gaussian-major scatter, fixed sum order | adds 8.7-10.2 + 0-0.7 (E); replaces a host backward of seconds | 64x64 synthetic tile: per-pair grads vs T0 rel err <= 1e-2 (bf16 inputs); then bicycle hero view backward timed alone (ms/view, traced split) | yes | 8-12 | T0 |
| T2 | **Fused projection/SH0 backward + sparse Adam on device** (§3 step 5), Adam epilogue writes pfwc's input buffers | 3.8 + 0.22 + 0.16 at K=1 (E); replaces host Adam (dense 22 ms on device, seconds on host) | one step on a 10k-Gaussian scene: updated params match T0's Adam step rel 1e-3; then 5.56M step time | yes | 6-8 | T0, T1 for the input layout; can start in parallel on synthetic grads |
| T3 | **Training mode for the caches** (§2): device buffers as master (builds on #671's version key), param version in the xview/fold/L-A key, opacity prune off, overflow retry replaced by cap growth | +0.14 + 1.4/K (E); removes 58 ms per step (M parts) and three stale-data bugs | unit check: after a param write, the next view renders the new params (md5 of a moved Gaussian), with xview on; b2b at K=1 with the hint dropped per step | yes (short) | 2 | #671 |
| M1 | **Minimal measured training step** (milestone, not a lever): K=1, SH0, final-size bicycle.ply, self-supervised target (T0), forward + loss + T1 + T2 + T3, 30 views | the first M number for step time per view | grads of one step match T0 on a 1/16-resolution crop (rel 1e-2); then b2b ms per training step per view over 30 steps, traced split per stage; PSNR to the target rises over 300 steps | yes | 3 | T0-T3, loss (T7) or L1-only first |
| T4 | **SH3 forward (K-batched read) + SH3 backward** | +0.78 at K=1, +0.20 at K=4, +0.10 at K=8 (E) | forward SH3 hero vs T0 CPU SH3 render >= 42.4 dB; fwd b2b delta at K=1 and K=4 | yes | 4-6 | T0; backward part in T2's pass |
| T5 | **K views per optimizer step** (shared param/Adam pass, per-step boundary paid once) | saves ~4 ms/view at K=4 vs K=1 (E: rows b, i, j, k, l) | step time per view at K=1/2/4/8 (M); PSNR after equal views seen vs K=1 on the CPU oracle or the device | yes | 4 | M1, T4 |
| T6 | **Blend-backward diet**: FPU row-reduce of per-pair partials; slot count of the backward body on paper first (as plan 4 R1) | 1-2 of the 8.7-10.2 (E) | slot count on paper (no device); then A/B b2b per training step | step 0 no, then yes | 4 | T1 |
| T7 | **Loss L1 + D-SSIM on device**, fused per tile, separable 11-tap filter as banded matmul on the FPU | 0.2-0.6 (E); host SSIM at 1 Mpx would be tens of ms | loss and dL/dimage vs T0 rel 1e-3; time per view | yes | 2 | T0 |
| T8 | **Densify / prune on device** (grad-norm accumulate in T2, clone/split/prune by stream compaction reusing gather_scan_bases + gather_visible_scatter) | ~0.12/K amortised (E); required for training from SfM points, not for M1 | one densify on bicycle: N and params match T0's densify rule; time per densify | yes | 4-6 | T2, T3 |
| T9 | **2-chip data-parallel step (proposal only)**: K/2 views per chip, Gaussian-owner split, ~100 MB bf16 each way per step | ~1.6x at K=2, ~1.85x at K=8 vs one chip (E) | exchange time per step host-staged vs QSFP (if cabled) on a 2-board p150 | yes, **only the 2-chip measurement task may reserve 'p150-2board'** | 3 | T5 |
| T10 | **Training-view data**: bicycle poses + images_4 (public Mip-NeRF360 dataset; #667 tooling ready) for real-loss runs, the K-union size in row i, and a full-run time to compare with published GPU run times | enabler | t667 bench at 1237x822 in NN-tour and shuffled order | yes | 1-2 | user: fetch the public dataset |

Not proposed (ledger check, plan 4 §6 and memory): cross-view sort reuse (#529, shelved; in
training the views also change with every step), FPU dense-GEMM blend and its log-domain variant
(t111/t511/t148: exp stays on SFPU; the backward reuses the SFPU body instead), persistent kernel
(#579), precision cuts below 42.4 dB (#580). The matrix-native order-independent blend of
tt-splat (github.com/kinginu/tt-splat, community, trains device-resident on a p150a) is a
different model (no sorted alpha compositing, -3 dB on lego, its README) and cannot train our
3DGS-format scenes; noted, not proposed.

## 5. Risks

- Blend backward LREG pressure: 9 accumulators plus per-pixel state exceed 8 LREGs; spills to
  Dst cost slots. Sized in (e); T6 measures.
- Reciprocal of (1 - a) near a = 0.99 (clamped as in the forward) loses precision in bf16
  paths: keep fp32 in Dst (`fp32_dest_acc_en`).
- Memory: params + bf16 moments + per-view pair grads (94 MB) x K + forward state x K, ~5 GB at
  K=8 and 6M Gaussians (E): fits in 28-32 GB, but L1 CB budgets in blend are already full; the
  backward is a separate program with its own CBs.
- Sparse Adam and K-batching change training dynamics: judged by PSNR against the oracle, not
  only speed.
- Every estimate above is E; M1 is the first number to trust. Re-rank after M1.

## 6. Upstream knowledge used

ANALYSIS.md (#668), structural plan 4 (#663), hw-ceilings (p150/p100 NoC costs), Faster-GS
(arXiv 2602.09999), Taming 3DGS (arXiv 2406.15643), FastGS per-scene table (arXiv 2511.04283,
P?), Grendel (arXiv 2406.18533, P?), tt-splat (github.com/kinginu/tt-splat, community TT trainer:
"params + optimizer state resident on-card ... captured as a single traced graph"; dense binning
cost ~200 ms per iteration there, which our K2 tile lists avoid).
