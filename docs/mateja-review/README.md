# Task #233: review of Mateja Stojkovic's TT gaussian-splatting work

No device run was needed. All numbers below come from their committed result
files or from our earlier measured docs. Savings for our pipeline are **modeled**.
GPU numbers are **published, not measured** by us.

## 1. Where things are

**Code (found).**
- Repo: `github.com/Kovelja009/gsplat_tt` (Vanja Kovinić's repo; Mateja's work
  landed there through PRs). Our repo has it as the `upstream` remote.
- GitHub now returns 404 for both the API and `git fetch` on that repo, so it is
  private or deleted. We still have a full local mirror of its refs.
- Last commit: `upstream/main` = `37750fefbfdf79fd67bf6832d9fe834118a725dd`,
  2026-06-17, "Merge pull request #9 from Kovelja009/mstojkovic/add_cuda_bench"
  (authored by GitHub user MatejaStojkovic, "Cloverleaf"; work commits are by
  `mstojkovicTT`).
- Other mirrored branches: `mstojkovic/add_cuda_bench` (b51d3dc),
  `intra-tile-parallelism`, `sched-comparison-experiment`, `ttnn-op-backend`,
  `tt-ipc-perf`, `benchmark-harness`, `fix-render-flickering`, `results`,
  `cuda_kernel` (76ea41a).
- Key paths on `upstream/main`:
  - `backends/tt/backend.py`, `backends/tt/lpt.py`, `backends/tt/segments.py`
  - `backends/tt/tt-metal/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/`
    `{alpha_blend, alpha_blend_partial, alpha_blend_combine}/` (ttnn ops)
  - `backends/cuda/kernels/alpha_blend.cu` (their own CUDA blend kernel)
  - `benchmark/results/{tt/sched_segmented,tt/sched_lpt,tt/sched_round_robin,cuda,cpu}/*.csv`
  - `docs/plan_progress.md` (section "Intra-tile parallelism 2026-06-08")

**Deck F0C1PMDG12M (blocked).**
- It is attached in the Slack DM D0C1CV1AJJV (message p1789378094243739,
  2026-09-14), owner Mateja Stojkovic.
- Glean indexes the message, but the file itself returns `NOT_FOUND` (also in raw
  mode). The Slack file URL returns `FILE_EXTENSION_UNSUPPORTED`. A Glean slides
  search finds no copy.
- To read it, someone has to download it from Slack by hand, or ask Mateja to share
  it as a Google Slides/Drive file.
- Risk: the deck is three months newer than the last commit we have. It may show
  work after 2026-06-17 that this review cannot see.

## 2. Their pipeline

- Host (Python/torch, CPU): load `.ply`, project, tile assignment, integer-key
  `torch.sort`, depth segmentation, LPT schedule.
- Device (Blackhole, 130 cores): one in-process ttnn blend op. A heavy tile is cut
  into K contiguous depth segments. `alpha_blend_partial` blends each segment from
  T=1 and writes (R,G,B,T). `alpha_blend_combine` merges them with the "over"
  rule: C = C0 + T0·C1 + T0·T1·C2 + ...
- The compute kernel is the original naive tile-op chain: per Gaussian it reads 9
  fp32 scalars, runs sub/mul/add tile ops, `exp_tile<true>` (approximate), clamps
  alpha at 0.99. Blend state (R/G/B/T/sat_mask) goes through circular buffers on
  every step. sat_mask is refreshed every 16 Gaussians.
- Culling: opacity below 1/255 and a bounding-radius cap.

## 3. Their numbers (from `benchmark/results/*/*.csv`, published by them)

| scene | backend | res | total ms | FPS | device blend ms | host project / tile / sort ms |
|---|---|---:|---:|---:|---:|---|
| train (741,883 G) | TT segmented | 960 | 257.0 | 3.89 | 55.3 | 51.0 / 20.8 / 21.9 |
| train | TT segmented | 640 | 215.6 | — | 45.7 | — |
| train | TT segmented | 256 | 177.2 | — | 40.7 | — |
| train | CUDA, GTX 4060 | 960 | 163.3 | 6.12 | 9.68 | project 73.7 (CPU) |
| train | CUDA | 640 | 140.7 | — | — | — |
| luigi (14.5K G) | TT segmented | 960 | 39.2 | — | — | — |
| luigi | CUDA | 960 | 13.8 | — | — | — |

- Scene: Mip-NeRF/T&T `train` (742K Gaussians) and `luigi`. No bicycle.
- Resolutions: 256, 640 and 960 square.
- "1.6x slower than a 4060" is the end-to-end ratio for train: 257/163 = 1.57x at
  960 and 1.53x at 640.
- Their GPU side is **their own naive CUDA blend kernel** with the same CPU host
  stages, not gsplat or INRIA CUDA. Most of both totals is host Python.
- Blend alone: their TT kernel is ~5.7x slower than their own CUDA kernel
  (55.3 vs 9.7 ms).
- Their README reports that the depth split cut train device time from 345 to
  41 ms at 256 (8.5x) and from 130 to 46 ms at 640. LPT gave ~21x over one core.

## 4. Ours, for scale

- Bicycle (~6M Gaussians), 1024x1024, all five programs on device, resident
  buffers: **12.665 ms/view (79.0 FPS)**, md5 46a725ab, yyzo-bh-07 p100a (t229).
- Traced busy ms/view (t230): pfwc 2.33, K2 0.98, sort_ol 1.44, mat+blend 7.56
  (blend SFPU makespan 4.42).
- Our GPU anchor (`docs/OPTIMIZATION-PLAN.md`): INRIA 3DGS on RTX A6000, bicycle,
  93 FPS at 1080p (published, not measured), ≈5.44 ms scaled to 1024².
- Their pipeline renders a scene ~8x smaller, at a slightly lower resolution, ~20x
  slower end to end. Their blend kernel alone takes ~4x our whole frame.

## 5. Technique by technique

| their technique | ours | modeled saving on bicycle (ms/view) | clears 0.3 gate? | build cost |
|---|---|---:|---|---|
| Depth-segment split + over-combine for heavy tiles | not built | ≤0.10 (see below) | no | high: new partial/combine path, mat + writer changes, md5 changes |
| LPT tile schedule | superseded: descending deal + late dynamic claim (t188) | 0 | — | — |
| Approximate exp | have (SFPU exp in fused blend) | 0 | — | — |
| Opacity < 1/255 cull | have (plus per-microblock contrib cull in mat) | 0 | — | — |
| Bounding-radius cap | have (opt/032 per-tile cap) | 0 | — | — |
| sat_mask refresh every 16 G | superseded: block early termination + T readback every 512 records | 0 | — | — |
| Blend state in CBs every step | superseded: state stays in SFPU registers/Dst, replayed bodies | 0 | — | — |
| Resident px/py grids, 4 KB DRAM pages | have (resident chain, no per-view D2H) | 0 | — | — |
| Host torch.sort, host project/tile | superseded: device project, tile assign, sort (pfwc/K2/sort_ol) | 0 | — | — |
| CUDA benchmark harness | n/a: we use published GPU numbers per charter | — | — | — |

**Why the depth split does not pay for us.**
- It fixes load imbalance when a few tiles are much heavier than the rest. That
  was huge for them (one heavy tile on a 130-core static schedule).
- Our remaining blend tail after t188 (late claim, descending order) is a core-end
  max−mean of **0.106 ms mean, 0.249 max** per view. That is the most a split
  could remove.
- t183 already modeled the closest version for us (splitting the top 32 tiles
  into 32x16 halves): under late claim it saved nothing beyond late claim alone
  (−0.453 vs −0.465 ms), and lost ground with any per-half overhead.
- A depth split adds more cost: each extra segment must start at T=1, so it loses
  early termination for the records a single pass would skip; the combine pass
  adds reads and writes; and rounding changes, so md5 46a725ab breaks.
- Net modeled saving ≤0.10 ms/view, probably about zero. Below the gate.

## 6. Ranked follow-ups that clear the 0.3 ms/view gate

**None.** Every technique in their code is one we already have, have replaced
with something faster, or (depth split) is modeled below the gate. Our open
blend lever is still t230's S2a/S2 TRISC0 decode-ahead staging (modeled
+0.66 to +1.20), which is unrelated to this work.

Non-gated items:
1. Read the deck once it is available as a Drive link, and check it for work after
   2026-06-17. Re-open this review only if it shows a new device technique.
2. Qualify the "1.6x off a 4060" anchor in `docs/OPTIMIZATION-PLAN.md`: it compares
   their naive CUDA blend with CPU host stages, end to end, on `train`. It says
   nothing about gsplat/INRIA CUDA speed. The A6000 INRIA bicycle number stays our
   reference.

## Reproduce

```
git fetch upstream            # now 404; the local mirror refs are enough
git show upstream/main:benchmark/results/tt/sched_segmented/tt.csv
git show upstream/main:benchmark/results/cuda/cuda.csv
git show upstream/main:backends/tt/segments.py
git show upstream/main:docs/plan_progress.md | sed -n '/Intra-tile parallelism/,$p'
```
