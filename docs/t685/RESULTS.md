# T1 device blend backward: results (#693 v1 2026-10-13, #704 v2 2026-10-14)

Kernel: render/train/kernels/{bwd_reader,blend_bwd_compute,bwd_writer}.cpp, host
render/train/blend_bwd_bench.cpp, driver docs/t685/drive.sh (one `ttp lock p100`: preflight,
opt/sync_remote.sh, build, run, pull, check). Device: bh-33, **p100 (provisional)**; the timing
of record is p150 and still to come. Raw files: docs/t685/results/.

v1 differs from DESIGN.md: one pair per pack (the 9 per-lane term sums of the ACC tile are
packed as one fp32 tile per pair) and the host does the 32-lane reduce and the finish
(A Su + B Sv, ...). Dst holds T, BH, g (3) and the x/y ramps as fp32; records stay fp32.

## Oracle check (`blend_backward_dev check`, bar max rel err <= 1e-2, floor 1e-4 of scale)

| dump | pairs | rel err | worst term | pass |
|---|---|---|---|---|
| t0_sh0_64x64_n50 (first run, forward's `_sfpu_exp_21f_bf16_`) | 74 | 1.25e-1 | dmy 1.8e-1 | **fail** |
| t0_sh0_64x64_n50 (fp32-accurate exp) | 74 | 8.8e-5 | dC 8.8e-5 | pass |
| t0_sh3_64x64_n50 | 74 | 6.8e-6 | dop 6.8e-6 | pass |
| synth s1 256x256, N=3000 (docs/t685/synth.py) | 7,975 | 3.5e-4 | dop 5.7e-3 | pass |
| synth s2 512x512, N=12000 | 35,052 | 9.4e-4 | dmy 4.8e-3 | pass |

Finding: the forward's exp (`_sfpu_exp_21f_bf16_`, a degree-2 2^f polynomial, about 1.7e-3
relative error) is not accurate enough for gradients. The errors were uniform at about 5e-4 of
each term's scale, so slots that are small after cancellation failed (only 1-3 slots per term).
The kernel now uses the fp32-accurate exp (Juffa's algorithm, the same as tt-metal's
`_sfpu_exp_fp32_accurate_`), inlined in the kernel. The numpy model with a model of the old exp (`blend_backward_dev.exp21f`)
reproduces the failed device run term for term (rel err 0.1250 vs 0.1253 on device; dmy 0.185
both), so the cause is confirmed; tests/unit/test_blend_backward_dev.py keeps it. Consequence for training: the forward's
training mode must use the same exp, or its image and T disagree with the backward's
recomputed alpha.

The synthetic dumps (s1, s2) are timing workloads, checked against the numpy reference on the
same inputs (the image is an fp32 forward with the device rules). The slightly larger errors
there (up to 5.7e-3 on single small slots) come from fp32 summation order over many more
pairs per pixel, still under the bar.

## Timing, p100 (provisional), bh-33

Host wall time of EnqueueMeshWorkload + Finish, median of 5, after a JIT warm-up run.
Includes dispatch; excludes the host lane reduce (listed separately).

| dump | pairs | live vectors | cores | max pairs/core | device ms | host reduce ms |
|---|---|---|---|---|---|---|
| sh0 | 74 | 721 | 4 | 21 | 0.069 | 0.05 |
| s1 | 7,975 | 76,701 | 64 | 224 | 0.499 | 4.70 |
| s2 | 35,052 | 326,498 | 110 | 454 | 1.367 | 17.8 |

Slope s1 to s2: about 3.8 us per pair on the busiest core (about 10 live vectors per pair).
v1 packs and writes a 4 KB tile per pair and leaves the lane reduce to the host, so its cost
per pair and its output size (4 KB per pair, 143 MB for s2) do not carry to the hero view.
No hero-view number yet: the hero needs v2 (below) and a hero dump.

## v2 (#704, 2026-10-14): lane reduce and finish on device

Changes against v1 (commit 60591b2b):
- Compute: after each pair, a per-lane finish (linear, so it can run before the reduce:
  dmx = A Su + B Sv and dmy = B Su + C Sv per lane, dA = -Su²/2, and so on). Then a 32-lane sum
  per group of 4 terms: SFPTRANSP, an add tree, and a row rotate-add by 1, 2 and 4
  (SFPSHFT2 SUBVEC_SHFLROR1). The result goes into column p % 8 of ACC vectors
  24 + 3 (p % 16 / 8) + {0, 1, 2}.
- Pack: one ACC pack per 16 pairs instead of one per pair, using the non-zeroing release.
- Writer: gathers 9 floats per pair and writes one 64 B page to slot rec[13] (= P[g] + j).
- Host: no host reduce. Readback is n_slots x 64 B (2.2 MB for s2) instead of 4 KB per pair
  (143 MB for s2).
- numpy model of the pack: `blend_backward_dev.pack_v2`, tested against the v1 host finish
  (`test_pack_v2_matches_host_finish`).

Oracle on device (bh-33, p100), bar 1e-2. Raw files: results/check-*-v2.json.

| dump | pairs | v2 rel err | worst term | v1 rel err |
|---|---|---|---|---|
| t0_sh0_64x64_n50 | 74 | 1.4e-4 | dC 1.4e-4 | 8.8e-5 |
| t0_sh3_64x64_n50 | 74 | 1.0e-5 | dop 1.0e-5 | 6.8e-6 |
| s1 256x256, N=3000, seed 1 | 7,975 | 3.7e-4 | dop 5.8e-3 | 3.5e-4 |
| s2 512x512, N=12000, seed 2 | 35,052 | 9.0e-4 | dmy 3.9e-3 | 9.4e-4 |

All four pass. The small change against v1 comes from a different fp32 summation order (a tree
on device instead of a sequential host sum).

Timing, **p100 (provisional)**, bh-33. The method is the same as v1: median of 5 host
EnqueueMeshWorkload + Finish after warm-up. Raw files: results/timing-*-v2.json.

| dump | pairs | max pairs/core | v1 device ms | v1 host reduce ms | **v2 device ms** | v2 host reduce |
|---|---|---|---|---|---|---|
| sh0 | 74 | 21 | 0.069 | 0.05 | 0.086 | none |
| sh3 | 74 | 21 | | | 0.073 | none |
| s1 | 7,975 | 224 | 0.499 | 4.70 | **0.463** | none |
| s2 | 35,052 | 454 | 1.367 | 17.8 | **0.819** | none |

- s2: device time 1.67x faster (1.367 to 0.819 ms). End to end it is 23x faster
  (19.2 to 0.82 ms), since the 17.8 ms host reduce and the 143 MB readback are gone.
- Slope from s1 to s2 on the busiest core: 1.55 us per pair, against 3.8 us in v1.
  That is about 2,100 cycles at 1.35 GHz for about 10 live vectors plus the finish.
- The two tiny dumps are dispatch-bound (4 cores).
- Per-stage split: not measured. The kernel has no in-kernel stage counters yet. Only device
  versus host is split here, and v2 has no host stage left except readback.

Hero view: not timed in this task (no hero dump yet, see the follow-ups). Rough scaling, not a
measurement: at 1.55 us per pair on 110 p100 cores, 1M pairs cost about 14 ms. The hero's pair
count from the forward has to be measured first. If it is several million, T1 at this
per-pair cost is far above the plan's 8.7-10.2 ms. The cost per pair then has to fall by
several times (see the T1 v3 levers below).

## Hero (#707, 2026-10-14): bicycle _DSC8679, 1237x822, p150 (bh-49 board 0, p150a)

Dump: `docs/t685/hero.py` (scenes/bicycle.ply, camera _DSC8679 from benchmarks/cameras_test.json,
fp32 forward with the device rules, L1 dL/dimage against images_4/_DSC8679.JPG); metadata in
`hero_meta.json`. **4,478,225 pairs**, 1,014 tiles, busiest tile 25,069 pairs, 1.81M visible
Gaussians. Commit 65326d1e, v2 kernels with `BWD_STATS=1`.

**Oracle: FAIL on the hero** (`results/check-hero-v2-p150-FAIL.json`): max rel err 49.4 (bar
1e-2); dr/dg/db about 1.0, geometry terms 35-113. On the same p150 build sh0 (1.4e-4) and sh3
(1.0e-5) pass (`results/check-sh{0,3}-v2-p150.json`). So the v2 kernel, or the vectorised prep,
breaks on something only the hero has (long tiles, many pairs per core); the timing below is
the cost of the work but not yet of a correct result. Breakdown by tile pair count and pair rank:
`hero_err.py` -> `results/err-hero-v2-p150.json`.

Timing (`results/timing-hero-v2-p150.json`, `runs-hero-v2-p150.json`): 6 runs, first 109.0 ms,
warm runs 86.33 / 86.67 / 86.32 / 86.32 / 86.32 ms, **median of 5 warm 86.32 ms**. Host reduce 0.

Per-stage split (MATH wall clock at 1.35 GHz, `results/stats-hero-v2-p150.json`, 110 cores):

| stage | busiest core ms | mean core ms |
|---|---|---|
| total | 86.28 | 51.31 |
| tile setup | 0.04 | 0.02 |
| record wait | 0.67 | 0.63 |
| pair setup | 4.35 | 4.23 |
| bodies | 77.65 | 42.98 |
| finish | 3.50 | 3.38 |
| pack sync | 0.08 | 0.07 |

- The bodies are 90% of the busiest core; record wait and pack sync are under 1%, so the
  reader/writer and DRAM are not the limit.
- Busiest core (107): 28 tiles, 41,398 pairs, 492,267 live microblock bodies: 2.08 us per pair
  total, 1.88 us per pair in the bodies, 213 cycles per body. (The core with the most records,
  52,187, is not the slowest: bodies per pair vary.)
- Load balance: busiest 86.3 ms vs mean 51.3 ms (1.68x). Balancing by body count instead of
  pair count alone has an ideal bound of about 51 ms (the mean; not measured).
- Against v1/v2 on the small dumps (1.55 us/pair on the p100), the hero per-pair cost is
  similar, so the time scales with pairs as sized.
- 86 ms per view is 40-80x off the 1-2 ms/view training target: v3 needs both fewer bodies per
  pair (skip dead microblocks) and much cheaper bodies, plus the balance fix.

Screenshot (`opt/metal-screenshots/t707-hero/`, same lock, commit 65326d1e, defaults): device
hero.png PSNR 42.85 dB against benchmarks/reference_v2/hero.png; looked at hero.png and the
x10 diff: no tile seams or blocky artifacts, the diff is thin edges (spokes, leaves) only.

Note: the hero forward's PSNR against the JPG is 13.1 dB (`hero_meta.json`), far below a
trained model's ~25 dB on this view. The camera or colour path of the dump may not match the
dataset image; the oracle check does not depend on it, but a training step would.

## Footprint (v2, analytic from the code)

L1 per core (CBs): HDR 2x64 B, PLANE 5x4 KB, RAMP 2x4 KB, REC 4x64 B, OUT 2x4 KB, writer
scratch 4 KB. Total 41.3 KB of about 1.4 MB on a p100 (1.5 MB on a p150). Dst: 8 fp32 tiles.
Tile 7 is ACC; vectors 0-8 are the per-lane sums and 24-29 the 16 finished pairs.

DRAM per view: tiles 64 B/tile, recs 64 B/pair, planes 20 KB/tile, slots 64 B/pair
(36 B used).
- Hero 1237x822 = 39x26 = 1,014 tiles: planes 20.8 MB, plus 128 B per pair (recs + slots),
  i.e. 128 MB per 1M pairs.
- Traffic per pair: the reader and the writer each read the 64 B record once (128 B), and the
  writer writes 64 B.
- Planes: T, BH and dL/dimage (3) are written once per tile by prep or the forward, and read once.

## T1 v3 levers (sized from the above, not yet measured)

1. **Pass the slot index reader to writer through a small CB.** This drops the writer's second
   64 B record read per pair: a third of the per-pair DRAM traffic.
2. **Records from the forward's slab.** The training-mode forward keeps per-tile pair lists and
   live masks. Prep then produces nothing per pair, and 4 B mx/my/A/B/C/op/rgb are read from the
   forward's per-Gaussian buffer.
3. **Cost per live vector.** The compute body is the cost driver (about 200 cycles per vector).
   Profile with in-kernel cycle counters split into setup, bodies, finish and pack sync. Then
   cut with the accurate exp only where alpha >= 1/255 (v_if), and batch the reciprocal.
   Lossless only, checked against the T0 oracle.

## T2 proposal (against the v2 slot layout; coordinate with #697, do not duplicate it)

v2 writes pair j of Gaussian g to slot P[g] + j, where P is the exclusive prefix sum of the
per-Gaussian tile-rectangle sizes. Each Gaussian's 9 partials are therefore contiguous, and T2
is a deterministic segment reduce with no atomics.
- One pass per core over a contiguous range of visible Gaussians:
  1. Read slots P[g]..P[g+1] (64 B pages; consecutive, so one NoC burst per Gaussian).
  2. Sum the 9 terms in fp32.
  3. Apply the projection backward (conic to cov2d, then EWA Jacobian to means3d, scales and
     quats) and the SH0 colour backward.
  4. Run sparse Adam on the visible set.
  5. Write the updated parameters directly into pfwc's input layout (PLAN.md row T2).
- Gradients for the parameters never reach DRAM.
- DRAM per step: 64 B x pairs (slots) + per visible Gaussian (params + 2 Adam moments) read and
  written.
- First check: one step on a 10k-Gaussian synthetic scene. Updated params must match T0's Adam
  step to rel 1e-3, starting from the device slots of s2.

Quality rule (msgs #384/#387): every step is lossless and must pass the T0 gradient oracle.
The training quality bar is published 3DGS/gsplat or LiteGS bicycle test PSNR/SSIM/LPIPS
(published numbers).

## Not done (moved to the follow-up)

- Hero oracle failure (#707): find and fix the cause, then re-time (see the hero section).
- Forward training flag (fp32 image, accurate exp, slab kept alive), with the flag-off b2b
  raw_md5 1d372b94 check and the on/off A/B (opt/ab_jit_check.py on both arms, plus a device
  hero.png, a diff and PSNR against benchmarks/reference_v2/hero.png).
