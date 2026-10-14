# Trainer ms/view breakdown and lever plan (task #773, 2026-10-14)

Plan only. No device work was run for this doc. "M" means measured on device (p150a bh-49 unless
noted), with the task that measured it. "E" means estimate or model. "P" means published, not
measured. "Mac" means computed on the Mac from the hero dump or the dataset.

**Superseded in part (2026-10-14, #799):** T2 98.4 -> 9.29 (#784) and T1 82.4 -> 51.95 (#785) have
landed; the step is now ~76.8 ms/view. See docs/training-plan/t799/PLAN.md for the current breakdown
and lever ranking.

## 0. Correction: 14.7 ms/view is the forward, not the step

The "~14.7 ms/view" in the task title is the training-mode forward only
(`GSPLAT_TT_BLEND_TRAIN=1`, fp32 op/colour), b2b (#745 14.700, #755 14.699, #727 14.681, #724 14.669).
One full training step on the hero view (_DSC8679, 1237x822, bicycle 6.13M) is about **196 ms/view**.
That is 13x the forward and about 100-200x the 1-2 ms target. The backward is 92% of it.

## 1. Per-stage breakdown of one training step (hero view, single p150a)

| stage | ms/view | kind | source |
|---|---:|---|---|
| forward, train mode (b2b) | 14.70 | M | #745, #755 |
| - of which blend | 13.42 | M | #755 |
| - of which project/TA/sort/gather | ~1.28 | E (14.70 - 13.42) | |
| - train flag cost vs render forward (7.41) | +5.52 | M, not split | #712 |
| - fp32 op/colour delivery (inside the +5.52) | 1.76 (gather 1.06, L1 copy 0.36, compute 0.34) | M | #724, #727 |
| loss: fused L1 + D-SSIM (K=4 + VF) | 0.89 | M, synthetic 1237x822 | #753 |
| blend backward T1 (warm median) | 82.39 | M | #767 (BWD_PRESTAGE=2) |
| - busiest core: bodies / pair setup / finish / waits | 73.07 / 5.30 / 3.49 / 0.74 | M | #747 |
| - mean core | ~49.7 | E from #728 + #747 | |
| projection + SH backward T2 (gather, warm) | 98.4 | M | #750 |
| - **T2 oracle fails on hero** (means/scales/quats rel 2454/52.6/9.6) | | M | #750, fix queued as #770 |
| optimizer (Adam) | **gap** | not measured on device in a step; ~3.9 dense, ~1.2 sparse+Morton | E #689 |
| host/dispatch between stages | **gap** | chain T1+T2 = 186.2 vs 82.4+98.4 = 180.8; ~5 ms unexplained | M #750 / E |
| **total** | **~196** | E (sum of M parts, optimizer and host gaps excluded) | |

Gaps that need measuring before the totals can be trusted:
1. No sustained, dump-free multi-view training loop has been timed. Every number above is a per-stage
   run on the hero. Host and dispatch time between stages is unknown.
2. The +5.52 ms train-flag cost is not split. It mixes the unfused program, accurate exp, fp32 RGBT
   writes, and SCHED 0 / T_PERIOD 0.
3. The optimizer has not been run on device inside a step.
4. T2 is wrong on the hero for means/scales/quats (#770). Its time could change once fixed.
5. Quality gate: `hero_meta.json` reports `psnr_vs_jpg_db` 13.10 (L1 0.172) for the bicycle.ply
   render vs the ground-truth JPG. A trained 3DGS bicycle model should give about 25 dB at this view.
   Either the camera/colour convention differs from the dataset or the PLY is not the matched model.
   Check on the Mac before any Q1 quality claim. (Not found as a known issue in memory or project.db.)

## 2. T2 model: why T2 is 98 ms (Mac, `t2_kmax_model.py`)

T2 groups 64 visible rows and runs one rank plane per slot index up to the group's largest slot count
(kmax). A model on the hero view (`t2_kmax_model.py`, output `t2_kmax_model.json`) reproduces the
device counts: 1,809,729 visible rows (exact), 4,490,511 slots (device 4,478,225, 0.3% off).

| quantity | value |
|---|---:|
| slots per row, 50/90/99/99.9% / max | 1 / 4 / 6 / 30 / 1014 |
| rows with k > 16 / > 64 / > 256 | 2,990 / 1,402 / 1,252 |
| rank planes (sum of group kmax) | 972,964 |
| padding factor (planes / mean k) | 13.9x |
| groups with kmax > 64, their planes | 889 groups, 787,329 planes (81%) |
| busiest core / mean core planes (contiguous split, 110 cores) | 34,292 / 8,845 |
| planes if rows capped at k=16 and the tail goes to a flat path | 191,492 |

About 1,250 very large Gaussians (k > 256 tiles each) pad their 64-row groups and make 81% of the work.
98.4 ms / 34,292 planes is about 2.9 us per plane on the busiest core (E). Balancing alone gives
about 25 ms (E). Capping k at 16 and summing the heavy tail on a flat path gives about 1,741 planes per
core, so T2 at about 5-8 ms (E).

## 3. Ranked levers

Ledger check: none of these repeats a shelved or landed lever without new evidence. Each is lossless
(fixed summation order, no approximation).

| # | lever | stage | bound (ms/view, hero) | evidence | ledger |
|---|---|---|---:|---|---|
| L1 | **T2 heavy-row split**: rows with k > 16 go to a flat contiguous segment-sum path; the rest keep rank planes with kmax ≤ 16; balance groups across cores by plane count | T2 | 98.4 → ~5-8 (**-90**) | Mac model §2 (E), device 98.4 (M) | new |
| L2 | **T1 load balance**: assign tiles by body count, split the heaviest tiles by microblock across cores (partial per-slot grads summed in T2) | T1 | 82.4 → ~50 (**-32**) | 1.68x imbalance (M #707), mean core ~49.7 (E) | listed in #685 v3, not done |
| L3 | **T1 body diet**: visit only vectors with live lanes; exp only where alpha ≥ 1/255; batched reciprocal | T1 | up to -30 to -40 (E) | 200 cycles/body (M #767) vs ~61 cycles/body in the forward (E); live pixels 36.0M (Mac) vs ~53M bodies x 32 lanes, so ~2-4% lane use (E) | listed in #685 v3, not done |
| L4 | **split then cut the train-flag +5.52**: fused program and T_PERIOD/SCHED in train mode with exact semantics | forward | ≤ -5.5 | +5.52 (M #712) | new split; fp32 delivery part shelved twice (#745, #755) |
| L5 | **whole-step fusion and SRAM residency**: one program per step, records read once for forward and backward, per-pixel T/C kept in L1 between forward and T1 | all | dispatch gaps (gap) + ~0.2-0.3 record re-read + ~5 chain overhead (E) | chain 186.2 vs 180.8 (M #750) | persistent kernel #579 was for render; training state differs, needs the §1 gap 1 timing first |
| L6 | **sparse Adam with Morton relabel** of Gaussians | optimizer | ~3.9 → ~1.2 (E) | 21% visible rows, 30% blocks after relabel (Mac #689) | new |
| L7 | fp32 op/colour from the sort slab (no gather) | forward | ≤ -1.06 | gather 1.06 (M #727) | shelved twice (#745 side slab, #755 RING3); only with new evidence |
| L8 | **published algorithmic levers** (P): budgeted densification to ~1-1.5M Gaussians (Taming-3DGS, LiteGS, FastGS), DashGaussian resolution schedule, fused per-Gaussian backward (gsplat "packed"/Faster-GS) | all | scales most stages with Gaussian count and pixels; Taming 16.2, Faster-GS 8.3, FastGS 3.84, LiteGS ~1.1 ms/iter on GPU (P) | PLAN-single-chip §2 | behind Q1 quality gate |

Not proposed (ledger, no new evidence): sort warm start (#688, ≤ 0.04), batch-union pre-cull (#688, parked),
B1 full route (#754, +4.19), FPU dense-GEMM blend, Morton chunk cull, u64 gather, per-view pfwc
compaction, precision cuts (#580), metal trace replay (#526, ≤ 0.083), FPU block-float T2 split
(slower than gather at K=1, #749).

Order of attack: L1 and L2 first (together about -120 of the 196 ms). After them the step is about
75 ms, with T1 bodies the largest part; then L3. L4-L6 matter once T1+T2 are under ~20 ms.
Even with L1-L6 the bound is about 20-30 ms/view at 6.13M Gaussians (E). 1-2 ms/view needs L8
(fewer Gaussians, as every fast published trainer does) on top.

## 4. Prep (off device) vs device A/B, per top lever

Rules for every spec below (restate in each task):
- Training levers must be lossless (msg #384). Fixed summation order; same results bit for bit
  or within the T0 tolerance.
- Every step passes the T0 gradient oracle (#674).
- Cross-view reuse only as an exact warm start, equal key for key to a full sort and checked on every
  view. No stale order, approximate cache or cull in training.
- Quality must match published 3DGS/gsplat or LiteGS bicycle test PSNR/SSIM/LPIPS at the same
  resolution and iteration count.
- Device work runs one at a time under `ttp lock p100 -- ...`, with the project's repo-scoped one-way
  rsync only; preflight with `ssh-preflight @p100`.
- Every A/B runs `python3 opt/ab_jit_check.py` on both arms. A delta under ~0.1 ms/view over 4 rounds
  is noise.

| lever | prep task (Mac, no device) | device A/B must show |
|---|---|---|
| L1 T2 heavy-row split | numpy reference of the split (capped rank planes + flat segment sum) in `gsplat/`, unit test vs the current T2 reference on the hero dump and on synthetic data, bit-equal sums; host-side group/plane assignment | hero T2 warm median 98.4 → ≤ 15 ms, T0 oracle passes for all params (after #770), 4 rounds |
| L2 T1 balance | per-tile body counts from the hero dump; LPT assignment by bodies; plan for splitting heavy tiles by microblock with partial slots; unit test of the partial-sum merge | busiest-core T1 ≤ 60 ms, mean/busiest ≥ 0.8, T0 oracle passes |
| L3 T1 body diet | count dead vectors and dead lanes per pair from the hero dump (ref forward `n_live`); model the saved bodies | T1 body cycles/pair down by the modelled share, T0 passes |
| L4 train flag split | none (needs device) | per-component split of +5.52 (profiling run, no code change) |
| L6 sparse Adam | Morton relabel + sparse Adam reference with a unit test vs dense Adam | optimizer ms/view on device in a step |

## 5. Profiling that should come first

1. Sustained, dump-free training loop over 30 dataset views: per-stage and host time per view
   (closes gaps 1-3). Device, one run.
2. Train-flag +5.52 split (L4). Device.
3. T2 per-core plane distribution and time per plane on device, to confirm the §2 model. Device,
   can ride with #770's fix run.
4. T1 dead-vector and dead-lane counts (L3). Mac, from the hero dump.
5. The 13.1 dB hero-vs-JPG check. Mac.

#772 (pair-setup sub-steps, 5.30 ms on the busiest core) is already queued and is not repeated here.
