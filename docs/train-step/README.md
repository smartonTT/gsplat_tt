# #748 on-device training step: T1 -> T2 -> Adam chain (first part)

Branch `ttp/t748-minimal-on-device-bicycle-training-step-`. Device: bh-49 p150a (measurement
reservation, `ttp lock p100`), run 2026-10-14 08:57 PDT at commit 6d808b6f (SYNC_STAMP MATCH).

## What runs on device

`render/train/train_step <dir> <out> [cores] [reps] [grads]` runs in one process, one device:

1. **T1 blend backward v2** (`bwd_reader` / `blend_bwd_compute` / `bwd_writer`). The writer is
   built with `BWD_SLOT_PAGE=4096`. It writes slot `s` straight into T2's slot-page layout:
   page `s/64`, offset `(s%64)*64`.
2. **T2 param pass + sparse Adam** (t697 gather reduce: `t2g_reader`, `t2_writer`,
   `t2g_compute`). It reads the same slot buffer, so **nothing is re-packed on the host between
   T1 and T2** (`host_repack_ms` = 0). The buffer is filled with NaN before T1, and the check
   counts any slot T1 left unwritten.

Not on device yet: the forward render and the fused L1 + D-SSIM loss. Here the image and
dL/dimage come from the T0 dump through the T1 prep planes. These are the next stages.

Host side: `gsplat/train_step_dev.py prep|check`, driver `docs/train-step/drive.sh <rev>`.
T2 uses T1's slot numbering (3-sigma tile rectangles). The T2 oracle's frozen-live
rectangles give a different numbering (74 vs 72 slots on the crop), so the prep builds T2's
inputs from T1's `blend_backward` output.

## Oracle results per stage (64x64 crop, 50 Gaussians, T0 dumps)

| dump | T1 slots vs blend_backward_ref | NaN slots | T2 vs t2_step (worst grad / Adam) | T0 autograd grads (worst) | T0 bf16 Adam params/delta (worst) | result |
|---|---|---|---|---|---|---|
| t0_sh0 | 3.7e-5 | 0 | 1.7e-4 / 1.0e-6 | 4.5e-4 | 1.4e-4 | PASS |
| t0_sh3 | 1.0e-4 | 0 | 2.7e-4 / 4.9e-6 | means 0.57, sh0 1990 | means delta 0.51 | T1/T2 PASS; T0 not gated |

Bars: slots 1e-2, grads 1e-2, Adam 1e-3 (relative). Full reports: `out/check-*.json`.

For sh3, T0 is reported but not gated. T2 handles SH band 0 only. The oracle's dL/dmeans
includes the view-direction term from SH bands 1-3. Its dL/dsh0 is zero wherever the
**full** colour clamps (e.g. Gaussian 1, green channel). T2 computes the clamp from band 0
alone. Both are real gaps for SH-3 training (see levers).

**A full view (hero) has not been checked yet.** The hero dump (`docs/t685/hero.py`) has no
T2 inputs (3D params, Adam state).

## Timing (crop only; not a ms/view number)

`out/timing-sh*.json`, 10 reps, 110 T2 cores, 4 T1 cores:
T1 0.058 ms, T2 0.043 ms, chained T1+T2 (one Finish) 0.085 ms, host re-pack 0. On a 64x64
crop this is almost all dispatch overhead. The sustained multi-view bicycle ms/view run and
the 1-2k-iteration loss curve have not been done yet.

## #750: hero full view and the dump-free chain

Hero (bicycle hero view 1237x822, 1,809,729 visible rows, 4,478,225 slots, 1014 tiles), run
2026-10-14 11:07 PDT on bh-49 before the radius fix:

| stage | result |
|---|---|
| T1 blend backward v2 vs ref_slots_fast(ties) | PASS, rel 4.4e-3 (bar 1e-2), 0 NaN slots, 55,108 tie slots |
| T2 grads vs t2_step | opacity 3.6e-4, sh0 1.4e-6; means/scales/quats NaN in the *reference* |
| T2 Adam m/v/params | FAIL: m = v = 0 on 77 rows, all with 663-1014 slots |
| timing (5 reps) | T1 87.8 ms, T2 98.4 ms, chain 186.2 ms per step |

Two causes, both fixed in this commit (device rerun pending):
- `t2_radius` in t2g_compute.cpp / t2_compute.cpp clamped the 3-sigma radius at 30000 px. The
  bad rows are near-camera Gaussians centred up to 46,000 px off screen whose footprint still
  covers every tile, so the device marked them invisible and sparse Adam skipped them. Now the
  ceil is exact at any size (int16 path to 30000, 2^23 add/subtract above it).
- The host reference gave NaN grads on 4 rows with no slots and a degenerate cov2d (0 * inf).
  A row with no slots now gets exactly zero grads, as the device writes
  (test_degenerate_cov2d_row_without_slots_has_zero_grads).

Hero rerun after the radius fix (3de1fafc, 2026-10-14 12:06 PDT on bh-49, log runs/2167/hero_chain3.log):

| stage | result |
|---|---|
| T1 blend backward v2 | PASS, rel 4.4e-3, 0 NaN slots (unchanged) |
| T2 grads | opacity 3.6e-4, sh0 1.4e-6 PASS; means 2454, log_scales 52.6, quats 9.6 FAIL (relative) |
| T2 per-Gaussian world cov (pfwc) | c00..c22 up to 5.4e-2, mx 1.6e-2 |
| T2 Adam | m/v follow the grads (v_means 34152); opacity/sh deltas fine |
| timing (5 reps) | T1 87.8 ms, T2 98.4 ms (first call 1819 ms), chain 186.2 ms per step |

No more m = v = 0 rows, so the radius fix took. The host reference still divides by zero in
`t2_param_backward.py:108` (idet of a degenerate cov2d) on rows that do have slots, which makes
its means/scales/quats grads inf/NaN-adjacent; the remaining gap is either those rows or a real
device error in the cov3d->cov2d backward. Not yet split: next step is a per-row compare on the
box (finite-reference rows only, then the degenerate rows alone).

Dump-free chain on the crops (blend_fwd_bench -> loss_bench LOSS_MODE=ssim -> train_step,
loop_box.sh): forward image vs oracle 8.6e-6 / 7.7e-6 (sh0 / sh3, bar 1e-4), device loss equals
oracle loss to 7e-6 relative, dL/dimage max error 3.2e-5 of its max (fp32 device vs fp64 oracle).
T1 and T2 PASS on both. T0 on sh0 first failed at sh0 grad 1.02e-2 (bar 1e-2, floor 1e-4 of max).
The host fp32 reference fed the same device image and loss gradient misses autograd by the
same 0.98e-2, at an element 3.8e-4 of the tensor's max, so the gap is the fp32 loss gradient,
not the device chain. Chain dumps now carry `chain_dl_dimage_max_rel` and their T0 gate uses:
relative error floored at 1e-3 of the max (bar 1e-2), grad L2 relative error <= 1e-3, Adam delta
<= one bf16 ulp (2^-8), loss gradient <= 1e-4 of its max. sh0 then passes: worst grad 3.8e-3,
grad L2 4.9e-5, delta 3.5e-3. sh3 T0 stays reported, not gated (SH bands 1-3 are a separate task).

Crop host steps per view (loop_box.sh STEP lines): fwd prep ~65 ms, loss prep ~58 ms (target
constants 3.9 ms, once per dataset), chain dump ~64 ms, T1/T2 prep ~80 ms. These are python
process launches on 64x64 crops; the device stages run 0.16 ms (forward), 0.069 ms (loss b2b)
and 0.086 ms (T1+T2). Not a ms/view number yet.

## Ranked next levers

1. Hero full-view chain check. Extend hero.py with T2 inputs: ply rows of `sel` as params,
   band-0 SH, Adam state at step 0. Then T1 check with `ref_slots_fast(ties=True)` and T2
   check with `t2_dev.check`.
2. Train-mode forward on device (blend_fwd_bench, #716) writing T1 records/planes, plus the
   fused L1 + D-SSIM loss kernel (#736/#742/#743) writing dL/dimage. This removes the dump
   from the loop.
3. SH bands 1-3 in T2 (or a separate pass), with the full-colour clamp mask from the forward.
   Needed for SH-3 training and the T0 gate on sh3.
4. Sustained NN-ordered multi-view timing (benchmarks/cameras_train_*.json), then the short
   training run with the loss curve.
5. T2 FPU reduce with an on-device block-float split (#740), replacing the gather reduce.

## #770: colour convention and the hero T2 mismatch

**Colour convention (decision 2026-10-14, review #763).** The trainer follows 3DGS/gsplat:
training colour = max(SH + 0.5, 0), with no upper clamp. SH colour gradients are masked only
where raw < 0. This holds in `gsplat/train_oracle.py`, `gsplat/t2_param_backward.py` (grads and
pfwc colour) and `render/train/kernels/t2g_compute.cpp` / `t2_compute.cpp` (grad mask and pfwc
epilogue) together. The renderer's display clamp to [0, 1] is unchanged. The sh3 crop dump was
regenerated (it had 2 colours >= 1); the sh0 dump is bit-identical. `t2sh_compute` (higher SH
bands) is on the t751/t756 branches, not here, and still needs the same change.

**Root cause of the #750 hero T2 FAIL (means 2454, log_scales 52.6, quats 9.6).** It is not the
degenerate-cov2d rows: the reference has no non-finite row that has slots, and the 4 degenerate
rows have no slots and are zeroed on both sides. The misses are about 64-81 of 1.8M rows:
Gaussians just past the near plane (z 0.20-0.23) whose centres are 4,000-195,000 px off
screen. For them dL/dconic is ~1e8 against a conic of ~2e-5, and dL/dcov2d = -Q G Q cancels to
~1e-7 of its terms. fp32 cannot give these grads in any order of operations: the host fp32
reference itself misses the fp64 evaluation of the same math by 3.6e5 (means), 3.7e6
(log_scales) and 2e4 (quats). Example row 953138: fp32 dL/dmeans [450, -47, 367], fp64
[-0.13, 0.014, -0.11]. The same happens in any fp32 trainer (3DGS/gsplat CUDA included). On the
other rows fp32 vs fp64 is ~1e-3 L2.

**Gate.** `t2_dev.check` now evaluates the reference math in fp64 (`ft=np.float64`) and marks a
row "fp32-ill" when the fp32 reference misses fp64 by more than grad_bar/10. Grads, Adam, pfwc
and moments are gated on the other rows. Ill rows are reported against fp64
(`ill.*.dev_vs_f64`, `ref32_vs_f64`); the gate also needs their count <= max(16, N/10000) and
the device finite on them. `grads_all_rows` keeps the ungated numbers. `out/t2_rows.npz` holds
the ill rows plus the 256 worst gated rows (dev, ref32, ref64 grads and g2d) for debugging.
