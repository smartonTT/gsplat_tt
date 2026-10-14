# #748 on-device training step: T1 -> T2 -> Adam chain (first part)

Branch `ttp/t748-minimal-on-device-bicycle-training-step-`. Device: bh-49 p150a (measurement
reservation, `ttp lock p100`), run 2026-10-14 08:57 PDT at commit 6d808b6f (SYNC_STAMP MATCH).
SH bands 1-3 (#751/#756): branch `ttp/t751-t2-sh-bands-1-3-and-full-colour-clamp-ma`, run
2026-10-14 10:56 PDT on bh-49 at commit 9c467a44.

## What runs on device

`render/train/train_step <dir> <out> [cores] [reps] [grads]` runs in one process, one device:

1. **T1 blend backward v2** (`bwd_reader` / `blend_bwd_compute` / `bwd_writer`). The writer is
   built with `BWD_SLOT_PAGE=4096`. It writes slot `s` straight into T2's slot-page layout:
   page `s/64`, offset `(s%64)*64`.
2. **T2 param pass + sparse Adam** (t697 gather reduce: `t2g_reader`, `t2_writer`,
   `t2g_compute`). It reads the same slot buffer, so **nothing is re-packed on the host between
   T1 and T2** (`host_repack_ms` = 0). The buffer is filled with NaN before T1, and the check
   counts any slot T1 left unwritten.
3. **SH mode** (only when `<dir>/t2/sh_rest.f32` exists; otherwise the step is exactly the
   sh0 chain above). Per step: T1 -> **t2sh pre** -> T2 (`T2_SH`, grads out) -> **t2sh post**.
   - pre (`t2sh_reader/compute/writer`): SH basis for bands 1-15 from the view direction
     (`sh_consts`: campos, rest scale, Adam betas, eps), full colour with the clamp-keep mask,
     and the view-direction Jacobian. Writes one aux page per group.
   - T2 with `T2_SH` reads the aux page: adds the direction term to dL/dmeans and applies the
     full-colour clamp mask to dL/dsh0.
   - post (`T2SH_POST`): rest-band grads from the T2 grads stream and sparse Adam on the rest
     bands (visible Gaussians only). Outputs `sh_rest_out.f32`, `sh_m_out.u16`,
     `sh_v_out.u16`, `sh_grads.f32`.

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
| t0_sh3 | 1.0e-4 | 0 | 2.7e-4 / 8.8e-6 | 3.9e-4 | 5.5e-4 | PASS (all bands gated) |

sh3 per SH band vs T0 (worst relative):

| band | grad | param after Adam |
|---|---|---|
| 0 | 1.5e-5 | 1.2e-7 |
| 1 | 1.5e-5 | 8.4e-8 |
| 2 | 1.5e-5 | 7.1e-8 |
| 3 | 1.5e-5 | 1.1e-6 |

Bars: slots 1e-2, grads 1e-2, Adam 1e-3 (relative). Full reports: `out/check-*.json`.

Before #756 the sh3 T0 check failed (means 0.57, sh0 1990): T2 had no view-direction term
in dL/dmeans and took the clamp from band 0 alone. With the t2sh passes both are fixed and
T0 is gated on all bands. sh0 numbers are unchanged with the SH code in the binary.

**A full view (hero) has not been checked yet.** The hero dump (`docs/t685/hero.py`) has no
T2 inputs (3D params, Adam state).

## Timing (crop only; not a ms/view number)

`out/timing-sh*.json`, 10 reps, 110 T2 cores, 4 T1 cores:
T1 0.058 ms, T2 0.043 ms, chained T1+T2 (one Finish) 0.085 ms, host re-pack 0.
SH3 (#756, min of 10): T1 0.057, t2sh pre 0.019, T2 (T2_SH) 0.043, t2sh post 0.028 ms;
chain 0.104 ms. sh0 rerun on the same build: T1 0.058, T2 0.044, chain 0.085 ms. On a 64x64
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
3. Done in #756 (SH bands 1-3 via t2sh pre/post). Next for SH: fold pre into T2's reader
   group walk to save one dispatch (~0.02 ms on the crop).
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
row "fp32-ill" when the fp32 reference misses fp64 by more than grad_bar/10. The error floor
(1e-4 of the tensor max) is taken from the kept rows and the split is iterated: with the
whole-tensor max the ill rows' ~1e5 grads hid 13 milder ill rows (fp32 vs fp64 up to 0.37,
e.g. row 1611162). Grads are gated on the other rows. Adam, moments and pfwc are gated given the
device's own grads, on all rows: grads that agree to 3e-3 still flip bf16 moment roundings (one
bf16 ulp, up to 2^-7 relative), so the end-to-end Adam comparison is reported (`adam_e2e`), not
gated. Ill rows are reported against fp64
(`ill.*.dev_vs_f64`, `ref32_vs_f64`); the gate also needs their count <= max(16, N/10000) and
the device finite on them. `grads_all_rows` keeps the ungated numbers. `out/t2_rows.npz` holds
the ill rows plus the 256 worst gated rows (dev, ref32, ref64 grads and g2d) for debugging.

**Hero T2 result (device run at 1a591bdf on bh-49, checked with the gate above).** T1 PASS
(rel 4.4e-3). T2 PASS: 94 fp32-ill rows (cap 180), device finite on all of them. Gated grads
rel: means 2.6e-3, log_scales 3.5e-3, quats 2.7e-3, opacity 1.1e-3, sh0 1.6e-7 (bar 1e-2).
Adam given device grads 4.8e-6, moments bit-equal, pfwc 5.1e-5 (bar 1e-3). End-to-end Adam
(reported): 5.8e-3 from bf16 rounding flips. On the ill rows the device is closer to fp64 than
the host fp32 reference (means 4.0e4 vs 3.6e5). Full report: `out/check-hero-t2.json`. The
device outputs are those of the 12:36 PDT run; only the check changed.

## #781: first dump-free hero chain (forward -> loss -> T1/T2), 2026-10-14

Driver: `docs/train-step/drive_loop.sh 11d4bf19` on bh-49 (p150, under `ttp lock p100`), log
`out/hero7-driver.log`, per-stage outputs in `out/loop/` (crops) and `out/loop-hero/` (hero).
The forward prep is binned and vectorized (`blend_forward_dev.prep_binned`, unit-tested against
`prep` on crops); the hero forward gate is the numpy model of the device math
(`hero_model.py`, `emulate_fast`), computed once on the Mac.

| run | forward vs oracle | loss grad vs oracle | T1 | T2 | chain gate |
|---|---|---|---|---|---|
| sh0 64x64 crop | 3.4e-7 (pass) | l2rel 6.1e-7 (pass) | pass | pass | pass, T0 pass |
| sh3 64x64 crop | 4.1e-7 (pass) | l2rel 5.7e-7 (pass) | pass | pass | pass (T0 flag false, see below) |
| hero 1237x822 | **fail**: 1.4e-2 vs model, 1.7e-2 vs dump | **no oracle** (see below) | pass | pass | T1/T2 pass vs device dL/dimage |

Hero device times (median of 5, single p150, no host round trips between stages):

| stage | ms | note |
|---|---|---|
| forward (blend_fwd_bench, 4.48 M records, 1014 tiles) | 440.4 | not oracle-clean, so not a headline |
| loss (D-SSIM+L1, 3042 tiles, 110 cores) | 0.96 | b2b; fwd part 0.44 |
| T1 | 81.4 | |
| T2 | 98.4 | |
| T1+T2 chain | 179.7 | was 186.2 in #770 |

Open problems found (none of these numbers counts as a training ms/view yet):
1. Hero forward misses the gate. 128 of 1014 tiles hold more than 8192 records (max 25069) and go
   through the CONTINUE sub-chunk path that the 64x64 crops never reach; the device image was
   deleted on the box before the pull, so the error is not yet localized to those tiles.
2. The hero loss stage has no oracle: `chain_dump` compares with the dump's `dL_dimage`, which at
   hero size is not a D-SSIM+L1 gradient (the dump has no `loss` key), hence l2rel 3.7. A host
   numpy D-SSIM+L1 gradient of the device image against the target is needed.
3. Forward at 440 ms is 2.4x the whole backward; a sustained ms/view (step 4) and the 1-2k-step
   loss curve (step 5) are not run.
4. sh3 crop, T0 flag false: every gradient matches T0 to about 1e-5 except `means`
   (max rel 0.57, l2rel 1.0e-2; Adam delta on means 0.26). The sh0 crop passes, so the likely gap is
   the SH-direction term of dL/dmeans (view-dependent colour) or the raw<0 colour mask on it
   (decision #763); `out/loop/loop-sh3/out/check.log`.
