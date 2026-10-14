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
