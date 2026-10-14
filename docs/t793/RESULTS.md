# t793: T2 split host plan off the critical path (+ long-row split verdict): hero A/B

Commit measured: fcb0174a (branch ttp/t793-t2-split-split-long-rows-across-lanes-fi), base 9a30c819 (#784).
Box: bh-49 (p150a, measurement reservation), hero runs 16:02-16:06 PDT 2026-10-14, under `ttp lock p100`.
Driver: docs/t793/drive.sh; raw outputs: docs/t793/out/remote/ (log: state/runs/2241/t793.log).
Both arms run `T2_SPLIT=1` (kcap 16); cand adds `T2_PERVIEW=1` (plan rebuilt, uploaded and re-argued every step).

## What changed

- Host split plan is one pass over active groups; metab is patched in place and uploaded to b_meta
  (no 14.5 MB copy, no extra buffer); the fill search is bounded to [lo, lo+wmax]. Plan output is identical.
- Split runtime args are fixed-size and b_htab is sized for 2x the heavy rows, so a new view only
  rewrites buffers and args (no program rebuild). `T2_PERVIEW=1` times that per step, serial and
  overlapped with T1.
- Kernels unchanged: ab_jit_check --runtime-arg-only, same JIT hashes for t2g_reader, t2g_compute, t2h_writer.

## A/B (hero, 1014 tiles, 4,478,225 slots, 110 cores, 4 alternating rounds ABBA)

| metric (median of round medians, ms) | base | cand | delta |
|---|---|---|---|
| host plan (one build) | 6.59 (6.57-6.60) | 1.02 (1.02-1.03) | **-5.57 (6.4x)** |
| plan upload | 1.09 | 1.10 | 0 |
| T2 (A+B) | 9.21 | 9.21 | -0.002 (noise) |
| phase A / phase B | 2.06 / 7.19 | 2.05 / 7.18 | noise |
| T1+T2 chain | 90.50 | 90.50 | -0.001 |

Per-view mode (cand, T2_PERVIEW=1, host wall clock per step, 4 rounds):

| per step | ms |
|---|---|
| replan only (warm, in loop) | 1.18-1.33 |
| serial: plan -> upload -> args -> T1 -> A -> B | 93.20-93.30 (chain + 2.75) |
| overlapped: T1 enqueued first, plan+args on host while T1 runs | 91.55 (chain + 1.05) |

Overlap hides the plan and runtime args fully; the remaining +1.05 ms is the plan upload, which is
enqueued on the same command queue as T1 and so waits behind it. Moving it to a second CQ (or
uploading the next view's plan during the previous step's T2) should bring the per-view host cost
to ~0. `plan_stable` true on every rep (each replan equals the step-0 plan).

## Long rows across lanes: rejected

Phase A's floor is one 1,014-slot row. Splitting one row's sum across lanes reorders the fp32 sum,
so it cannot be bit-equal to the base (sequential) order, and the spec is lossless only. Modelled gain
was at most ~0.6 ms (phase A 1014 -> ~706 planes on the busiest core). Not implemented.

## Correctness

- Step-0 trainer outputs cand vs base on the hero (`same.json`): params_out, m_out, v_out, pfwc, grads **bit-equal**.
- T0 oracle crops, kcap 1 (every row heavy) and kcap 16, with T2_PERVIEW=1: sh0 t0/t1/t2 pass;
  sh3 t1/t2 pass, t0 ungated (`gated: false`, false at kcap 16 too where nothing is heavy), the same
  pre-existing state #784 reported. Overall pass on all four.
- JIT: JIT-HASH OK (runtime-arg-only); markers `[SPLIT] kcap 16` (base) and `[PERVIEW] plan_stable 1` (cand) found.

## Recommendation

Keep. Host plan 6.6 -> 1.0 ms with identical outputs and device time, and with overlap the per-view
host cost on the trainer critical path drops from ~7.7 ms (plan + upload) to 1.05 ms (upload only).
Next: put the plan upload on a second CQ / prefetch it one view ahead to remove the last 1.05 ms.
