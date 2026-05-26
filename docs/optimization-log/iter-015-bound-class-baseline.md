# iter-015 — Bound-class baseline (measurement-only, --profile)

- Class: kernel-algebra (code identical to iter-010 baseline; this is pure measurement)
- Track: post-iter-014-revert / tt-buddy-pivot
- Date: 2026-05-26
- Status: PLANNED
- Predecessor: iter-010 e-fuse-fpu (KEEP, 97.77 ms / 40.40 / 43.70 / 40.15 dB)

## Why this iter exists

Iters 007–014 were all kernel-algebra fusion attempts on the inner Gaussian
loop (FPU acquire merges, binary_dest_reuse triads, init coalescing,
sat-mask removal). Two deadlocked outright (iter-013, iter-014). Net win
across 8 iters: under 2 ms vs the iter-007 baseline (~99 ms → ~97.8 ms).

We have never confirmed *what* the kernel is bound on. Every fusion
hypothesis implicitly assumed compute-bound, but tt-buddy's profiler skill
calls out exactly this anti-pattern:

> "Compute-bound levers on an overhead-bound op waste iterations."

This iter is **pure measurement**, no code change. Build with
`-DENABLE_TRACY=ON`, run the standard trajectory benchmark with
`TT_METAL_DEVICE_PROFILER=1`, and pull back the per-RISC kernel durations
that tt-metal auto-emits to `generated/profiler/reports/*/ops_perf_results_*.csv`.

## Hypothesis (about ourselves, not the kernel)

The kernel is **not** compute-bound. Likely classifications:
- **reader-bound** (BRISC ≳ TRISC1): the per-Gaussian inner loop reads
  9 fp32 attributes from CB_SCALARS, plus the per-tile px/py. With 10K
  Gaussians × 64 cores the reader has substantial sustained traffic.
- **overhead-bound** (`(FW - KERNEL) / FW > 40%`): the kernel has many
  small acquire/commit cycles per Gaussian (currently ~10 acquires per
  Gaussian after the iter-007/iter-010 fusions). Dispatch + CB-flush
  could dominate.

If either is true, the entire iter-007..014 lever family was misaimed.
We need a different lever family (CB depth, batched reads, kernel
restructure) to make further progress.

## Bound-class taxonomy (from tt-buddy interpretation.md)

| Tag | Signal |
|---|---|
| reader-bound | BRISC ≳ TRISC1 |
| compute-bound | TRISC1 (math) dominates |
| writer-bound | NCRISC dominates |
| under-parallelized | low CORE COUNT + high per-core duration |
| NOC-stall | BRISC + NCRISC high, TRISC low |
| host-dominated | FW short, HOST DURATION large past first iter |

Derived metric: `overhead_ratio = (FW - KERNEL) / FW`
- < 15%: compute/bandwidth dominates
- 15-40%: mixed
- > 40%: dispatch/sync/CB-flush bound

## Procedure

1. Run `scripts/run_iter.sh 15 bound-class-baseline kernel-algebra --profile`.
2. Harness:
   - Builds `build-tracy/` with `-DENABLE_TRACY=ON`.
   - Sets `TT_METAL_DEVICE_PROFILER=1`.
   - Wipes `generated/profiler/reports/` before run.
   - Runs render_fixed.py (1 warmup + 10 measure cycles).
   - scps back `device_profile/ops_perf_results_*.csv` and `profile_log_device.csv`.
3. Analyze the per-op CSV: rank by `DEVICE FW DURATION [ns]`, read each
   RISC's `DEVICE *RISC* KERNEL DURATION [ns]` column.
4. Compute `overhead_ratio = (FW - max(RISC durations)) / FW`.
5. Tag the bound class.

## Validation gate

This is a measurement iter — there is no PSNR / ms gate beyond "the run
produced a valid render and didn't deadlock." If kernel ms drifts > 5%
vs iter-010 baseline (97.77 ms), the Tracy overhead is suspicious and
we should compare against a non-tracy run to factor out the
instrumentation cost.

## Files edited

None (kernel code identical to iter-010).

`scripts/run_iter.sh` was modified in a prior commit (35605ca) to scp
back the device profile CSVs; that change is foundational for this iter
and any future --profile run.

## Outcome

(to be filled in after run)

- per-RISC durations:
- overhead_ratio:
- bound-class tag:
- next-iter lever family:
