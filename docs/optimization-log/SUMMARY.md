# Optimization Log Summary — Alpha-Blend Kernel on Blackhole P150

**Target device:** Tenstorrent Blackhole P150 (`bh-30`)
**Branch:** `smarton/optimization`
**Last updated:** 2026-05-21 (after iter 012)

## Headline progress

| Iter | Optimization | Status | Kernel ms (stitch hero) | Δ vs prev | Cumulative |
|------|--------------|--------|------------------------:|----------:|-----------:|
| 000  | Baseline (bh-30) | locked | **68.92** | — | — |
| 001  | Block-wide early term | **reverted** (no-win) | 69.86 | +1.4% | 0.0% |
| 002  | Stage B2 fusion (acquire-only) | **reverted** (no-win) | 69.34 | +0.6% | 0.0% |
| 003  | Collapse B2+B3a; drop CB_DX²/DY²/DXDY | **KEPT** | 65.79 | −4.5% | −4.5% |
| 004  | Absorb sat_mask into T_state | **KEPT** | 63.12 | −4.1% | −8.4% |
| 005  | Stage E: `T -= CB_CONTRIB` (1 acquire) | **KEPT** | 61.03 | −3.3% | −11.4% |
| 006  | SFPU `add_binary_tile` D2 fuse | **reverted** (no-win) | 62.36 | +2.2% | −11.4% |
| 007  | Batch Stage D2 producers | **KEPT** | 59.10 | −3.2% | −14.2% |
| 008  | Batch Stage D2 adders | **KEPT** | 57.97 | −1.9% | −15.9% |
| 009  | `copy_dest_values` D1+D2 fuse | **reverted** (hard-reject) | 58.49 / PSNR 15 | regression | −15.9% |
| 010  | Fold B3b1 into B2+B3a via `add_binary_tile` | **KEPT** | 57.31 | −1.1% | −16.8% |
| 011  | Fuse D2 adder + Stage E (4-slot dst) | **KEPT** | 56.54 | −1.3% | −18.0% |
| 012  | D1+D2 producer FPU recompute fuse | **reverted** (no-win) | 56.90 | +0.6% | −18.0% |

**Net kernel improvement:** `68.92 ms → 56.54 ms` = **−18.0% on stitch hero**
**Acquires per Gaussian:** ~10 → **6** (+2/16 for Stage F refresh)

## Multi-scene validation (iter-011 kernel)

| Scene      | View | Kernel ms | Iter-000 ms | Δ        | PSNR  | SSIM  | image_diff |
|------------|------|----------:|------------:|---------:|------:|------:|:----------:|
| stitch     | hero | **56.54** | 68.92       | −18.0%   | 44.00 | 0.988 | clean-keep |
| luigi      | hero | **12.82** | 15.11       | −15.2%   | 37.16 | 0.980 | clean-keep |
| strawberry | hero | **207.44**| 250.33      | −17.1%   | 46.18 | 0.993 | clean-keep |

All scenes pass the permissive two-gate visual check. No regressions.

## Profiler breakdown (current state, stitch hero)

| Stage of pipeline | Median ms | Share |
|-------------------|----------:|------:|
| `sub.blend.prep` (host packs attributes) | **71** | 50% |
| `sub.blend.daemon_rt.device_kernel` | **58** | 41% |
| Daemon stdin/stdout + misc | 14 | 9% |
| **Total daemon round-trip** | **143** | 100% |

Inside the 58 ms `device_kernel`:

| Component | Median ms | Notes |
|-----------|----------:|-------|
| On-chip TRISC busiest-core | **~25** | Real per-core compute time (down from ~39 ms on iter-000 baseline) |
| Dispatch / sync / grid-completion overhead | **~33** | Kernel program launch + wait + per-tile dispatch + final readback |

**Implication:** ~57% of host-measured `device_kernel` is now dispatch overhead, not compute. The kernel itself is already running near the on-chip floor for the current algorithm.

## Lessons learned (informing future iters)

1. **FPU and SFPU dst layouts are incompatible.** `copy_dest_values<Float16_b>(idst_in, idst_out)` after an FPU `mul_tiles` produces garbage. Iter 009 hit this trap → PSNR 15 dB.
2. **SFPU dst-to-dst compute ops are expensive (`~630 cycles/entry` for `add_binary_tile`).** Only worth it when it eliminates a full acquire/release (iter 010), never as a per-channel inner loop op (iter 006).
3. **Acquire-block fusion alone (without eliminating intermediate CBs / `copy_tile`s) doesn't pay.** Iter 002 showed this; the wins in iters 003, 007, 008, 011 all came from genuinely eliminating L1 round-trips, not just merging blocks.
4. **FPU `mul_tiles` does not pipeline for free.** Iter 012 added 3 extra `mul_tiles` to save 1 acquire and net-regressed +0.6%.
5. **Front-to-back compositing has the identity `T·(1-α) = T - α·T`.** Reusing the already-computed `α·T` (CB_CONTRIB) as `T -= contrib` saved one full stage (iter 005).
6. **Sat-mask is redundant if T is hard-zeroed.** Once T saturates near 0, downstream multiplications keep it dead. Iter 004 eliminated CB_SAT_MASK entirely.

## Gap to theoretical peak

`scripts/binning_log.py` (Blackhole P150, 130 cores @ 1.4 GHz, realistic 200 cyc/pair):

| Scene      | View | Current ms | Realistic peak ms | Slowdown vs peak |
|------------|------|-----------:|------------------:|-----------------:|
| stitch     | hero | 56.54      | 0.91              | **62×**          |
| luigi      | hero | 12.82      | 0.19              | 67×              |
| strawberry | hero | 207.44     | 3.47              | 60×              |

We have closed the gap from ~75× to ~62× via on-chip optimizations.

## Why kernel-only optimizations are stalling

Per the profiler split: 58 ms `device_kernel` = 25 ms on-chip + 33 ms dispatch overhead. Even if we drove on-chip to 0 ms, host-measured kernel time would not drop below ~33 ms. So the remaining headroom in purely kernel-side iters is bounded at roughly **25 ms** — and we're already getting diminishing returns on each iter (recent kept iters delivered ~0.7–1.1 ms each, four reverts in twelve iters).

## Next-direction options for the user

To approach the user's **1–10 ms total per frame** target, kernel-only work won't suffice. The big remaining levers are:

1. **Host prep (71 ms):** `prepare_kernel_inputs` packs Gaussian attributes for the kernel every frame. If the underlying Gaussians don't change between frames (typical: camera moves but scene is static), most of prep can be cached.
2. **Dispatch overhead (~33 ms):** Reduce kernel-launch / per-tile-grid-sync / readback overhead. Possible via fewer program launches, persistent kernels, or larger work units.
3. **Algorithm changes:** Per-tile Gaussian culling, 16×16 tiles, screen-space binning on host. Both reduce useful work and change the kernel's input shape.
4. **More aggressive precision:** `float16`/`bf8`/`int8` for some intermediate values (small SSIM/PSNR cost, sometimes large perf cost).

Recommendation: pivot the next phase to **prep caching + dispatch reduction**, with occasional kernel iters when a clean opportunity appears.
