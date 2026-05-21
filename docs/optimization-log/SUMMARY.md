# Optimization Log Summary — Alpha-Blend Kernel on Blackhole P150

**Target device:** Tenstorrent Blackhole P150 (`bh-30`)
**Branch:** `smarton/optimization`
**Last updated:** 2026-05-21 (Step 1 binary IPC + DRAM reuse @ 1024×1024)
**Benchmark convention going forward:** stitch hero at **1024×1024**
(`scripts/render_fixed.py stitch hero --resolution 1024x1024`).
Iters 001–013 were measured at 480×640; see the archive table below.

## Headline (stitch hero @ 1024×1024)

| Step | Optimization | `device_kernel` ms | `daemon_rt` ms | `total` ms | Δ total vs prev | PSNR vs Step 0 |
|---:|---|---:|---:|---:|---:|---|
| 0 | Baseline | **106.47** | 256.85 | **981.93** | — | — |
| 1 | Binary IPC + DRAM reuse | **102.53** | **195.14** | **880.98** | **−10.3%** | ∞ (identical) |

### Step 0 detail (reference)

| Metric | ms | Notes |
|---|---:|---|
| `sub.blend.prep` (host SoA pack) | **473.85** | dominates |
| `timings.sort` (CPU radix sort) | 160.86 | scales w/ 1.59 M pairs |
| `timings.tile_assign` (CPU bin) | 43.84 | |
| `sub.blend.daemon_rt.device_kernel` | **106.47** | 25 ms on-chip + ~80 ms dispatch/readback |
| `sub.blend.daemon_rt` (kernel + DRAM I/O + sync) | 256.85 | |
| `sub.blend.save_npy` (host→daemon IPC) | 21.82 | 4 .npy files |
| `sub.blend.load_npy` (daemon-side) | 2.58 | |
| **`timings.total`** | **981.93** | **end-to-end 1.02 fps** |

### Step 1 detail (current best)

| Metric | ms | Notes |
|---|---:|---|
| `sub.blend.prep` | 245.31 | variable; same CPU path |
| `timings.sort` | 161.41 | unchanged |
| `timings.tile_assign` | 37.69 | unchanged |
| `sub.blend.daemon_rt.device_kernel` | **102.53** | −3.7%; no per-frame alloc + selective zero-fill |
| `sub.blend.daemon_rt` | **195.14** | −24%; binary IPC, cached DRAM |
| `sub.blend.save_npy` | 80.27 | stdin payload write (replaces .npy) |
| `sub.blend.load_npy` | 8.27 | stdout image read |
| **`timings.total`** | **880.98** | **end-to-end 1.14 fps** |

`binning_log.py stitch hero --resolution 1024x1024` → **1,591,353** G-tile pairs at 32×32 tiles
(280,007 visible Gaussians of 341,426, 82%).

| Theoretical peak (130 cores @ 1.4 GHz) | Per-pair cycles | ms |
|---|---:|---:|
| Aggressive | 80 | **0.70** |
| Realistic | 200 | **1.75** |
| Realistic + 5× early-term | 200/5 | 0.35 |
| Current on-chip extrapolated | 1500 | 13.12 |

**Gap to realistic peak (1.75 ms):** `device_kernel`/peak = **60.8×**, `total`/peak = **561×**.

The kernel itself is now ~10× off realistic peak (after the 18% kernel-only wins); the **other 550×** comes from host prep, IPC, dispatch overhead, and CPU-side projection/sort that the plan addresses in steps 1–6.

## Archive — iter 001–013 (stitch hero @ 480×640)

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
| 013  | Stage F fusion via `mul_binary_tile` | **reverted** (no-win) | 56.78 | +0.4% | −18.0% |

**Net kernel improvement (480×640):** `68.92 ms → 56.54 ms` = **−18.0% on stitch hero**
**Acquires per Gaussian:** ~10 → **6**
**Kept:** 7 / 13 iters. Most recent 3 attempts (011, 012, 013) all reverted or yielded < 1% — clear convergence signal at this resolution.

## Multi-scene validation (iter-011 kernel, 480×640)

| Scene      | View | Kernel ms | Iter-000 ms | Δ        | PSNR  | SSIM  | image_diff |
|------------|------|----------:|------------:|---------:|------:|------:|:----------:|
| stitch     | hero | **56.54** | 68.92       | −18.0%   | 44.00 | 0.988 | clean-keep |
| luigi      | hero | **12.82** | 15.11       | −15.2%   | 37.16 | 0.980 | clean-keep |
| strawberry | hero | **207.44**| 250.33      | −17.1%   | 46.18 | 0.993 | clean-keep |

All scenes pass the permissive two-gate visual check at 480×640. No regressions.

## Lessons learned (informing future iters)

1. **FPU and SFPU dst layouts are incompatible.** `copy_dest_values<Float16_b>(idst_in, idst_out)` after an FPU `mul_tiles` produces garbage. Iter 009 hit this trap → PSNR 15 dB.
2. **SFPU dst-to-dst compute ops are expensive (`~630 cycles/entry` for `add_binary_tile`).** Only worth it when it eliminates a full acquire/release (iter 010), never as a per-channel inner loop op (iter 006).
3. **Acquire-block fusion alone (without eliminating intermediate CBs / `copy_tile`s) doesn't pay.** Iter 002 showed this; the wins in iters 003, 007, 008, 011 all came from genuinely eliminating L1 round-trips, not just merging blocks.
4. **FPU `mul_tiles` does not pipeline for free.** Iter 012 added 3 extra `mul_tiles` to save 1 acquire and net-regressed +0.6%.
5. **Front-to-back compositing has the identity `T·(1-α) = T - α·T`.** Reusing the already-computed `α·T` (CB_CONTRIB) as `T -= contrib` saved one full stage (iter 005).
6. **Sat-mask is redundant if T is hard-zeroed.** Once T saturates near 0, downstream multiplications keep it dead. Iter 004 eliminated CB_SAT_MASK entirely.

## Gap to theoretical peak (1024×1024 stitch hero, current)

`scripts/binning_log.py stitch hero --resolution 1024x1024`:

| Component | Current ms | Realistic peak (1.75 ms) gap | Notes |
|---|---:|---:|---|
| `device_kernel` (host-measured) | 106.47 | 60.8× | 25 on-chip + ~80 dispatch/readback |
| `daemon_rt` | 256.85 | 147× | DRAM upload x5 + sync + program enqueue |
| `prep` (host SoA) | 473.85 | 271× | 9 fp32 cols × 1.59 M entries |
| `total` | 981.93 | 561× | end-to-end |

## Optimization plan (Step 0 done, six steps remaining)

See [docs/optimization-log/plan-stitch-hero-1024.md](./plan-stitch-hero-1024.md) for the full plan. Step summary:

| Step | Goal | Estimated saving | Status |
|---|---|---|---|
| **0** | Measure & expose at 1024×1024 | — | **DONE** |
| 1 | Binary IPC + reused DRAM buffers | -30-40 ms (kills .npy + per-frame alloc) | pending |
| 2 | Static-attribute persistence + on-device gather | -200+ ms (kills most of `prep`) | pending |
| 3 | Persistent kernel + mailbox dispatch | -25-80 ms (kills per-frame `EnqueueProgram`) | pending |
| 4 | DST-resident R/G/B/T accumulators | -3-7 ms on-chip | pending |
| 5 | 16×16 tiles | -3-10 ms (conditional) | pending |
| 6 | Move `project_gaussians` to device | -5-15 ms (stretch) | pending |

Cumulative target: **~5-10 ms total per frame** at 1024×1024 (a >100× speedup vs the 981 ms baseline above).
