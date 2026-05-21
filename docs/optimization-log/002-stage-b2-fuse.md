# Iter 002 — Stage B2 fusion

- **Idea**: Fuse Stage B2a/B2b/B2c (dx², dy², dx·dy) into a single `tile_regs_acquire()` block instead of three separate acquire/release pairs.
- **Hypothesis**: Save ~150–250 cycles per Gaussian-tile pair (~5–10% kernel speedup) by eliminating two acquire/release overheads; Dst has 4 fp32 slots — three simultaneous products fit.
- **Branch**: `opt/002-stage-b2-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **REVERTED — no perf win** (kernel +0.6% vs same-session baseline, bit-identical output). Per gating policy, no measurable speedup → revert even though PSNR = ∞.

## Code diff

Collapsed three separate acquire blocks (B2a dx², B2b dy², B2c dx·dy) into one block with `mul_tiles_init` mid-block reconfiguration and three `pack_tile` calls to `CB_DX2`, `CB_DY2`, `CB_DXDY`. Stage B3a unchanged (still copies from intermediate CBs). Only file touched: `alpha_blend_compute.cpp` (~40 lines → ~25 lines in Stage B2).

Step B (fuse B2 + B3a, eliminate intermediate CBs) was **not attempted** — Step A failed the perf gate.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | 68.90* | 69.30 | +0.6% | ∞ | ∞ | 1.0000 | 1.0000 |

\*Same-session baseline re-measured on bh-30 immediately after revert (iter-000 locked baseline was 68.92 ms). Fused kernel measured first in session; baseline check second — both with `--warmup 3 --frames 10`.

Daemon RT median (fused): 122.52 ms. End-to-end fps (fused): 3.26.

## Screenshots

- Render: `/tmp/iter002_stitch_hero.png` on bh-30 (bit-identical to baseline)
- `docs/optimization-log/002-amplified-diff.png` (empty — PSNR ∞)

## Notes

- **`mul_tiles_init` mid-block works correctly.** Output is bit-identical to baseline (PSNR ∞, SSIM 1.0). No compile or JIT errors.
- **No measurable speedup despite fewer acquire blocks.** Fused run median 69.30 ms vs same-session baseline 68.90 ms (+0.40 ms, +0.6%). All 10 timed frames reported exactly 69.3 ms (fused) vs 68.9 ms (baseline) — the delta is real, not median noise from outlier frames.
- **Possible explanation:** acquire/release overhead for B2 may be a smaller fraction of total kernel time than estimated (~150–250 cycles × ~832K Gaussian-tile pairs). The dominant cost is likely elsewhere (B3b2+C long block, Stage D six acquire blocks, CB wait/pack latency). Saving 2 acquire pairs per Gaussian may be hidden in NoC/CB synchronization or doesn't reduce wall-clock because the FPU was already pipelined across separate blocks.
- **Step B not attempted.** Would have fused covariance scalar multiplies into the same block and eliminated CB_DX2/DY2/DXDY — only worth trying if Step A showed a win.

## Lessons for the loop

1. **Acquire-block count isn't always the bottleneck.** Similar to iter 001 (reduce overhead with no skip benefit), micro-fusion of SFPU blocks can be correctness-clean but perf-neutral.
2. **Same-session A/B matters.** Comparing fused (69.30) against iter-000 locked number (68.92) showed +0.6%; same-session baseline (68.90) confirms the sign. Always re-benchmark baseline in the same daemon session when the delta is sub-1%.
3. **Next candidates:** Stage D `addcmul_tile` fusion, Dst-resident R/G/B/T accumulators, or host-side tile culling — targets with larger estimated cycle savings than B2 acquire overhead.
