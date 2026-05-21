# Iter 008 — Stage D2 adder batch

- **Idea**: Collapse Stage D2's three per-channel FPU adder acquire blocks into one batched acquire using dst slots 0/1/2 (mirrors iter 007 producer batching).
- **Hypothesis**: Save 2 tile_regs acquire/commit/wait/release cycles per Gaussian → ~1.5–3% kernel speedup; same FPU op count, fewer acquire overhead.
- **Branch**: `opt/008-d2-adder-batch`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- Stage D2: one batched adder block — three `add_tiles_init` + `add_tiles` (dst 0/1/2) in a single acquire, then in-place spill pack to CB_COLOR_R/G/B_STATE with `pack_tile(0/1/2, ...)`.
- Stage D2 now **2 acquires/Gaussian** (1 batched producer + 1 batched adder), down from 6 before iter 007.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 59.10 | **57.97** | **−1.9%** | 44.10 dB | 0.9885 |

Daemon RT median: 110.57 ms. All 10 timed frames: stable 58.0 ms `device_kernel`.

## Visual gate

- **exit 0 (clean-keep)**: PSNR 44.10 dB, SSIM 0.9885 vs original baseline; max-abs-diff R17/G19/B29; mean-abs-diff 1.05 LSB.
- No NaN/Inf.
- Identical metrics to iter 007 vs baseline (no cumulative drift).

## Screenshots

- Render: `docs/optimization-log/screenshots/008_stitch_hero_after.png`
- `docs/optimization-log/008-amplified-diff.png`

## Notes

- **Measurable win:** −1.13 ms (−1.9%) vs iter-007 baseline 59.10 ms — exceeds gating threshold (>0.5 ms and >1%).
- **Pattern confirmation:** Multiple `add_tiles_init` calls within one acquire (same idiom as Stage B3b2+C) — no JIT compile warnings or visible compile noise (compute kernel JIT at daemon startup only).
- **Stage D2 acquire count:** 6 → 4 (iter 007 producer) → **2** (this iter adder batch).
- **Surprise:** Win slightly below the 1.5–3% estimate but solidly above the no-win revert floor; symmetric to iter 007's producer batch (−3.2%) at roughly half the remaining acquire savings.

## Next

Profile whether Stage D2 is still dominant or shift to dst-resident accumulators / other stages per tt-profile.
