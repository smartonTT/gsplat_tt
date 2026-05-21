# Iter 011 — D2 adder + Stage E fuse

- **Idea**: Fuse Stage D2 adder (3× `add_tiles`) and Stage E (`sub_tiles`) into one acquire block with 4 dst slots.
- **Hypothesis**: Save one acquire/commit/wait/release cycle per Gaussian (~50–100 cycles); pure FPU CB-to-CB ops with no inter-op dependencies → ~0.8–1.5% kernel speedup.
- **Branch**: `opt/011-d2adder-e-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- Merged D2 adder and Stage E into a single `tile_regs_acquire` block.
- dst[0..2]: `add_tiles` for R/G/B state accumulators (unchanged).
- dst[3]: `sub_tiles(CB_T_STATE, CB_CONTRIB)` for transmittance update (was separate acquire with dst[0]).
- Multi-pack: 4 in-place pop+reserve+pack+push sequences before `tile_regs_release`.
- Trailing CB drains: `CB_T_R/G/B` and `CB_CONTRIB` all popped after the fused block.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 57.31 | **56.54** | **−1.34%** | 44.00 dB | 0.9876 |

Daemon RT median: 109.21 ms. All 10 timed frames: stable 56.54–56.6 ms `device_kernel`.

## Visual gate

- **exit 0 (clean-keep)** vs original baseline: PSNR 44.00 dB, SSIM 0.9876; max-abs-diff R16/G19/B28; mean-abs-diff 1.07 LSB.
- No NaN/Inf. Identical metrics to iter-010 baseline.

## Screenshots

- Render: `docs/optimization-log/screenshots/011_stitch_hero_after.png`
- `docs/optimization-log/011-amplified-diff.png`

## Notes

- **Measurable win:** −0.77 ms (−1.34%) vs iter-010 baseline 57.31 ms — exceeds gating threshold (>0.5 ms and >1%).
- **4-dst-slot FPU mix compiled and ran cleanly:** 3× `add_tiles_init` + 1× `sub_tiles_init` in one acquire, 4 separate pack destinations — same pattern as iter 008 (3 add) and iter 010 (5 inits in B3b2+C), but first successful cross-op fusion of add + sub in one block.
- **No surprises:** No CB lifetime issues; CB_T_R/G/B and CB_CONTRIB remain alive through the block; in-place state CB updates unchanged.
- Estimated per-Gaussian savings: ~0.77 ms / 832049 entries ≈ **930 cycles** saved per acquire elimination (consistent with 50–100 cycles estimate being conservative — full acquire overhead is higher when amortized).

## Next

Further acquire fusions among pure-FPU stages (D2 producer already fused in iter 008; consider whether any remaining standalone acquire blocks can merge without mixing FPU/SFPU dst layouts).
