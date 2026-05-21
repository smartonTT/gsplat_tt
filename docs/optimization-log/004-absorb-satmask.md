# Iter 004 — absorb sat_mask into T_state

- **Idea**: Eliminate `CB_SAT_MASK` from the compute kernel by hard-zeroing `T_state` in Stage F (where `T < 1e-4`) and dropping the per-Gaussian `sat_mask` multiplies in Stage D1 and Stage E.
- **Hypothesis**: Save 2 `mul_tiles` + 2 acquire blocks + 2 `CB_T_TMP` round-trips per Gaussian-tile pair; pay ~1 extra `mul_tiles` every 16 Gaussians in Stage F → net ~3–5% kernel speedup.
- **Branch**: `opt/004-absorb-satmask`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- **Stage F**: `unary_ge_tile` mask into `CB_T_TMP`, then `mul_tiles(T_state, mask)` to hard-zero saturated pixels (CB-intermediate path; `mul_binary_tile` not in tt-metal API).
- **Stage D1**: single acquire — `contrib = alpha · T_state` (was two-step via `sat_mask`).
- **Stage E**: two acquires — `T_state = T_state · (1 - alpha)` (was three-step with `sat_mask`).
- Removed per-tile `CB_SAT_MASK` init/wait/drain; host still allocates CB 21 (unchanged).

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | 65.79 | **63.12** | **−4.1%** | 51.45 dB | 51.45 dB | — | 0.9939 |

Daemon RT median: 115.73 ms (iter-003 ~118 ms). All 10 timed frames reported exactly 63.1 ms `device_kernel` (stable).

## Screenshots

- Render: `/tmp/iter004_stitch_hero.png` on bh-30
- `docs/optimization-log/004-amplified-diff.png`

## Notes

- **Measurable win:** 63.12 ms vs iter-003 baseline 65.79 ms (−2.67 ms, −4.1%). Exceeds gating threshold (> 0.5 ms and > 1%).
- **Visual gate clean-keep** (exit 0): PSNR 51.45 dB, SSIM 0.9939 vs `benchmarks/reference/stitch_hero.png` — bit-identical metrics to iter 003; max-abs-diff unchanged (R16/G19/B29).
- **Stage F refresh cost did not dominate:** extra `mul_tiles` every 16 Gaussians (~7.5 cycles/Gaussian amortized) was far outweighed by removing 2 muls + 2 acquires per Gaussian. Savings landed near the low end of the 3–5% estimate (4.1%), consistent with iter 003’s lesson that CB spill elimination beats acquire-count tweaks alone.
- **`mul_binary_tile`:** not present in this tt-metal tree (grep over `backends/tt/tt-metal` = 0 hits). Stage F uses two acquire blocks + `CB_T_TMP` intermediate. A dst-to-dst binary mul API would fuse Stage F to one acquire for iter 005+.
- **Outlier frames:** frames 6–10 showed inflated `blend.prep` (1.2–1.8 s) on some runs; `device_kernel` stayed 63.1 ms throughout — prep noise unrelated to this kernel change.

## Next

Merge into `smarton/optimization`. Host cleanup: drop `CB_SAT_MASK` allocation once stable. Candidates: Stage D `addcmul_tile` fusion, Dst-resident accumulators, or explore dst-resident Stage F if `mul_binary_tile`-equivalent appears in LLK docs.
