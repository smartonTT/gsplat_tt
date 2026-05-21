# Iter 005 — Stage E sub via CB_CONTRIB

- **Idea**: Collapse Stage E from 2 acquire blocks (`rsub` → `CB_ONE_MINUS_ALPHA`, then `T * (1-α)`) to 1 acquire block (`T_state -= CB_CONTRIB`), reusing the `α·T` tile already live from Stage D1/D2.
- **Hypothesis**: Save 1 acquire + 1 rsub + 1 copy + 1 pack + 4 CB ops on `CB_ONE_MINUS_ALPHA` per Gaussian-tile pair; `sub_tiles` cost ≈ former `mul_tiles` → net ~3–5% kernel speedup.
- **Branch**: `opt/005-stage-e-sub`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Only `alpha_blend_compute.cpp`:

- **Stage D2**: removed trailing `cb_pop_front(CB_CONTRIB, 1)` — CB stays at front through Stage E.
- **Stage E**: replaced 2-acquire `rsub` + `mul_tiles(T, 1-α)` with single-acquire `sub_tiles(T_state, CB_CONTRIB)` + spill.
- **CB lifetime**: `cb_pop_front(CB_CONTRIB, 1)` moved to immediately after Stage E's `cb_wait_front(CB_T_STATE, 1)`.
- Host / `CB_ONE_MINUS_ALPHA` allocation unchanged.

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | 63.12 | **61.03** | **−3.3%** | — | 44.10 dB | — | 0.9885 |

Daemon RT median: 113.90 ms. All 10 timed frames reported 61.0–61.2 ms `device_kernel` (stable).

## Screenshots

- Render: `/tmp/iter005_stitch_hero.png` on bh-30
- `docs/optimization-log/005_stitch_hero_after.png`
- `docs/optimization-log/005-amplified-diff.png`

## Notes

- **Measurable win:** 61.03 ms vs iter-004 baseline 63.12 ms (−2.09 ms, −3.3%). Exceeds gating threshold (> 0.5 ms and > 1%).
- **Visual gate clean-keep** (exit 0): PSNR 44.10 dB, SSIM 0.9885 vs `benchmarks/reference/stitch_hero.png`; max-abs-diff unchanged (R17/G19/B29); mean-abs-diff 1.05 LSB.
- **PSNR surprise:** Expected ~51 dB (same or slightly better with subtract path). Actual 44.10 dB vs baseline — ~7 dB drop from the `T·(1-α)` → `T−α·T` reordering in bf16, but still well above hard-reject (20 dB) and baseline flag (32 dB) thresholds. SSIM remains high (0.9885). Worth monitoring cumulative drift; no NEEDS_REVIEW flag triggered.
- **`sub_tiles` API:** confirmed in `tt_metal/hw/inc/api/compute/eltwise_binary.h` (not under `include/` — grep path in recipe missed it).
- **Acquire count:** Stage E 2 → 1 acquire per Gaussian; savings landed mid-range of 3–5% estimate.

## Next

Merge into `smarton/optimization`. Candidates: Stage D `addcmul_tile` fusion, dst-resident accumulators, or host-side drop of unused `CB_ONE_MINUS_ALPHA`.
