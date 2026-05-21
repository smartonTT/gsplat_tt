# Iter 003 — B2+B3a collapse

- **Idea**: Collapse Stage B2 (dx², dy², dx·dy) and Stage B3a (covariance weighting) into one `tile_regs_acquire()` block; pack weighted terms directly to `CB_Q` without staging through `CB_DX2` / `CB_DY2` / `CB_DXDY`.
- **Hypothesis**: Eliminate 3 CB round-trips, 3 `copy_tile` SFPU loads, and 3 extra acquire blocks per Gaussian-tile pair (~750 cycles/pair → ~4–10% kernel speedup).
- **Branch**: `opt/003-b2b3a-collapse`
- **Worker model**: composer-2.5-fast
- **Decision**: **clean-keep**

## Code diff

Replaced four acquire blocks (B2a/B2b/B2c + B3a) with one fused block: `mul_tiles` for dx²/dy²/dx·dy into Dst[0..2], immediate `mul_unary_tile` by `cov_a` / `cov_c` / `two_cov_b`, pack three tiles to `CB_Q`. Removed `cb_pop_front` for `CB_DX2` / `CB_DY2` / `CB_DXDY` in the inner-loop cleanup (host CBs unchanged, unused). Only file touched: `alpha_blend_compute.cpp`.

Note: `mul_unary_tile_init()` from the supervisor recipe does not exist in this tt-metal API; compile failed until that call was removed (baseline already uses `mul_unary_tile` without it).

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs prev | PSNR vs baseline | SSIM vs prev | SSIM vs baseline |
|---|---|---|---|---|---|---|---|---|
| stitch | hero | 68.91* | 65.79 | **−4.5%** | 51.45 dB | 51.45 dB | — | 0.9939 |

\*Same-session baseline re-measured on bh-30 immediately before fused run (`git show dbb856a` kernel synced via rsync). Locked iter-000 baseline: 68.92 ms.

Daemon RT median (fused): 118.06 ms (baseline same-session 120.73 ms). End-to-end fps (fused): 3.30 (baseline 3.18).

## Screenshots

- Render: `/tmp/iter003_stitch_hero.png` on bh-30
- `docs/optimization-log/003-amplified-diff.png`

## Notes

- **First measurable kernel win in the loop.** Fused median 65.79 ms vs same-session baseline 68.91 ms (−3.12 ms). vs locked baseline 68.92 ms (−3.13 ms, −4.5%). All 10 timed frames reported exactly 65.8 ms (fused) vs 68.8–69.0 ms (baseline) — stable, not median noise.
- **Visual gate clean-keep** (exit 0): PSNR 51.45 dB, SSIM 0.9939 vs `benchmarks/reference/stitch_hero.png`. Small max-abs-diff (R16/G19/B29) from fusing mul+scale in Dst (different bf16 rounding order than copy-then-mul path); well above hard-reject floors.
- **Iter 002 lesson confirmed in reverse:** acquire-block count alone (B2 fuse only) did not help (+0.6%); eliminating **CB spill + copy_tile** between B2 and B3a does. The ~750 cycles/pair estimate landed at ~4.5% wall-clock, not ~10% — remainder is still dominated by B3b2+C, Stage D six acquire blocks, and per-Gaussian CB traffic elsewhere.
- **Operational footgun:** `pkill -9 -f metal_example_gaussian_splatting` over SSH can match the remote shell’s argv and kill the session (exit 255). Use `killall -9 metal_example_gaussian_splatting` instead.

## Next

Merge into `smarton/optimization`. Candidates: Stage D `addcmul_tile` fusion, Dst-resident accumulators, or host tile culling — still targeting CB round-trips and acquire count where operands allow single-block fusion.
