# Iter 013 — Stage F fuse (mul_binary_tile)

- **Idea**: Fuse Stage F sat-mask refresh into one acquire using FPU dual `copy_tile` + SFPU `unary_ge_tile` + `mul_binary_tile`, eliminating CB_T_TMP traffic and the second acquire/mul_tiles block.
- **Hypothesis**: Save one acquire + CB push/pop + FPU `mul_tiles` + pack every 16 Gaussians; pay one extra `copy_tile` + one `mul_binary_tile` in the same acquire. Net positive if eliminated overhead exceeds SFPU cost at 1/16 rate.
- **Branch**: `opt/013-stagef-fuse`
- **Worker model**: composer-2.5-fast
- **Decision**: **reverted-no-win**

## Code diff (experiment only — reverted)

Only `alpha_blend_compute.cpp` Stage F block:

- Replaced 2-acquire pattern (mask to CB_T_TMP → `mul_tiles`) with 1-acquire fused path:
  - `copy_tile` T to dst[0] and dst[1]
  - `unary_ge_tile(1, T_THRESH_BITS)` for mask in dst[1]
  - `mul_binary_tile(0, 1, 0)` for T × mask in dst[0]
  - Single in-place CB_T_STATE update; no CB_T_TMP traffic
- **Reverted** to iter-011 Stage F after bench (no measurable win).

## Bench

| Scene | view | prev kernel ms | this kernel ms | Δ% | PSNR vs baseline | SSIM vs baseline |
|---|---|---|---|---|---|---|
| stitch | hero | 56.54 | **56.78** | **+0.42%** | 44.00 dB | 0.9876 |

Daemon RT median: 118.93 ms. All 10 timed frames: stable 56.78 ms `device_kernel`.

## Visual gate

- **exit 0 (clean-keep)** vs original baseline: PSNR 44.00 dB, SSIM 0.9876; max-abs-diff R16/G19/B28; mean-abs-diff 1.07 LSB.
- **exit 0 (clean-keep)** vs iter-011 previous kept: PSNR ∞, SSIM 1.0000 (pixel-identical).
- No NaN/Inf. Correctness confirmed; perf gate failed (no win).

## Screenshots

- Render: `docs/optimization-log/screenshots/013_stitch_hero_after.png`
- `docs/optimization-log/013-amplified-diff.png`

## Notes

- **No measurable win:** +0.24 ms (+0.42%) vs iter-011 baseline 56.54 ms — below gating threshold and in the wrong direction. Reverted per no-win policy.
- **Correctness OK:** FPU dual-copy → SFPU `unary_ge` → SFPU `mul_binary_tile` in one acquire compiled and ran; output identical to iter-011.
- Stage F runs every 16 Gaussians (~52k invocations/frame on stitch hero); both eliminated work and added SFPU op scale at the same 1/16 rate.

### mul_binary_tile calibration (Stage F fused context)

Frame-level (832,049 sorted entries, stitch hero):

| Metric | Value |
|---|---|
| Net kernel Δ | **+0.24 ms** (flat/regression) |
| Stage F invocations/frame | ~52,003 (every 16 g, g>0) |
| Per-invocation cycle accounting (iter-010 scale) | saves ~260 cyc, adds ~660 cyc (incl. ~630 cyc `mul_binary_tile`) |
| Per-Gaussian amortized net (÷16) | **≈ +25 cyc/Gaussian** → ~0.017 ms/frame predicted |
| Observed | +0.24 ms — consistent with flat/no-win; delta likely noise + slight mul_binary_tile dominance |

**Interpretation:** `mul_binary_tile` appears to cost similarly to `add_binary_tile` (~630 cycles/op from iter-010/006 calibration). At Stage F's 1/16 rate, one `mul_binary_tile` roughly cancels the savings from eliminating the second acquire, CB_T_TMP round-trip, `mul_tiles`, and extra pack. Unlike iter-010 (per-Gaussian B3b1 elimination saving ~1.1 ms gross), Stage F's eliminated work is also amortized 1/16 — so SFPU binary mul is **not worth it** here even though the FPU→SFPU pattern is valid.

**Do not retry** Stage F SFPU fuse unless the refresh rate drops further or a cheaper mask-multiply path appears (e.g. pure FPU with pre-packed mask CB).

## Next

Look for acquire fusions where eliminated work is **per-Gaussian** (not 1/16), or pure-FPU merges without adding SFPU ops.
