---
name: image-diff
description: Compare a rendered PNG to a reference PNG using PSNR, SSIM, RMSE, max-abs-diff, and an amplified-diff PNG. Implements the permissive two-anchor gate from the alpha-blend optimization plan (previous-kept vs original-baseline). Exit codes 0/1/2 drive keep / hard-reject / flag-for-review. Use after every kernel optimization iteration to gate the change.
---

# image-diff

Per-iteration visual regression gate. The philosophy: **keep changes unless clearly broken**. Numerical PSNR is too noisy for floating-point math reorderings; SSIM is the perceptual safety net. Anything kept-but-suspicious gets a `NEEDS_REVIEW` flag in the iter log; the user audits later.

## Two-anchor gate

Every kept iter is compared against two reference PNGs:

- **Previous kept** (`<...>/screenshots/<N-1>_stitch_hero_after.png`): catches one-step regressions.
- **Original baseline** (`benchmarks/reference/stitch_hero.png`): catches cumulative drift across many small kept changes.

Both diffs are computed every iteration. Logging is unconditional; only hard-reject criteria block a keep.

## Quick start

```bash
python scripts/image_diff.py <reference.png> <candidate.png> --amplified-diff <candidate>_diff.png
```

Prints all metrics + exit code:

| Exit | Meaning |
|---|---|
| `0` | clean keep |
| `2` | kept but `NEEDS_REVIEW` (perceptual drop) |
| `1` | hard reject (broken output) |

## Hard reject (revert the iteration)

Any one of these triggers exit `1`:

- Output contains NaN or Inf
- SSIM (vs previous kept) < 0.75
- PSNR (vs previous kept) < 20 dB
- Mean abs diff > 25 LSB (=`mean(|a-b|)*255` over the whole image)

## Flagged for review (kept, exit `2`)

Any one of these triggers exit `2`:

- SSIM (vs previous kept) drops by > 0.05 in one step
- PSNR (vs previous kept) < 35 dB
- SSIM (vs original baseline) < 0.92
- PSNR (vs original baseline) < 32 dB

## Clean keep (exit `0`)

Neither hard-rejected nor flagged.

## What the script outputs

Stdout layout:

```
=== <candidate>.png vs <reference>.png ===
psnr_db            = 41.20
ssim               = 0.9876
rmse               = 0.00237
max_abs_diff (per chan, 0-255) = R 6  G 5  B 7
frac_pixels > 4LSB = 0.0123
has_nan_or_inf     = False
=== gate ===
result             = clean-keep | needs-review | hard-reject
```

If `--amplified-diff` is given, the script writes a 10x-amplified diff PNG with gamma 2.2 so the user can see where pixels differ at audit time.

## When to call both anchors

`render_fixed.py` produces one candidate PNG. The supervisor / worker calls `image_diff.py` twice:

```bash
# Against previous kept (per-step gate)
python scripts/image_diff.py <prev_kept>.png /tmp/it.png --amplified-diff /tmp/it_diff_step.png
step_exit=$?

# Against original baseline (drift gate)
python scripts/image_diff.py benchmarks/reference/stitch_hero.png /tmp/it.png --amplified-diff /tmp/it_diff_drift.png
drift_exit=$?
```

Combine: if either gives `1` → hard reject. Else if either gives `2` → flag `NEEDS_REVIEW`. Else clean keep.

## Determinism

`scripts/render_fixed.py` sets seeds and reuses fixed cameras from `benchmarks/cameras.json`, so the only sources of run-to-run variance are:

- Kernel floating-point (deterministic within a build; tt-metal does not race on lanes)
- bf16 storage rounding (deterministic within a build)
- A reorder of math inside the kernel (a different build → different bf16-rounding noise)

When the kernel and build are unchanged, two consecutive renders should produce **identical** PNGs (PSNR = ∞). If they don't, something nondeterministic crept in and must be tracked down before the loop continues.

## Audit workflow (user does this)

When the user wants to verify the loop didn't accumulate visual bugs:

1. Open `docs/optimization-log/SUMMARY.md`; scan for `NEEDS_REVIEW` rows.
2. For any suspicious row, open `docs/optimization-log/<NNN>-<name>.md`; review the per-iter screenshots and the amplified diff.
3. To revert: `git revert <commit>` for that iter, then re-run the loop from there.

## When NOT to use this skill

- For pure perf benchmarking (no visual comparison): use [tt-profile](../tt-profile/SKILL.md) directly.
- For one-off "does the kernel produce something reasonable": just run the viewer, no need for the script.
