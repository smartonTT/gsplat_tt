# T7 D-SSIM on device (#706) — results so far

All timings: bh-33 **p100 (provisional)**, 1237x822, b2b over 50 launches. The p150 box (#687) was
not in place when these ran.

## Step 2: split3 banded-matmul Gaussian filter (`LOSS_MODE=filt`)

Kernel: `render/bench/kernels/loss/filt_compute.cpp` (+ `reader_filt.cpp`, `writer_filt.cpp`),
driven by `render/bench/loss_bench.cpp run_filt`. One output tile = 9 row-pass + 9 column-pass
tile matmuls (split3: zh·Rh + zl·Rh + zh·Rl per band tile), fp32 dest, operands split on the SFPU.
Check: `opt/t690/loss_check.py check <dir> --mode filt` vs the fp64 filter (pass: l2rel < 2e-6,
maxabs < 1e-5) and vs the split3 model (`gsplat/train_loss.py filt_tiles(.., "split3", "trunc")`).

| run | commit | change | dump64 l2rel | syn l2rel | b2b ms |
|---|---|---|---|---|---|
| 1971 #1 | 98fefed2 | first version (whole strips per core, 4-tile ring CB, 10-bit hi) | 5.8e-4 FAIL | 5.9e-2 FAIL | 0.2523 |

Causes found in run #1:
- Period-4 tile-row errors (0.1–0.2) on syn: `matmul_tiles` indexes a CB from its read pointer
  and does not wrap, so a 3-tile sliding window in a 4-page ring CB read past the end. Fixed by
  three rotating one-tile CB pairs.
- ~5e-4 error everywhere (TF32 level): the split hi part kept all 10 TF32 mantissa bits. Working
  hypothesis: the HiFi4 multiply covers only part of srcA's mantissa, so the hi part must keep
  fewer bits. Next run sweeps the hi mask (`LOSS_HIBITS` 8/9/10).
- Load imbalance: 117 strips on 110 cores (7 cores did 2 strips). Fixed with balanced contiguous
  tile runs plus recomputed halo rows.

Pending: run t706filt2 (eb319dd5), queued behind #687 on the p100 lock.
