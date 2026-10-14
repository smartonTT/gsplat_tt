# T7 D-SSIM on device (#706) — results so far

Runs 1971 #1/#2: bh-33 **p100 (provisional)**, 1237x822, b2b over 50 launches (the p150 box from
#687 was not in place yet). Later runs: bh-49 board 0, p150a (#687's measurement box).

## Step 2: split3 banded-matmul Gaussian filter (`LOSS_MODE=filt`)

Kernel: `render/bench/kernels/loss/filt_compute.cpp` (+ `reader_filt.cpp`, `writer_filt.cpp`),
driven by `render/bench/loss_bench.cpp run_filt`. One output tile = 9 row-pass + 9 column-pass
tile matmuls (split3: zh·Rh + zl·Rh + zh·Rl per band tile), fp32 dest, operands split on the SFPU.
Check: `opt/t690/loss_check.py check <dir> --mode filt` vs the fp64 filter (pass: l2rel < 2e-6,
maxabs < 1e-5) and vs the split3 model (`gsplat/train_loss.py filt_tiles(.., "split3", "trunc")`).

| run | commit | change | dump64 l2rel | syn l2rel | b2b ms |
|---|---|---|---|---|---|
| 1971 #1 | 98fefed2 | first version (whole strips per core, 4-tile ring CB, 10-bit hi) | 5.8e-4 FAIL | 5.9e-2 FAIL | 0.2523 |
| 1971 #2 | eb319dd5 | one-tile row slots, balanced runs, hi 9 bits | 2.9e-4 FAIL | 2.8e-4 FAIL | 0.2124 |
| 1971 #2 | eb319dd5 | same, hi 8 bits | – | 2.8e-4 FAIL | 0.2124 |
| 1971 #2 | eb319dd5 | same, hi 10 bits | – | 5.4e-4 FAIL | 0.2125 |

Causes found in run #1:
- Period-4 tile-row errors (0.1–0.2) on syn: `matmul_tiles` indexes a CB from its read pointer
  and does not wrap, so a 3-tile sliding window in a 4-page ring CB read past the end. Fixed by
  three rotating one-tile CB pairs.
- ~5e-4 error everywhere (TF32 level): the split hi part kept all 10 TF32 mantissa bits. Working
  hypothesis: the HiFi4 multiply covers only part of srcA's mantissa, so the hi part must keep
  fewer bits. Next run sweeps the hi mask (`LOSS_HIBITS` 8/9/10).
- Load imbalance: 117 strips on 110 cores (7 cores did 2 strips). Fixed with balanced contiguous
  tile runs plus recomputed halo rows.

Run #2 findings:
- The period-4 tile-row errors are gone (syn 5.9e-2 -> 2.8e-4) and balancing cut b2b 0.252 -> 0.212 ms.
- A ~2.8e-4 error floor remains, the same for hi 8 and 9 bits, so it does not come from operand
  splitting. It is a systematic **positive** bias: device/fp64 - 1 = +2.6e-4 mean (about 2^-12),
  std 1.5e-4, flat across in-tile rows and columns. With hi 10 bits the bias flips to -5.1e-4
  (srcA's unconsumed LSB, as tt-isa-documentation SrcASrcB.md says for TF32: phases use the
  implicit bit + 4 + 5 explicit bits of srcA and all 10 of srcB, so 9 hi bits is the right mask).
- Numpy emulations of the FPU (products exact, partial sums rounded to m bits by trunc / rne /
  ceil / half-up, per 16-wide MVMUL chunk and per Dst add) do not reproduce the device output
  (best: ceil at 15 bits, still 1.7e-4 off). The cause is not identified from documentation.
- Next: a direct diagnostic (`LOSS_RDIAG=c`: R[0] = c I, R[+-1] = 0, so the result must be c^2 x
  within 2^-18). c = 1 and 0.5 test the split/pack/accumulate path with trivial products,
  c = 0.2734375 (5-bit mantissa) and 0.26596152 (the Gaussian's centre tap, full fp32) test the
  multiplier. Run t706diag (a02ca59c) on bh-49 p150.
