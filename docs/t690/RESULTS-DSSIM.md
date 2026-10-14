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

Run t706diag (a02ca59c, bh-49 p150a, the #687 measurement box; log in run 1983 `t706diag.log`, rc 0):

| case | hi bits | R | error vs expected | verdict |
|---|---|---|---|---|
| dump64 | 9 | 1·I | 5.7e-7 (c²x) | PASS |
| dump64 | 9 | 0.5·I | 5.7e-7 | PASS |
| dump64 | 9 | 0.2734375·I (5-bit mantissa) | 7.2e-7 | PASS |
| dump64 | 9 | 0.26596152·I (full fp32 centre tap) | 5.2e-6 | FAIL |
| dump64 | 8 | 0.26596152·I | 1.1e-5 | FAIL |
| dump64 | 10 | 1·I / 0.26596152·I | 5.0e-4 / 1.3e-3 | FAIL |
| dump64 | 9 | Gaussian band (11 taps) | 2.9e-4 | FAIL |
| syn | 9 | 1·I | 5.1e-7 | PASS |
| syn | 9 | Gaussian band | 2.8e-4 | FAIL |

Timing (p150, bh-49): split3 Gaussian filter at 1237x822, hi 9 bits: **0.209 ms/view b2b**
(launch median 0.224 ms), 117 strips on 110 cores. p100 bh-33 run #2 was 0.212 (provisional).

What this shows:
- The split / pack / unpack / Dst path is exact to fp32 level (5e-7) whenever each output is a
  single product with a short-mantissa weight. Hi 9 bits confirmed again (10 bits fails even at R = I).
- A full-precision weight in a single product loses ~5e-6: the R operand also needs a hi/lo split
  (or the weights rounded to 9+9 bits), but that is 50x smaller than the floor.
- The 2.8e-4 floor appears only when an output sums several non-trivial products (the 11-tap band).
  So the bias comes from the FPU's inner-product accumulation inside one matmul (partial sums
  narrower than fp32), not from operand splitting or Dst accumulation.

Next (not done in this task):
1. Confirm with R = I + shift(I) (two exact taps): if that fails, accumulation is the cause.
2. Fixes to try, cheapest first: (a) quantise the Gaussian weights to 9-bit hi + 9-bit lo and
   split R too; (b) keep one non-zero tap per K-column group per matmul call so every partial sum
   is a single product and the sum happens in fp32 Dst (more matmul calls; time it);
   (c) judge the 2.8e-4 filter error by its effect on the D-SSIM loss and dL/dimage against the
   T0 oracle, not on the filtered maps alone.
3. Then the fused L1 + D-SSIM fwd + bwd kernel (not started).

## Box diagnostic t706box (06b690e5, bh-49 p150, 2026-10-14 01:15 PDT)

LOSS_RDIAG=-c: R is an 11-tap box with weight c (short mantissa), so every product is exact and
only the in-matmul accumulation can err. Hi 9 bits.

| case | c | l2rel vs exact | mean rel bias | verdict |
|---|---|---|---|---|
| dump64 | 1/16 | 1.09e-5 | +7.6e-6 | FAIL (tol 1e-6) |
| dump64 | 3/32 | 1.10e-5 | +1.0e-5 | FAIL |
| dump64 | R = 1·I (control) | 5.7e-7 | -4.1e-7 | PASS |
| syn | 1/16 | 6.1e-7 | -3.8e-7 | PASS |

Filter timing unchanged: 0.209 ms/view b2b at 1237x822 (launch median 0.224 ms).

What this shows:
- Multi-term accumulation inside one matmul does cost precision (~1e-5, small positive bias on
  real image data; exact on the smooth synthetic image), but that is ~25x below the 2.8e-4 floor
  of the Gaussian band. So accumulation is not the main cause.
- The rest must come from the full-precision Gaussian weights in R (products not exact). Next fix
  is (a): split R into 9-bit hi + 9-bit lo (or quantise the 11 weights to 18 bits) and do the
  extra R_lo pass; expected to bring the filter to ~1e-5. If ~1e-5 is still too much for the
  T0 oracle on loss and dL/dimage, use (b) from the list above for the remaining accumulation part.

## t714: tap-structure and fixed-point diagnostics, error propagation to the T0 oracle (bh-49 p150a, 2026-10-14 01:45-02:10 PDT)

Spec step 1 premise: R was already split on the host (band_consts: h = trunc9(r), l = trunc9(r - h),
CB_R holds Rh/Rl/RTh/RTl and every pass runs zh·Rh + zl·Rh + zh·Rl). So the 2.8e-4 floor is not
from unsplit weights. New diagnostics (fe8578b4, 36041f84):
- `LOSS_RDIAG=t<preset>` replaces the Gaussian with other 11-tap sets; the bench writes the taps
  it used (taps.txt) and loss_check compares against an fp64 separable filter with those taps.
  g9 = Gaussian masked to 9 bits (no lo part, so every product is exact), g5 = 5 bits,
  pw = powers of two (2^-10..2^-2), gl = lo parts only.
- `_q<n>` suffix quantises the input to multiples of 2^-n (fixed point), to test whether the
  error depends on the inputs' exponents.

| case | taps | l2rel vs fp64 (same taps) | mean rel bias |
|---|---|---|---|
| dump64 | default split3 Gaussian | 2.92e-4 | – |
| dump64 | g9 (exact products) | 2.94e-4 | +2.28e-4 |
| dump64 | g5 | 2.43e-4 | +1.80e-4 |
| dump64 | pw | 1.06e-4 | +6.83e-5 |
| dump64 | gl | 1.28e-3 | -1.27e-3 |
| syn | g9 | 2.85e-4 | +2.61e-4 |
| syn | pw | 9.5e-5 | +7.9e-5 |
| dump64 | R = c·I, c of 5/6/9 bits (single product) | 7.2e-7 .. 7.5e-7 | PASS |
| dump64 | pw, input q12 / q16 / q20 | 1.07e-4 each | |
| dump64 | g9 q12 | 2.94e-4 | |
| dump64 | g5 q8 | 2.54e-4 | |
| syn | pw q12 / g9 q12 | 9.55e-5 / 2.85e-4 | |

What this shows:
- Single products are exact (also with two 9-bit operands). The error is all in the
  multi-term accumulation inside one FPU matmul: a positive relative bias that does not depend
  on content and grows with the spread of product magnitudes (box ~1e-5, powers of two ~7e-5,
  Gaussian ~2.6e-4). Fixed-point inputs do not change it. Spec step 1 (split R) cannot reach
  1e-5: it is already in place and the floor is the accumulator, not the operands.

### Propagation to loss and dL/dimage (opt/t714/dssim_prop.py, model, T0 tolerance)

Every filter output multiplied by (1 + e), run through loss_and_grad_device, compared to exact
fp32. T0 bar: grad l2rel <= 1e-4 and |loss err| < 1e-5. bike_* = 256x384 crops of the real
bicycle photo _DSC8679 (images_4) against a blurred + noisy copy (23.9 dB).

| case | e = measured 2.8e-4 bias | 1e-5 | 3e-6 | 1e-6 | fp32 rounding only |
|---|---|---|---|---|---|
| dump64 | 9.2e-4 FAIL | 3.6e-5 PASS | PASS | PASS | PASS |
| syn | 332 FAIL | 2.3e-3 FAIL | 6.1e-4 FAIL | 1.8e-4 FAIL | 1.3e-5 PASS |
| bike_top | 5.2e5 FAIL | 5.7e-3 FAIL | 1.5e-3 FAIL | 4.3e-4 FAIL | 2.0e-5 PASS |
| bike_mid | 0.42 FAIL | 7.5e-4 FAIL | 2.0e-4 FAIL | 6.0e-5 PASS | 3.0e-6 PASS |

(grad l2rel; bike_top at 1e-5 also has loss err 7.5e-5, SSIM err -3.7e-4.)

Why: sigma^2 = E[x^2] - mu^2 cancels, so a relative filter error is amplified by about
mu^2 / C2 (C2 = 9e-4) in flat regions. Real images need the filter at about fp32 precision
(<~1e-7 relative).

Oracle verdict: the FPU split3 filter FAILS the T0 oracle on loss and dL/dimage, by orders of
magnitude. Even the box-level 1e-5 accumulation error fails on real images, so no FPU-fp32
accumulation path is good enough as is.

Not done, and why:
- Spec 2b (one non-zero tap per K group per matmul, so sums happen in fp32 dest): exact, but
  each band matmul turns into ~11 single-diagonal matmuls, about 7x the matmul count. Estimate
  (not measured) ~1.5 ms per filtered map, x6 maps per step: not viable against the 1-2 ms/view
  training target.
- Spec step 3 (fused L1 + D-SSIM kernel): its precision prerequisite fails; not started.
- Filter timing unchanged: 0.209 ms/view b2b at 1237x822 (#706, not re-measured).

Exact paths for the next task (sized only, none measured):
1. Integer FPU matmul: Blackhole LLK has Int8 srcA/srcB with int32 dest
   (tt_llk_blackhole/llk_lib/llk_math_common.h, ALU_ACC_CTRL_INT8_math_enabled, needs fp32 dest
   mode). Sums are exact integers, so the only error is the fixed-point quantisation of x and the
   taps, chosen up front (e.g. x as 3 int8 limbs, taps as 2-3 limbs). Needs a feasibility probe
   (int8 x int8 -> int32 matmul_tiles with a known answer) and a cost size.
2. SFPU fp32 FMA filter (11 taps x 2 passes per pixel per map), exact to fp32 rounding; size it.
3. Better conditioning as a complement: filter (x - c_T) with a per-tile constant c_T and add
   c_T back (G * c is known), which cuts the mu^2 / C2 amplification; still needs a filter far
   better than 1e-5 on real images.
