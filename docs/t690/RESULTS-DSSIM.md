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

## t720: SFPU fp32 FMA separable filter (`LOSS_MODE=sfilt`)

Kernel: `render/bench/kernels/loss/sfilt_compute.cpp` (11-tap H pass then V pass, fp32 FMA on the
SFPU, zero padding, taps hoisted into LRegs: `SF_CMODE=1`), driven by `loss_bench.cpp`. Check:
`opt/t720/sfilt_check.py` (device output vs fp64 filter, vs its bit model, and D-SSIM loss and
dL/dimage vs the T0 oracle). Driver `docs/t720/drive.sh`; run log
`docs/t720/run-23e60ae1-bh-49.log` (commit 23e60ae1, p150 bh-49, 2026-10-14 03:56 PDT).
No per-tile conditioning was needed: the plain fp32 filter already sits at fp32 rounding.

Filter error (device output, measured):

| case | planes | l2rel vs fp64 | maxrel | bias | bit-match vs fused FMA model |
|---|---|---|---|---|---|
| dump64 (T0 dump, 64x64) | 24 | 3.90e-8 | 1.63e-7 | +6e-12 | 97.7% |
| syncrop (256x384) | 24 | 3.76e-8 | 1.80e-7 | -3e-11 | 97.3% |
| syn1 (1237x822) | 3 | 3.70e-8 | 1.80e-7 | -3e-11 | 99.2% |

The FPU split3 filter was 2.9e-4 on the same dump64 planes, so this is about 7000x lower and
unbiased. The device and the bit model have the same error against fp64 (3.9e-8 each); the
remaining bit mismatches are last-bit FMA rounding differences (large ulp counts only near zero).

T0 oracle (D-SSIM loss and dL/dimage with the filter's bit model, pass: grad l2rel <= 1e-4,
|loss err| < 1e-5): **PASS** on all 4 cases.

| case | loss err | dL/dimage l2rel | exact fp32 filter, same algorithm |
|---|---|---|---|
| dump64 | +9.7e-10 | 2.2e-7 | 1.8e-7 |
| syncrop | -4.8e-8 | 8.1e-6 | 5.6e-6 |
| bike_top (render vs _DSC8679) | +1.9e-8 | 1.2e-5 | 7.0e-6 |
| bike_mid | -5.0e-9 | 2.2e-6 | 1.4e-6 |

The FPU split3 filter failed the same oracle by orders of magnitude (t714).

Timing (p150 bh-49, 1248x832 padded = 1237x822, 110 cores, b2b over 4 rounds, iters=50 each):

| arm | planes | b2b ms/view (4 rounds) |
|---|---|---|
| FPU split3 filt (x, 3 ch) | 3 | 0.2089, 0.2089, 0.2090, 0.2091 |
| SFPU sfilt (x, 3 ch) | 3 | 0.1831, 0.1829, 0.1821, 0.1819 |
| SFPU sfilt, 5 forward maps | 15 | 0.8003, 0.7997, 0.7997, 0.7998 |
| SFPU sfilt, per training step (x, x^2, xy + 3 backward maps; target maps cached) | 18 | 0.9533, 0.9532, 0.9542, 0.9531 |

Same-workload comparison: SFPU 0.182 vs FPU 0.209 ms/view per 3-plane map. The 0.027 ms gap is
under the spec's 0.1 ms noise line, so call them equal in speed; the SFPU one is exact. Cost is
linear in planes (about 0.053 ms per 1237x822 plane).
A/B: `opt/ab_jit_check.py` on both arms exits 1 as expected: each arm's compute kernel
(filt_compute / sfilt_compute) exists only in its own arm, and the JIT hashes differ.

Fit: a full training step's filtering is 0.95 ms/view, about half of the 2 ms upper target and
almost all of the 1 ms lower one. It is exact, but too slow to leave as is. At ~22 FMAs per pixel
per plane it runs about 0.43 TFMA/s on 110 cores, well under the SFPU's peak, so the kernel is
bound by Dst load/store and tile movement, not FMAs.

Next: (1) fused L1 + 0.2*(1-SSIM) forward+backward kernel built on this filter (the products,
SSIM map and backward maps formed in the same pass, so the 18 planes never go to DRAM);
(2) cut the per-plane cost (keep the H-pass result in Dst/LRegs for the V pass, fewer Dst
round trips per tap, reuse loaded rows across the 6 maps of one channel).

## t736: fused L1 + D-SSIM loss and dL/dimage on the sfilt filter (`LOSS_MODE=ssim`)

Kernels: `render/bench/kernels/loss/ssim_fwd_compute.cpp` (15 filtered planes: mu_x, mu_y, E[x^2],
E[y^2], E[xy] per channel, SSIM map, L1 + 0.2*(1-SSIM) loss partials and the 9 backward d-map
planes, all formed in L1/Dst; only the d maps leave the core) and `ssim_bwd_compute.cpp` (filters
the 9 d-map planes, read back from DRAM for the filter halo, and forms dL/dimage with the L1
sign term). Lambda 0.2 as in 3DGS. Shared SFPU helpers in `ssim_sfpu.h`; host in `loss_bench.cpp`.
Check: `opt/t736/ssim_check.py` (bit model, prep, device check vs the T0 oracle). Driver
`docs/t736/drive.sh`; log `docs/t736/run-7b6d018a-bh-49.log` (commit 7b6d018a, p150 bh-49,
2026-10-14 04:58-05:02 PDT).

T0 oracle (pass: grad l2rel <= 1e-4, |loss err| < 1e-5). Bit model: `docs/t736/model-oracle.txt`.
Device (measured): **PASS** on all 6 cases.

| case | model loss err | model grad l2rel | device loss err | device grad l2rel | device max abs rel |
|---|---|---|---|---|---|
| dump64 (T0 dump) | +9.7e-10 | 2.2e-7 | -6.2e-9 | 2.0e-7 | 4.8e-7 |
| syncrop | -4.8e-8 | 8.1e-6 | -4.6e-8 | 8.5e-6 | 2.4e-5 |
| bike_top (render vs _DSC8679) | +1.9e-8 | 1.2e-5 | +1.7e-8 | 1.2e-5 | 4.7e-5 |
| bike_mid | -5.0e-9 | 2.2e-6 | -6.3e-9 | 2.1e-6 | 1.4e-5 |
| hero (T1, 1237x822) | +8.3e-10 | 1.8e-6 | +5.5e-9 | 1.8e-6 | 8.8e-6 |
| syn (1237x822) | - | - | -6.0e-8 | 7.5e-6 | 2.9e-5 |

Timing (p150 bh-49, 1237x822, 110 cores, b2b, 4 alternated rounds, iters=50 each):

| arm | planes filtered | b2b ms/view (4 rounds) | mean |
|---|---|---|---|
| fused ssim: full loss + dL/dimage (fwd + bwd programs) | 24 | 1.3110, 1.3100, 1.3106, 1.3078 | 1.310 |
| of which forward program | 15 | 0.664, 0.665, 0.666, 0.667 | 0.666 |
| unfused sfilt syn6: filter only, target maps cached | 18 | 0.9541, 0.9565, 0.9530, 0.9537 | 0.954 |

The two arms are not the same work. The fused arm is the whole loss and gradient but filters 24
planes (it refilters mu_y and E[y^2] of the target each step). The sfilt arm filters only, with the
target maps cached. Per filtered plane: fused 0.0546 ms vs filter only 0.0530 ms, so all the SSIM
math, L1, loss reduction and d-map forming cost about 0.04 ms per view on top of filtering
(under the 0.1 ms noise line). Filtering is still about 97% of the loss cost.
`opt/ab_jit_check.py` exits 1 as expected: each arm's compute kernels (ssim_fwd/ssim_bwd vs
sfilt_compute) exist only in their own arm, so the gate cannot be A/A here; both arms ran from
their own JIT caches in the same session. The render path is not touched (loss bench only), so no
hero.png is needed for this step.

Next toward the 1-2 ms/view training target:
1. Cache the target's filtered maps (mu_y, E[y^2]: 6 planes) once per training image. Expected
   about 0.33 ms off (24 -> 18 planes), bringing the loss to about 0.98 ms/view.
2. Cut the per-plane filter cost (~0.053 ms): keep the H-pass result in Dst/LRegs for the V pass,
   share loaded rows across the 5 maps of one channel, fewer Dst round trips per tap.
3. Fuse the loss into the end of the device rasterizer backward so the rendered image never
   goes through DRAM.

## t742: caching the target's filtered maps — already in t736 (no change)

The t736 section above says the fused arm filters 24 planes per step and refilters mu_y and
E[y^2] of the target. That is wrong; the t736 code already caches them:

- `loss_bench.cpp` `run_ssim` reads `consts.f32` (6 planes: mu_y and vy = E[y^2] - mu_y^2 per
  channel), written once per GT image by `opt/t736/ssim_check.py prep` (`consts()`, the bit model of
  the device filter), and uploads them to DRAM once, outside the timed loop.
- `ssim_fwd_compute.cpp` `h_sfpu()` forms only x*y and x^2 and filters 3 maps per channel
  (G*x, G*x^2, G*(x y)); `v_pass()` filters the same 3. mu_y and vy come from DRAM via CB 2
  (`reader_ssim.cpp`, const planes ch and 3 + ch). Forward: 9 planes.
- `ssim_bwd_compute.cpp` filters the 9 d-map planes. Total: **18 planes per step**, the same
  count as the sfilt syn6 arm. `gsplat/train_loss.py` documents the same design.

So the 1.310 vs 0.954 ms/view gap (0.356 ms) is not 6 extra filtered planes. It is overhead the
fused programs carry beyond filtering. From the t736 numbers (measured, bh-49): forward program
0.666 ms and backward 1.310 - 0.666 = 0.644 ms, each for 9 planes, vs 9 x 0.053 = 0.477 ms of
filter-only work per half. About 0.19 ms (forward) and 0.17 ms (backward) per view are spent on
other work: the extra transposes and product passes in the H step, the 4 fp32 reciprocals per
pixel in `ssim_pt`, the pointwise reads (consts, wm, x, y), the 9 d-map planes written to and
read back from DRAM, and the second program launch. Which of these dominates needs a profile.

The one-off target pre-filter is 6 planes per GT image. On the device it would cost about
6 x 0.053 = 0.32 ms per image (derived from the measured sfilt per-plane cost, not measured),
once per training image, so it does not affect the per-step time. No device run was made for
t742: the change it asked for is already in the measured t736 arm.
