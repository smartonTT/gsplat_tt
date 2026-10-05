# Task #206: single-pass SFPU cov_cam (GSPLAT_TT_PFWC_COVCAM_SFPU, default 0)

Built and host-tested only. No device run yet: the numbers below are estimates, not
measurements.

## What changes

pfwc step 6 (cov_cam, `project_pfwc_compute.cpp`) computes the six camera-space covariance
entries `cc_e = sum_k s_ek * c_k` (per-view scales `s_ek` = runtime args 17 + 6e + k).

- Default path: six acquires. Each one does 6 copy_tile, 6 mul_unary_tile and 5
  add_binary_tile, then 1 pack. That is 36 copy_tile, 36 mul_unary and 30 add_binary per
  chunk. t197 traced 0.664 ms/view: copy 0.223, mul_unary 0.232, add_binary 0.155.
- `GSPLAT_TT_PFWC_COVCAM_SFPU=1` (`pfwc_covcam_sfpu.h`): one acquire with these steps:
  - 6 copy_tile, one per cov3d input, into DEST tiles 0..5.
  - The 36 scales are staged once into DEST tiles 6..7.
  - One SFPU pass computes everything. For each of the 32 vectors, it loads the 6 inputs into
    LREG0..5, then computes each entry in LREG6 and writes it back in place.
  - 6 packs.

  Per chunk this is 3264 TTI + 492 TT_ SFPU instructions.

## Bit-exactness (md5 46a725ab)

Every product is its own SFPMUL (`c_k * s`, + 0.0) and every sum its own SFPADD
(`acc * 1.0 + p`). Both are taken in the same order as mul_unary_tile and add_binary_tile.
`tests/unit/test_covcam_sfpu.cpp` runs the header's actual instruction stream on a model of
DEST and the LREGs, and compares the result with the default op sequence:

- 18.4M values (scene-like data, any finite bits, ±0/inf/NaN/denormals): 0 mismatches.
- Swapping the SFPADD operands gives the same result (one rounding).
- A fused SFPMAD variant (`acc = c_k * s + acc`) would save 960 instructions per chunk. It
  changes 24.7% of scene-like values, so it would change the md5. Not used.

Device-only residual risks, to be settled by the first device md5 check:

- The old path round-trips each partial sum through fp32 DEST between ops; the new path keeps
  it in LREG6. The new path round-trips the scales through DEST (SFPSTORE/SFPLOAD mode 0).
  Both are exact if fp32 DEST load/store is bit-exact; in question are only denormal or NaN
  payloads, which bicycle values should not hit.
- The sfpi multiply is assumed to add +0.0 (LCONST_0). This matters only for signed-zero
  results when every term is zero.

## Expected savings (estimate)

The old copy, mul and add work is about 15.1k cycles per chunk. The new path is 6 copies
(about 0.9k), plus an SFPU pass of 4.5-6.5k depending on SFPLOAD/SFPMAD stalls, plus about
0.3k overhead. That saves about 7.5-9.5k cycles per chunk, or 0.30-0.38 ms/view of math-thread
time (54.4 chunks per core, 1.35 GHz). On the critical path, pfwc is compute-bound (about
2.75 ms) against a writer busy for 2.0-2.2 ms, so the expected gain is about 0.25-0.38 ms/view.
That is near the 0.3 gate.

## Device check (next task)

1. First real SFPI compile with the knob on (`GSPLAT_TT_PFWC_COVCAM_SFPU=1`). Also build
   with `PFWC_VIS`/`PFWC_PRECULL` defaults to check TRISC1 code size.
2. Smoke: md5 must stay 46a725ab.
3. `GSPLAT_TT_PFWC_STEPCYC=2` split as in docs/pfwc-breakdown-t197. With the knob on,
   cov_cam is booked as copy (6 copies), mulu (the SFPU pass) and pack.
4. Paired A/B, knob 0 vs 1, untraced. Flip the default if the gain is at least 0.3 ms/view.
