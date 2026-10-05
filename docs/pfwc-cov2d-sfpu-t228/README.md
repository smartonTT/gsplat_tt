# t228: P2, cov2d a/b/c + conic + radii as two SFPU sections

Build of "P2" from [docs/pfwc-trisc-model-t226](../pfwc-trisc-model-t226/README.md)
(modeled -0.48 ms/view, range 0.39-0.50). Knob `GSPLAT_TT_PFWC_COV2D_SFPU` (default 0
while building, `=1` on).

## What changed

Steps 7-11 of `render/kernels/compute/project_pfwc_compute.cpp` (cov2d a, b, c, the
conic fold and the two radii) ran as six DEST acquires: 38 `copy_tile` and 87 SFPU tile
ops per chunk plus the conic pass, because j00, j02, j11, j12 were rebuilt in five of
them and a and c twice. With the knob on they run as two acquires:

- **S_AC**: copy cc00, cc02, cc22, cc11, cc12, inv_tz, tx, ty into slots 0-7; one looped
  SFPU pass (`pfwc_cov2d::run_ac`, `render/kernels/compute/pfwc_cov2d_sfpu.h`) computes
  j00, j02, a, then j11, j12, c and stores a to slots 0 and 6 and c to slots 3 and 7;
  then the same `relu_tile`, `sqrt_tile`, `mul_unary_tile(k)`, `ceil_tile` as before on
  slots 6 and 7. Packs: 0 -> CB_TMP_A, 3 -> CB_TMP_C, 6 -> CB_RX (+ CB_TMP_RX),
  7 -> CB_RY (+ CB_TMP_RY).
- **S_BC**: copy cc02, cc01, cc12, cc22, inv_tz, tx, ty into slots 0-6; `run_b` computes
  b into slots 1 and 7; copy CB_TMP_A -> 0 and CB_TMP_C -> 2; the unchanged conic fold
  (`pfwc_conic_unroll<0>`); packs 0/1/2 -> CB_A/B/C and 7 -> CB_TMP_B.

The passes are raw `TTI_SFPLOAD/SFPMUL/SFPADD/SFPSTORE` with fixed tile addresses and
`sfpi::dst_reg++` per vector, in a `#pragma GCC unroll 0` loop. Every product
(SFPMUL, + 0.0) and sum (SFPADD, * 1.0) is rounded on its own in the default path's
order, so the result is bit-identical (no SFPMAD fusion, as in t206's
`pfwc_covcam_sfpu.h`). Two loop-invariant constants per pass live in LREG0 / LREG7
(2.0 and 0.3 in run_ac, fx and fy in run_b); the rest come per vector from `SFPLOADI`
pairs. Per chunk: copies 38 -> 17, SFPU tile ops 87 -> 8 (the radii's relu / sqrt / k /
ceil) plus the two passes, packs unchanged (10).

## Host check

`tests/unit/run_cpp.sh tests/unit/test_cov2d_sfpu.cpp`: the header's real instruction
stream on a model of DEST, LREGs and the DEST counter, LREGs starting as garbage, against
steps 7-9 op by op. 3000 chunks (scene-like, any finite bits, specials):
18,432,000 lane checks, 0 mismatches, inputs the passes must keep are kept, the DEST
counter ends at 64 rows. A planted wrong load is caught. Per chunk: run_ac 1316 TTI +
256 runtime SFPLOADI + 32 INCRWC, run_b 896 + 132 + 32 (~2.7k SFPU instructions in all).
`tests/syntax_stub/check.sh` compiles the kernel for all three TRISCs with the knob, with
and without VIS / PRECULL / WSPLIT / STEPCYC.

## Device run

`dev/drive.sh <rev>` (yyzo-bh-07 p100a, tree /localdev/smarton/gstt2-t228, every step under
`ttp lock p100`): sync, 30-view smoke with the knob on (md5 gate; a "too large" program
retries at `GSPLAT_TT_KCFG_EXTRA_KB=32`), Tracy STEPCYC=1 split for knob 1 and knob 0
(views 0-4, +32 KB), then 4 swapped untraced 30-view rounds base / cov2d. Logs in
`dev/out/`, summary in `dev/out/summary.txt` (`dev/summ.py`).

Gates (task spec): md5 46a725ab on every arm; TRISC_1 steps 6-11 (S_AC + S_BC) <= 16k
cycles/chunk; paired untraced drop >= 0.3 ms/view.

Results: pending (chain started 2026-10-05 11:00 on 543869b).
