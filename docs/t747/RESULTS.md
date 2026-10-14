# #747: win back the hero blend backward regression from the tile-centred power

Base: #728 kernel (32d726ec, `render/train/kernels/blend_bwd_compute.cpp` md5 bb21df9c).
Candidate: c84e9e03 on `ttp/t747-win-back-the-2-8-ms-hero-backward-regres`.
Device: bh-49 (p150), 2026-10-14 10:32-10:40 PDT, one `ttp lock p100` run of `ab.sh`
(one sync and build, the two kernel sources swapped in place, each arm with its own JIT cache;
sh0/sh3/s1/s2 once per arm with 3 reps, hero 5 ABBA rounds of 8 reps). Raw files: `results/`.

## Profile first (#728 per-stage split)

The 2.76 ms regression was bodies +1.87 ms (213 -> 218 cycles per body: 3 more Dst loads and
2 subtracts for the centred power) and pair setup +0.95 ms (P0, Lx, Ly staged into Dst).
The body was also loading 11 fp32 constants (exp polynomial, alpha_max, alpha_min, T_min) with
two SFPLOADI each, every body.

## Change (c84e9e03)

- 3 of the constants go to the programmable SFPU constant registers (no load in the body); the
  other 8 live in Dst vectors written once per kernel (one SFPLOAD instead of two SFPLOADI).
- Room in Dst: the x/y pixel ramps are read only from their canonical vectors (x of vector V =
  x of V&9, y of V = y of V&~9; unit test `test_ramp_vectors_repeat_for_kernel_reads`), which frees x ramp
  vectors 16-31.
- Same arithmetic in the same order: outputs are bit-identical.

## Timing: hero warm median

| arm | round medians (ms) | median of 35 warm runs | range |
|---|---|---|---|
| base (#728) | 89.107 89.075 89.078 89.105 89.105 | **89.10** | 89.07-89.17 |
| cand (c84e9e03) | 82.638 82.667 82.661 82.638 82.646 | **82.65** | 82.63-82.75 |

Delta -6.45 ms (-7.2%), 60x the 0.1 ms noise line. It is also 3.67 ms under the pre-fix
86.32 ms, so the centred-power cost is fully recovered and more.

Per-stage split, busiest core 107 (41,398 pairs, 492,267 bodies), ms at 1.35 GHz, busiest / mean:

| stage | before fix (#707) | base #728 | cand #747 |
|---|---|---|---|
| total | 86.28 | 89.04 / 53.31 | 82.60 / 49.76 |
| tile setup | 0.04 | 0.04 / 0.02 | 0.04 / 0.02 |
| record wait | 0.67 | 0.62 / 0.59 | 0.62 / 0.59 |
| pair setup | 4.35 | 5.30 / 5.20 | 5.30 / 5.20 |
| bodies | 77.65 | 79.52 / 44.05 | 73.07 / 40.45 |
| finish | 3.50 | 3.48 / 3.37 | 3.49 / 3.42 |
| pack sync | 0.08 | 0.08 / 0.08 | 0.08 / 0.08 |

Cycles per body 218.1 -> 200.4; pair setup unchanged at 172.9 cycles per pair.
Pair setup (+0.95 ms against #707) is the part of the regression left untouched.

## Oracle (T0 gradient oracle, `blend_backward_dev check`, bar 1e-2)

| dump | base rel err | cand rel err | first 9 floats per slot |
|---|---|---|---|
| sh0 | 3.74e-5 | 3.74e-5 | bit-identical |
| sh3 | 1.03e-4 | 1.03e-4 | bit-identical |
| s1 | 1.02e-3 | 1.02e-3 | bit-identical |
| s2 | 1.57e-3 | 1.57e-3 | bit-identical |
| hero (`--fast`, ties) | #728: 3.28e-3, 55,108 tie slots skipped, rel_err_all 0.0278 | HERO_CAND | |

Only the 7 padding floats per slot (never read) differ between runs and arms; the 9 output
terms match bit for bit on all four small dumps.

## JIT check

`opt/ab_jit_check.py --kernel blend_bwd_compute` on both arms: **JIT-HASH FAIL (A/A)**, both arms
have hash 11722400452286732710 (`results/ab_jit_check.txt`). That is a false positive of this
setup: tt-metal's kernel hash does not include the source text, and the A/B swaps the source in
place with a separate `TT_METAL_CACHE` per arm. The compiled binaries differ in both arms
(`results/jit-elf-md5.txt`: trisc0/1/2 ELF md5 da87cbb2/c84d273f/64b8288e base vs
52d8f0f3/9ee956d5/e732dd51 cand), and so do the timings.
