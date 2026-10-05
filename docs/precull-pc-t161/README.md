# Task #161: device A/B of the pixel-centre pre-cull (GSPLAT_TT_PRECULL=2)

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024,
untraced, 3 rounds with the arm order rotated. Tree: b54c54e (PRECULL=1 default)
plus the t157 commits (e6fabe4, ee2800f, cherry-picked), commit 6a00513.
Driver: `drive.sh` (phases, one `ttp lock p100` hold per call); summary:
`summarize.py` -> `out/summary.txt`.

Arms: off = PRECULL=0, base = PRECULL=1 (default since #156, lever C integer
rect), pc = PRECULL=2 (pixel-centre rect, #157).

## Fit in the kernel config buffer (70656 B)

The PFWC_PRECULL+PRECULL_PC program builds, loads and runs (an overflow throws
at load, as in t142). TRISC load sizes (readelf LOAD segments): mode 2 trisc1
30992 B vs mode 1 30988 B (+4 B), trisc0 10984 B both, trisc2 6000 B both. No
trim needed.

## Time (untraced, ms/view)

| round | off (0) | base (1) | pc (2) | pc - base |
|---|---|---|---|---|
| r1 | 19.690 | 19.169 | 18.370 | -0.799 |
| r2 | 19.649 | 19.148 | 18.424 | -0.724 |
| r3 | 19.682 | 19.108 | 18.359 | -0.749 |
| mean | 19.674 (50.8 FPS) | 19.142 (52.2 FPS) | 18.384 (54.4 FPS) | **-0.757 (-4.0%)** |

Host stages base -> pc: project 4.65 -> 4.60, sort 5.00 -> 4.65, blend 9.28 -> 8.89.

The host estimate was 1.2-2.3 ms over mode 1; the measured gain is 0.76 ms, below
that range: the post-cull stages do not scale linearly with record count (see Tracy).

## Tracy (10 views, per-view makespan of the busiest core, ms)

| zone | off | base | pc | pc - base |
|---|---|---|---|---|
| pfwc | 3.096 | 3.214 | 3.213 | 0.00 |
| sort_ol_count | 1.028 | 0.985 | 0.886 | -0.10 |
| sort_ol_emit | 4.016 | 3.933 | 3.668 | -0.27 |
| sort_ol_barrier | 0.497 | 0.472 | 0.437 | -0.04 |
| mat_cull_mask | 3.946 | 3.382 | 3.110 | -0.27 |
| tile_blend_sfpu | 6.197 | 6.091 | 6.146 | +0.06 |
| tile_blend_load | 5.852 | 5.785 | 5.768 | -0.02 |

Per-program busy (all-core, ms/view, `out/t161-m*-gaps.txt`), base -> pc: pfwc
2.944 -> 2.942, TA program (unnamed) 1.458 -> 1.404, sort 4.601 -> 4.266,
materialize 3.398 -> 3.017, blend 5.920 -> 5.829. Traced makespan 19.54 -> 18.72.
`ta_bucket_scatter` has no zone in this build (TA runs as the unnamed program).
Blend does not move: dead records are already dropped by mat_cull_mask, so the
gain is all in sort and materialize, which scale with record count only in part
(fixed per-tile work).

## Dead records (hero view, GSPLAT_TT_DUMP_CULL slab dump)

| arm | records | dead | dead share | records vs off |
|---|---|---|---|---|
| off | 3,369,033 | 664,773 | 19.73% | |
| base | 3,236,947 | 532,687 | 16.46% | -3.92% |
| pc | 2,895,701 | 191,441 | 6.61% | -14.05% |

Live records 2,704,260 in all three arms (no live record lost). The host model
predicted -3.94% (dev) and -13.96% (pc) on the hero view: matches. Max records
per tile 25,699 / 25,019 / 22,980 (model: dev 25,019, pc 22,976).

## Correctness

* pc vs base: every differing pixel off by exactly 1 LSB, at most 0.61% of
  pixels, PSNR 74.24-77.81 dB over 90 view comparisons (gate >= 70 dB).
  Deterministic: the same md5 list in all 3 rounds.
* off vs base: <= 1 LSB, PSNR 74.89-78.24 dB (as in t156).
* base md5 equals the golden (md5-r82new.txt = PRECULL=1) in all 3 rounds.
* BLEND_T_PERIOD=0, views 0:5: modes 0, 1 and 2 are byte-identical (`out/t161-ab3.log`,
  t0 lines). So mode 2 loses no live contribution; the 1 LSB comes from the T
  early-out checkpoints shifting, as for lever C.

## Recommendation

Make PRECULL=2 the default (separate review flips it): -0.76 ms/view (-4.0%)
measured, same live records, <= 1 LSB vs the current default, +4 B code. Flip
needs a golden refresh (md5-r82new.txt and hero_golden_8bit.png) like t156.
