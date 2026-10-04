# Task #142: lever C (dead-record pre-cull) device A/B

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024,
untraced, 4 rounds with the arm order rotated. Driver: `drive.sh` (one
`ttp lock p100` hold). Pixel diff: `imgdiff.py`.

## Run 1: on the lever A tip (before lever B), commit cc139c2

| round | base ms/view | pre-cull ms/view | delta |
|---|---|---|---|
| r1 | 24.673 | 23.996 | -0.677 |
| r2 | 24.555 | 24.078 | -0.477 |
| r3 | 24.406 | 24.231 | -0.175 |
| r4 | 24.418 | 24.084 | -0.334 |
| mean (stdev) | 24.513 (0.126) | 24.097 (0.098) | -0.416 (0.213), -1.70% |

Host stage means (base -> pre-cull): project 6.44 -> 6.71 (+0.27, pfwc does the
extra rect math), tile_assign 1.43 -> 1.40, sort 5.10 -> 4.97, blend 11.27 -> 10.73.

Tracy (10 views, per-view busiest-core makespan, base -> pre-cull): pfwc 2.69 ->
2.99, sort_ol_emit 4.03 -> 3.91, sort_ol_count 1.02 -> 0.98, ta_bucket_scatter
1.55 -> 1.50, mat_cull_mask 3.95 -> 3.34, tile_blend_sfpu 7.82 -> 7.75.
(`out/oldtip-t142-*-zones.txt`; the pre-cull gaps table failed: a zone
straddles a frame boundary in the stitcher.)

Dead records (hero-slab dump, 30 views): 3,369,033 records, 19.7% dead ->
3,207,709 records, 15.7% dead (-4.8% records, -24% dead).

Correctness: not byte-identical. Every differing pixel is off by exactly 1 LSB,
0.3-0.6% of pixels, PSNR vs base 74.5-78.0 dB, the same in all 4 rounds
(deterministic). hero_vs_ref 75.78 dB. Cause: the blend T-saturation early-out
reads T back every 512 records, dead ones included, so fewer dead records move
where a microblock stops. With the early-out off (BLEND_T_PERIOD=0) both arms
are byte-identical over 5 views (`out/oldtip-t0.log`): no live record is lost.
