# t647: cull-side gate for absolute means (reopen check for #634)

Run: bh-33 (measurement reservation, under `ttp lock p100`), 2026-10-13 14:31-14:38 PDT,
commit b095d258 (kernel gate code from f764a093). 4 rotated rounds of
`render/run.py --back-to-back`, 30 bicycle views, 20 passes, warmup 1.

Arms (both timing-only, both with `OL_ABS_MEANS_ABL=1`):
- **abl**: `CULL_ABS_RAW=0`, cull still runs `dm_fp32::add_sub_roundtrip` (microblock_cull_compute.cpp:420-421).
- **raw**: `CULL_ABS_RAW=1`, cull uses rec[4]/rec[5] directly.

| round | order | abl ms/view | raw ms/view | abl - raw |
|---|---|---|---|---|
| r1 | abl, raw | 7.431 | 7.331 | 0.100 |
| r2 | raw, abl | 7.359 | 7.318 | 0.041 |
| r3 | abl, raw | 7.438 | 7.391 | 0.047 |
| r4 | raw, abl | 7.408 | 7.328 | 0.080 |

Median delta 0.064 ms/view (mean 0.067). All 4 rounds positive. The r2 abl run drifted down
mid-run (passes 7.42 -> 7.31), the same pattern #634 saw; using per-run pass medians instead
of means gives 0.098 / 0.023 / 0.043 / 0.087, median 0.065.

Correctness: both arms give identical output (raw md5 a7bdd047 in every run; 30/30 dumped view
md5s identical between arms). Both fail the golden md5 (list f3d43bfe != 077caaba), as expected
for the timing-only ABL path, same as #634's abl arm.

## Verdict: reopen criterion passes (narrowly)

0.13 (emit, #634) + 0.064 / 2 = **0.162 ms/view b2b >= 0.15**. Margin 0.012 ms. Using the
smallest round (0.041) it is 0.150, right at the bar. If the full cull delta is realised (cull
consumes absolute means with no roundtrip) the bound is 0.13 + 0.064 = 0.19 ms/view (~2.6 %).

Proposed step 2 (not built here): real absolute-means build: emit writes absolute fp32 means
(drop the sub-int32 split) and cull/blend read them directly, with a correctness path that keeps
the golden md5 (or a new golden with PSNR vs reference_v2), then a kept-iteration b2b A/B
against the current best.
