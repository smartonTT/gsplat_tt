# t397: ETH dispatch vs worker dispatch on the bh-30 p150 (device A/B)

Code: tree392 = ttp/t392 e2d6e822 (opt tip 58b47824 + t383 `GSPLAT_TT_DISPATCH`, opt/eth overlay v2)
plus the scripts in this dir (288aeffc). Box: bh-30 (p150b, aiclk 1350, IRD job 135630, the
user's viewer reservation; run under the 2026-10-07 viewer exception, no reserve/extend/release).
Both arms use the same overlay (`ttm-eth12v2`, 12 ETH dispatch cores, idle-ERISC .ld 32 KB) and the
same JIT cache. Untraced, `GSPLAT_PER_VIEW_STAGES=1`, 30 views per round, alternating order.

Viewer: stopped 2026-10-08T01:06:57Z, restarted 01:10:49Z, READY + localhost:8091 -> 200 at
01:10:58Z (selftest 11.52 ms/view, sha 8329be50). Down ~4 min.

## Result

| round | worker (11x10, 110 cores) | eth (12x10, 120 cores) |
|---|---|---|
| r1 (W,E) | 11.145 | 10.595 |
| r2 (E,W) | 11.117 | 10.576 |
| r3 (W,E) | 11.150 | 10.561 |
| mean ms/view | **11.137** (sd 0.018) | **10.577** (sd 0.017) |
| FPS | 89.8 | 94.5 |

Delta **-0.560 ms/view (-5.03%)**; the model predicted ~-0.50. GPU reference: 10.75 ms (published, not
measured), so eth on p150 is 1.6% under the published GPU number. Worker is 3.6% over it.

Stage means (ms/view, 3 rounds):

| stage | worker | eth | delta |
|---|---|---|---|
| project (pfwc) | 2.978 | 2.959 | -0.02 |
| sort | 1.045 | 1.108 | +0.06 (bin_emit 0.435 -> 0.463) |
| blend (mat+blend) | 6.492 | 5.976 | **-0.516** |
| d2h | 0.545 | 0.450 | -0.09 (noisy; r2-E 0.312) |

The blend gain matches pure core-count scaling: 6.492 x 110/120 = 5.951 ms.

## Per-core busy

Worker Tracy (`out/tracy-u-worker.csv.gz`, `out/ana-worker.txt`): mat+blend TRISC busy per core
mean 5.967 ms, min 5.912, max 6.048 over 110 cores (656 core-ms per view; LPT balance is tight).
656 core-ms spread over 120 cores = 5.47 ms, i.e. -0.50 ms, which is what the untraced blend stage shows.

Eth Tracy failed: with `TT_METAL_DEVICE_PROFILER=1` the profiler-instrumented `cq_prefetch` built
for idle ERISC is 0x4668 bytes and overflows the 0x4490 region (`out/T-eth.log`). So the 120-core
per-core table could not be captured; the 12th column's share is inferred from the untraced blend
stage scaling exactly 110/120. Fix idea: a profiler-only overlay with fewer dispatch zones, or
profile only Tensix (dispatch-core profiling off).

## Correctness

- worker: md5 list = 906e0435 golden, 30/30, all 3 rounds.
- eth: **all 30 views differ from 906e0435**, but eth is deterministic: the list md5 is 39d84b28
  in all 3 rounds (`out/md5-r*-E.txt`).
- Hero, eth vs worker: 242 of 1,048,576 pixels differ (221 by 1 LSB, max 13), spread over 76 of
  1024 screen tiles with no bias to tile edges or to any tile column; PSNR eth vs worker 82.2 dB.
- Hero vs `benchmarks/reference_v2/hero.png` (the named reference): eth **42.513 dB**, worker
  42.512 dB. md5/golden badge kept separate: eth hero c07dfb8d does not match the worker golden hero.
- Visual check (hero/hero.png, hero/diff_vs_ref_x8.png): no tile seams or blocks, no artifacts along
  any column; diff vs reference shows the usual spoke/edge structure, same as worker.

Most likely cause: some step partitions work by core count (110 vs 120) and that changes the
float/tie order for a few pixels. The run logs show no capacity or overflow messages, and the only
log difference is `cores 110` -> `cores 120` in the sort ONELAUNCH lines. Not isolated yet.

## Why eth differs: core count, not eth dispatch (attempt 2, commits 2b4f093f + 758ea493)

`GSPLAT_TT_GRID_X` / `GSPLAT_TT_GRID_Y` (new, default off) cap the compute grid every stage partitions
over (`device_state::cap_grid`, applied at all 8 `ctx.grid` sites). Built at 758ea493 in tree392 (nice 19,
ionice -c3, half the cores, viewer running), then one viewer-down window on bh-30 (`drive397b.sh`,
`bench397b.sh`; same overlay and JIT cache as the A/B; untraced, 30 views per run, one run per arm):

| run | dispatch | stage grid | ms/view | project | sort (bin_emit) | blend | d2h | view md5 list | hero md5 |
|---|---|---|---|---|---|---|---|---|---|
| E11 | eth | 11x10 (capped) | 11.338 | 3.072 | 1.217 (0.593) | 6.509 | 0.464 | **906e0435, 30/30 = golden** | 86524912 |
| W11 | worker | 11x10 | 11.152 | 2.980 | 1.114 (0.451) | 6.430 | 0.549 | 906e0435, 30/30 = golden | 86524912 |
| E12 | eth | 12x10 | 10.592 | 2.942 | 1.199 (0.526) | 5.867 | 0.489 | 39d84b28 (= A/B eth list) | c07dfb8d |
| W10 | worker | 10x10 (capped) | 12.024 | 3.244 | 1.017 (0.415) | 7.159 | 0.526 | e1bd4bfc (all 30 differ) | 739127f8 |

- Eth dispatch on the worker-dispatch shape (11x10) is **bit-identical** to worker dispatch: all 30 views
  and the hero match the 906e0435 golden.
- Changing the core count alone changes bits under worker dispatch too (10x10 -> e1bd4bfc). So some stage
  partitions work by core count in a way that changes rounding or tie order for a few pixels. That is a
  property of the existing pipeline, not an eth dispatch bug. 39d84b28 is the legitimate 12x10 golden
  (reproduced in 4 of 4 eth runs).
- At equal core count, eth dispatch is ~0.19 ms/view slower than worker (single run each: sort/bin_emit
  +0.14, project +0.09, d2h -0.09). At 12x10 the 10 extra blend cores more than pay for it (-0.56 ms).
  A launch-overhead cut for sort/project under eth dispatch is a possible further gain.
- Viewer (second window): stopped 2026-10-08T01:31:53Z, restarted 01:33:33Z, READY and
  localhost:8091 -> 200 at 01:33:42Z (selftest 11.55 ms/view, sha 8329be50). Down ~1 min 50 s.
  Logs: `out397b/` (`drv397b.log`, `E11/W11/E12/W10.log`, `md5-*.txt`).

## Recommendation

Eth dispatch is a real -5.0% on the p150 (gate is 1%; 10.577 vs 11.137 ms/view) and the image is visually
clean at the same PSNR (42.513 vs 42.512 dB against reference_v2). The md5 gap is explained: it comes from
the core count, not from eth dispatch (eth at 11x10 = 906e0435 30/30). Recommend making eth the p150 default
in a separate reviewed landing task, with a per-grid golden (11x10: 906e0435, 12x10: 39d84b28) and a new
iteration with its own device hero. Not landed, tagged or made default here.

Files: `out/drv397.log` (driver, viewer stop/start), `out/r*-{W,E}.log`, `out/md5-*.txt`,
`out/s0.log` (eth smoke, S0_PASS, grid 12x10), `hero/hero.png` (eth device render),
`hero/hero_worker.png`, `hero/diff_vs_ref_x8.png`, `hero/diff_eth_vs_worker_x20.png`.
