# Task #181: faster sort window fill under the K2 fold

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 1024x1024.
Code: affaad2 (bulk fill on top of #187's stack fix, 8c09d1b = 61773b8 on the branch).

## What changed
With the K2 fold, each mover of `sort_bin_onelaunch` fills its gid/tid window at
kernel start, one 64 B NoC read per page and plane. The gid/tid buffers are
interleaved, so page p + nb sits right after page p in the same DRAM bank
(nb = bank count). `OL_FILL_BULK` reads the window as nb runs per plane (one
large read per bank run) into a staging plane (tid / keep), then copies each
page to its window slot. Kill switch: `GSPLAT_TT_OL_FILL_BULK=0`
(`=2`: debug, re-reads every page and DPRINTs mismatches).

## Fill (Tracy, views 0:10, zones sort_ol_fill / sort_ol_fillb incl. read barrier)
| arm | NCRISC fill makespan | BRISC fill makespan | fill union | GB/s (<= 20.7 MB/view) | prefix makespan |
|---|---|---|---|---|---|
| off (64 B pages) | 412.5 us | 764.1 us | 764.1 us | 27.1 | 377.0 us |
| bulk | 131.9 us | 147.0 us | 147.0 us | 140.6 | 83.9 us |

The page fill was request-bound (27 GB/s); larger reads give 5.2x the rate.
Files: out3/tracy-{on,off}-{fill,zones,gaps}.txt.

## Paired untraced A/B (30 views, 3 rounds, arm order rotated)
| round | off view_total | bulk view_total | delta | sort off -> bulk |
|---|---|---|---|---|
| r1 | 16.983 | 16.587 | -0.396 | 3.780 -> 3.407 |
| r2 | 16.994 | 16.640 | -0.354 | 3.818 -> 3.429 |
| r3 | 17.025 | 16.749 | -0.276 | 3.858 -> 3.510 |
| mean | 17.001 | 16.659 | **-0.342 ms/view (-2.0%)** | -0.370 |

avg_frame_ms deltas: -0.401 / -0.355 / -0.278 (mean -0.345). Blend unchanged.
All 6 runs: ALL_VIEWS_IDENTICAL vs md5-r82new.txt (46a725ab set).
Files: out3/run-r*.log, out3/md5-r*.txt.

Note: the base predates #188 (blend late claim). The gain is in the sort stage,
which #188 does not touch; the combined tip was not measured here.

## History
The first device chains (out/, out2/) hung in sort_ol in both arms: the kill
switch hung too. Cause: kernel_main ran at ~0 B stack margin (4 KB cur_lm on the
stack, task #186/#187); the extra code overflowed it. #187's fix (cur_lm -> CB_CUR)
removed the hang.

Decision: LAND (gate: md5-identical and >= 0.3 ms/view).
