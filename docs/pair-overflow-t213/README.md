# t213: no-CQ1 error path + forced pair-overflow smoke

Two follow-ups from the t198 review (docs/two-cq-t198/REVIEW.md, 49699a1).

## Error path (no CQ1)

With `GSPLAT_TT_SORT_OL_EARLY=1` and `GSPLAT_TT_MAT_CQ1=0`,
`tile_assign_fused_k2` reads proj_M into the stack vector `mread` with a
non-blocking `ReadShard`, then enqueues the early sort. If that enqueue threw,
the stack unwound while the read was still pending. The enqueue now runs
through `pair_guard::drain_on_throw` (render/host/pair_guard.h), which calls
`Finish(CQ0)` before rethrowing. The normal path is unchanged (no extra sync).
Unit test: tests/unit/test_pair_guard.cpp.

## Forced pair overflow

`GSPLAT_TT_PAIR_CAP_TEST=<pairs>` (test only, default 0 = off) sizes the pair
buffers for that many pairs instead of `pair_ceiling()`, so views over it take
the K2 overflow path: drain the early sort, grow the buffers, rerun the K2. Each
regrow logs `[TA] K2 pair overflow: P=.. cap=.., regrow and rerun`.

Run: `drive.sh yyzo-bh-04 e0916b79` (2026-10-07 12:17-12:18 UTC, yyzo-bh-04
p100a, the current measurement box; the spec named yyzo-bh-07, which is no longer
reserved). 30-view bicycle bench, defaults on, one devrun under `ttp lock p100`.

| arm | env | regrows | ms/view | md5 (30 views) |
|---|---|---|---|---|
| base | defaults | 0 | 10.885 | 906e0435 |
| ovf | `PAIR_CAP_TEST=1000000` (early sort + CQ1) | 6 | 12.754 | 906e0435, matches base |
| ovf_nocq1 | same + `MAT_CQ1=0` | 6 | 13.000 | 906e0435, matches base |

No `fused K2 failed` fallbacks, no errors. The extra ms/view is the six regrows
(max view 24 ms); it only matters for this test knob. The spec's expected md5
46a725ab is the t198-era default; the current default (iter 207 on) is 906e0435,
and all three arms match it. Logs and md5 lists: out/.
