# Task #410: eth-dispatch launch overhead in sort/project — not worth chasing (shelved)

## Question
t397 (pre-xvpin stack) measured eth dispatch ~0.19 ms/view slower than worker at an equal 11x10
grid in single runs (sort bin_emit 0.593 vs 0.451, project 3.072 vs 2.980 ms). Does that gap
still exist on the xvpin + zero-copy + eth stack (t409 head f1632a5d)?

## Confirm step on bh-30 (p150b, viewer reservation IRD job 135630, no reserve/extend/release)
Bicycle, 30 views, 1024x1024, untraced, GSPLAT_PER_VIEW_STAGES=1, 3 alternating rounds
(`drive410.sh` -> `bench410.sh`, logs in `out-confirm/`). E = eth dispatch capped to 11x10
(`GSPLAT_TT_GRID_X=11`), W = `GSPLAT_TT_DISPATCH=worker` (11x10). Tree p150bench/tree392 at d6228b6f
(= f1632a5d + these scripts; render .so unchanged, ninja no work).

| round | E eth11 (ms/view) | W worker11 (ms/view) |
|---|---|---|
| r1 (E,W) | 8.889 | 8.946 |
| r2 (W,E) | 8.878 | 8.872 |
| r3 (E,W) | 8.890 | 8.860 |
| mean | **8.886** | **8.893** |

Gap eth - worker = **-0.007 ms/view** (eth is not slower; within round noise of ~0.04 ms).
md5: all 6 runs `906e0435` 30/30 = 11x10 golden (`MD5_GOLDEN_OK`).

Stage means (E vs W): sort bin_emit 0.441 vs 0.436, bin_layout 0.081 vs 0.084, publish_host
0.319 vs 0.317, mat 0.094 vs 0.096; project pfwc_rtargs 0.091 vs 0.093, pfwc_enqueue 0.034 vs
0.028, gather_wait 1.120 vs 1.126. No launch or wait grows under eth dispatch.

## Verdict
Gap < 0.1 ms/view (< 1%), so per the spec the task stops here: the t397 gap does not exist on the
current stack (the t397 runs predate xvpin/zero-copy and were single runs). No code change, no new
iteration, no screenshot needed.

The eth12 default vs worker11 difference in sort (t409: 1.173 vs 1.060 ms) comes from the grid
(mat 0.083 -> 0.124, publish_host 0.315 -> 0.341 at 120 vs 110 cores), not from dispatch.

## Viewer
Stopped 2026-10-08T02:40:15Z, restarted 02:42:56Z, READY and localhost:8091 -> 200 at 02:43:04Z.
(An earlier attempt stopped it 02:37:10Z-02:37:56Z; its E smoke check regex was wrong and the
bench exited before any timed run.)
