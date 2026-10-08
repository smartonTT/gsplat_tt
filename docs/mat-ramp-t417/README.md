# t417: smaller first mover item (GSPLAT_TT_MAT_RAMP) — measured, not kept

Cause of #413's mat start gap (0.611 / 0.805 ms mean/max per core per view):
`build_mat_worklist` deals mat items largest-first (LPT), so every mover slot
starts by sorting its biggest tile while TRISC1 waits.

Change (off by default): `GSPLAT_TT_MAT_RAMP=n` moves each mover slot's n
smallest items to the front, ascending (`render/host/sort_mover_split.h`
`ramp_front`). Only the order changes, so output is bit-exact by construction;
unit test `tests/unit/test_sort_tail_dual_mover.cpp` (60 random scenes, 0 failures,
run on the box: `out-ab/unit.txt`).

## A/B (p100a yyzo-bh-04, 11x10 grid, untraced, 3 alternating rounds x 30 views, commit a6457506)

| arm | ms/view (r1 r2 r3) | mean | blend | sort | project | xview | md5 |
|---|---|---|---|---|---|---|---|
| ramp 0 (default) | 8.765 8.742 8.715 | 8.741 | 6.983 | 0.532 | 1.099 | 0.065 | 906e0435 x3 |
| ramp 1 | 8.716 8.709 8.707 | 8.711 | 7.027 | 0.483 | 1.091 | 0.055 | 906e0435 x3 |
| ramp 3 | 8.722 8.716 8.728 | 8.722 | 7.038 | 0.481 | 1.090 | 0.056 | 906e0435 x3 |

Best arm (ramp 1) is -0.030 ms/view (-0.34%), below the 1% (0.09 ms) keep bar.
Sort time drops ~0.05 ms, but blend grows ~0.045 ms: the mat jobs start
earlier, yet the frame end barely moves, so the critical path after the start
gap is the movers' total work (between-job gaps), not the first item.
No Tracy confirmation run was made (gate not met). Default stays ramp 0.

Raw outputs: `out-ab/` (summary.txt, per-round logs, md5 lists).
