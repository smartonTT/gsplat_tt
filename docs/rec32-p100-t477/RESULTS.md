# t477: p100 check of the REC32 default (best-iter-221, b6de9ccd)

Machine: p100 yyzo-bh-04 (existing measurement reservation), 2026-10-10 07:12-07:18 PDT.
Arms: `def` = defaults (GSPLAT_TT_PFWC_REC32 on), `off` = GSPLAT_TT_PFWC_REC32=0.
Alternating order, r0 = warm-up (not counted).

## Back-to-back ms/view (headline; run.py --back-to-back, 30 views, 3 passes, median)

| round | def   | off   |
|-------|-------|-------|
| r0 (warm-up) | 9.697 | 9.832 |
| r1    | 9.674 | 9.839 |
| r2    | 9.665 | 9.833 |
| r3    | 9.668 | 9.826 |
| **median r1-r3** | **9.668** | **9.833** |

REC32 on: -0.165 ms/view (-1.7%), 103.4 vs 101.7 FPS. raw_md5 5a438f5a in all 8 b2b runs,
identical across passes in every run.

## Latency (secondary; no --dump-views), avg / p50 ms

| round | def | off |
|-------|-----|-----|
| r1 | 9.6 / 9.7 | 9.8 / 10.0 |
| r2 | 9.6 / 9.7 | 9.8 / 10.0 |
| r3 | 9.6 / 9.8 | 9.8 / 9.9 |

## Correctness

--dump-views pass per arm: `MD5_GOLDEN_OK grid 11x10: list 906e0435 = golden (30/30 views)` for
both def and off. No fix needed; the default stays on.
