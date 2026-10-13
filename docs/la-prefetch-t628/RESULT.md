# t628 L-A prefetch — result: rejected

Lever: with `GSPLAT_TT_LA_PREFETCH=1`, the fold after-hook enqueues view N+2's early sort and
proj_M read right behind mat+blend(N+1) (CQ1, behind the K2-done event). Default off. Commit 7cbf73fc.

Measured on bh-33 (p150), 2026-10-13 ~11:28 PDT, render/run.py --back-to-back, 30 bicycle views,
20 passes, interleaved rounds (r0 warm-up pair + r1, r2):

| round | base ms/view | LA ms/view |
|---|---|---|
| r0 | 8.463 | 8.456 |
| r1 | 8.465 | 8.452 |
| r2 | 8.445 | 8.460 |
| mean | 8.458 | 8.456 |

Delta: -0.002 ms/view (noise; bar was >= 0.10). md5 sweep with LA on: 077caaba 30/30
(golden match, PSNR 42.85 dB vs benchmarks/reference_v2/hero.png).

Step 1 phase split (phases.py, median over 600 views, HTL on):

| run | A | A1 | B | C | D | G |
|---|---|---|---|---|---|---|
| r1 base | 0.261 | 0.136 | 1.173 | 0.143 | 6.916 | 0.055 |
| r1 LA   | 0.261 | 0.140 | 1.143 | 0.141 | 6.948 | 0.054 |
| r2 base | 0.263 | 0.143 | 1.143 | 0.144 | 6.955 | 0.055 |
| r2 LA   | 0.264 | 0.139 | 1.173 | 0.142 | 6.923 | 0.055 |

Phase A (view start to gather result) does not move with the prefetch: it is device time on the
in-order queue, not host dispatch idle. The host sits blocked in D (6.9 of 8.5 ms) waiting for
blend(N+1), so the b2b view boundary is device-bound and an earlier enqueue cannot hide anything.
This agrees with t614's <=0.30 ms bound and shows the dispatch-idle part of it is below 0.10 ms.
Decision: kill. Code stays on this branch, default off; not landed.
