# Task #298 — lever B: fold K2 into sort_ol — SHELVED

Device A/B, yyzo-bh-07 p100a, commit aaf3342, 30-view bicycle sweep, untraced,
4 alternating rounds per arm (driver `drive.sh`, logs in `out/`).
Fold engaged in every fold run (`[TA] K2 folded into sort=1 (diff words 1024)`).

| round | off (GSPLAT_TT_K2_FOLDED=0) ms/view | fold ms/view |
|---|---|---|
| r1 | 11.174 | 11.301 |
| r2 | 11.138 | 11.282 |
| r3 | 11.118 | 11.411 |
| r4 | 11.129 | 11.314 |
| mean | **11.140** | **11.327** (+0.187) |

Gate was -0.3 ms/view; the fold is 0.19 ms/view **slower**.
Correctness: sweep md5 906e0435 (golden) in all 8 runs, 30/30 views, every
per-view png md5 identical across all 8 runs.

Host stage split (mean of 4 runs):

| stage | off | fold | delta |
|---|---|---|---|
| project (pfwc + count/K2 wait) | 3.061 | 2.669 | -0.39 (K2 pair writes gone; count-only K2 left) |
| sort (host side) | 0.857 | 0.856 | 0 |
| blend wait (sort_ol + fused mat+blend) | 6.942 | 7.517 | +0.57 (pair walk + emit inside the sort movers) |

Why: the count-only K2 saves ~0.39 ms, close to the #291 estimate of the K2
mover time (0.63 ms max on the slowest core, part of which overlaps pfwc).
But the rectangle walk that makes the pairs now runs on the sort/mat movers,
which are the critical path (the #297 profile shows mat is mover-bound,
NCRISC 2.48 / BRISC 2.38 ms busy), and it costs ~0.57 ms there. The K2
pair-writing was cheaper on its own cores than the same walk on the movers.
No Tracy capture was taken (hang risk noted in #291; the host split already
locates the loss), so per-zone count/walk/emit times on device are not split
further than the table above.

Disposition: code kept behind `GSPLAT_TT_K2_FOLDED=1` (default off). The
off path is the measured off arm (same binary, env default flipped).
Unit test tests/unit/test_pfwc_fuse.cpp (diff rows == per-pair counts, window
walk == emit_pairs) stays.

Idea for later (not queued here): the walk only pays off if it runs on idle
cores (cull TRISCs idle ~2.4 ms in mat, #297) rather than the movers; that is
lever A's territory.
