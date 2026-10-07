# t362: read the K2 rows on CQ1 right behind proj_M (GSPLAT_TT_K2_ROWS_EARLY)

Idea (#356, docs/p150-gap.md): mat waits on host-side per-tile totals that arrive over the
CQ1 bridge. Issue the K2 rows 4 KB-view read on CQ1 right after the proj_M read, so it runs
when the K2 ends and the sort only waits on its event.

Result: **no gain. Default set to off** (flag kept; `=1` enables it). Code at b00ce011 (default on),
same build for both arms, alternating rounds, md5 906e0435 on every run (30/30 identical views).

| board | arm | r1 | r2 | r3 | mean ms/view |
|---|---|---|---|---|---|
| p150 bh-30 | early (on) | 12.774 (cold, warmup 21.7 s) | 12.638 | 12.658 | 12.648 (r2-r3) |
| p150 bh-30 | off | 12.646 | 12.653 | 12.643 | 12.647 |
| p100a yyzo-bh-04 | early (on) | 10.932 | 10.926 | 11.049 | 10.969 |
| p100a yyzo-bh-04 | off | 10.875 | 10.899 | 10.924 | 10.899 |

Why: the host-side sort stage does shrink (p100a sort 0.42-0.44 vs 0.48-0.59 ms, bin_emit
0.14-0.16 vs 0.19-0.23 ms; p150 sort 0.89-0.97 vs 1.00-1.12 ms), but blend wait grows by the
same amount (p100a blend 7.16-7.18 vs 7.00-7.11; p150 8.10-8.19 vs 7.93-8.05). The frame is
device-bound; mat's wait for the K2 totals is not on the critical path, so reducing the totals
on device instead would not help either. The p150-vs-p100a gap now sits in blend
(~8.1 vs ~7.1 ms), which is device time.

ONELAUNCH_CHECK (p100a, early on): bad_tiles=0. Device hero (p100a, b00ce011, defaults =
early on): opt/metal-screenshots/t362-k2early/hero.png, diff hero_diff10.png, 42.51 dB vs
benchmarks/reference_v2/hero.png, golden match, no tile seams seen by eye.

p150 bench window on bh-30 (viewer stopped only for it, existing viewer reservation, no
reserve/extend/release): viewer stopped 22:42:46Z, restarted 22:45:25Z (2 min 39 s), sha 4307f076.

Raw logs: out/ (p100a round*.out, p150/run-*.log).
