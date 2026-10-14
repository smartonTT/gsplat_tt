# B1' (#722) device A/B on bh-49 (p150a), 2026-10-14

B1' = heavy tiles (records > total / num_cores) blended as units of their parent tile plus a
microblock mask, behind `GSPLAT_TT_B1` (0 = today's kernels, 1 = 4 units, 2 = 4 or 16 units).
It splits only the blend work; the emit and sort still bucket per 32x32 tile (no own sub-tile lists).

## b2b, cameras_train_nn (169 views, 6 passes), default BLEND_T_PERIOD, commit 80c35b73

| config | B1=0 ms/view | B1=1 | B1=2 | raw_md5 (all arms) |
|---|---|---|---|---|
| s4 0.5M | 4.370 | 3.211 | 3.074 | c775b600 |
| s4 1M | 6.531 | 4.955 | 4.786 | b35dad8b |
| s4 1.5M | fails | fails | fails | sort bucket overflow |
| s4 3M | fails | fails | fails | sort bucket overflow |
| s2 1M | 3.750 | 3.255 | 3.230 | 6e5b5fa0 |
| s2 3M | 7.359 | 6.389 | 6.329 | 16ba349e |
| s2 6.13M | fails | fails | fails | sort bucket overflow |
| s1 6.13M | 8.023 | 7.924 | 7.928 | fba4b01a |

"fails": every arm, baseline included: a 32x32 tile gets more than 65472 records; the overflow
retry raises the contrib floor until the run gives up. B1' does not change emit/sort, so it cannot
fix this. raw_md5 is equal across arms at the default period too (bit-exact output).

Pass bar (1M s4 <= 2.0 ms/view, 3M s4 runs): not met. Best 1M s4 is 4.79 ms/view (-27%).

## Hero (s1, cameras_v2 hero view, default period)

All three arms are byte-identical (max u8 diff 0). Against benchmarks/reference_v2/hero.png:
PSNR 42.85 dB, max u8 diff 45, mean 0.835. Diff image (x8): out/try2/logs/hero-b2-diff-x8.png.
Visual check: no tile seams or block artifacts; the diff follows spokes and fine edges only.

## md5 at BLEND_T_PERIOD=0 and JIT check

First try: every arm failed to build (`alpha_blend_compute_mb.cpp:1602: division by zero`,
-Werror, present since 0066e274). Fixed in b889a6af; rerun results below.
