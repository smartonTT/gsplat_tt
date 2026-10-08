# t394: cross-view overlap (t379) confirmed on the p150 (bh-30)

Board: bh-30 (p150, tt_aus), under the existing viewer reservation (viewer exception,
user 2026-10-07). The viewer was stopped from 00:53:49Z to 00:57:13Z on 2026-10-08 and
restarted right after. Check: localhost:8091 page 200, websocket 101, selftest 11.55 ms.
Code: t388 tree (2138a42c) plus the bench scripts (1993cce7, 0d02b982). Built with nice -n 19,
ionice -c3 and -j12 in /localdev/smarton/p150bench/tree394, not in the viewer dir.

Method: 3 rotating rounds x 30 bicycle views at 1024x1024, the same views as #388 on the p100a.

## ms/view (avg_frame_ms)

| arm | r1 | r2 | r3 | mean | vs base |
|---|---|---|---|---|---|
| base | 10.966 | 11.067 | 11.048 | 11.027 | — |
| xv (`GSPLAT_TT_XVIEW_OVERLAP=1`) | 9.578 | 9.561 | 9.569 | 9.569 | -13.2% |
| xvpin (xv + `GSPLAT_TT_OUT_PINNED=1`) | 9.224 | 9.156 | 9.147 | **9.176** | **-16.8%** |

For comparison, the p100a (#388) gave base 10.940, xv 9.169 and xvpin 8.934 (-18.3%).
The published GPU reference for this scene and resolution is 10.75 ms/view. That number is
published, not measured. xvpin on the p150 comes in 1.57 ms below it.

## Stage breakdown (ms, mean of 3 rounds)

| arm | project | sort | blend | d2h | xview |
|---|---|---|---|---|---|
| base | 2.975 | 1.054 | 6.490 | 0.428 | 0.000 |
| xv | 1.157 | 1.092 | 6.697 | 0.384 | 0.138 |
| xvpin | 1.164 | 1.043 | 6.436 | 0.292 | 0.131 |

As on the p100a, the gain comes from project (2.98 -> 1.16 ms). The pfwc gather wait is
hidden because view N+1's pfwc runs under view N's tail. Plain xv adds about 0.2 ms to
blend. Pinned output removes that and also trims d2h by about 0.1 ms.

## Correctness

- md5: all 9 runs match golden `906e0435`, 30/30 views each. Every run printed `ALL_VIEWS_IDENTICAL`.
- XVIEW_HITS_OK 29/30 (hits=29, misses=0) in all 6 xv/xvpin runs.
- Hero screenshot (device, xvpin, r1): `out/hero.png` (md5 86524912). Against
  `benchmarks/reference_v2/hero.png`: **PSNR 42.51 dB**, max abs diff 46, 32x32-tile mean
  diff median 0.95 and max 6.65. Diff x10: `out/hero_diff10.png`.
- Visual check (by eye): the hero shows no tile seams, block edges or missing tiles. The diff
  shows only fine edge detail (spokes, foliage, bench slats), with no 32-px grid pattern.

## Files

`out/run-r*-*.log`, `out/md5-r*-*.txt`, `out/summary.txt`, `out/bench394.log`,
`out/hero-r{1,2,3}-xvpin.png` (all identical), `out/hero.png`, `out/hero_diff10.png`.
