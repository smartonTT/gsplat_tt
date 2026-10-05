# Task #162: review + default flip of the pixel-centre pre-cull (GSPLAT_TT_PRECULL=2)

Board: yyzo-bh-07 (Blackhole p100a, not a p150). Bicycle, 30 views, 1024x1024, untraced.
Tree: tip 220ba33 (includes t160 fast emit, iter-184) + 1aa999d (default 2).

## Code review: pass
precull_axis PRECULL_PC (r' = max(sqrt(t cov) - 3/8, 1/8)) is superset-safe: the blend
and the band cull sample pixel centres at c + 0.5, so the rect keeps every tile whose
pixel centres the band cull can reach, plus 1/8 px. The on-screen tests (fl(m + r) <= 0,
W - fl(m - r) <= 0) and the tile-cell floors stay correct with a fractional r (1 rounding,
fp32 ulp at 1024 px << 1/8). RECHECK lanes keep the 3-sigma radii. Kill switches 0/1 intact.
tests/unit/test_precull.cpp: PASS (Mac).

## Confirm A/B: FAIL on the current tip (out/t162-ab.log)
| arm | r1 | r2 | r3 | mean | sort | blend |
|---|---|---|---|---|---|---|
| old (PRECULL=1) | 18.566 | 18.535 | 18.514 | 18.538 | 4.387 | 9.287 |
| new default (2) | 18.765 | 18.811 | 18.827 | 18.801 | 5.096 | 8.883 |
Paired new - old: +0.199 / +0.276 / +0.313, mean **+0.263 ms/view (slower)**.
Old arm md5 = golden md5-r82new.txt in all rounds; new arm deterministic over rounds
(md5 list 46a725ab...), <= 1 LSB vs old, PSNR 74.24-77.81 dB.

## Diagnostic: t160 fast emit regresses under PRECULL=2 (out/t162-diag-emitfast0.log)
With GSPLAT_TT_OL_EMIT_FAST=0 (both arms, 2 runs each): old 19.164 / 19.231, pc 18.304 / 18.381
ms/view -> pc - old = -0.86 (as in t161). Sort: fast on: old 4.39, pc 5.10; fast off: old
5.04, pc 4.65. So the fast emit loop saves ~0.65 ms with PRECULL=1 but costs ~0.45 ms with
PRECULL=2. Not investigated further (review scope). Candidates: the fast-path gate
(ring_on needs tile_cap % REC_PAGE_RECS == 0, sort_bin_onelaunch.cpp:366) or the
sub_int fast path vs cold fallback rate. Default not flipped; goldens not refreshed.
