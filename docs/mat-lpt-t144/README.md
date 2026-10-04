# t144: materialize LPT cost calibration (per kind, per RISC)

Board: yyzo-bh-07 p100a (IRD), bicycle, 30 views, base = smarton/tt-project-opt
4b6c092 (iter-180 defaults). Remote tree /localdev/smarton/gstt2-t142.

## Fit (MPROF capture, views 0:10, 11 frames, 11,521 items)

`GSPLAT_TT_MAT_DUMP` (new) writes the host worklist per call; `mat_fit.py`
matches it item by item to the `mat_*` zones (core x/y rank -> logical core,
0 of 2,420 mover-frames mismatched). Item time in us, n = tile records:

| kind | RISC | n | fit | rms |
|---|---|---|---|---|
| whole, n <= 6144 | NCRISC | 508 | 5.5 + 0.197 n | 4 |
| whole | BRISC | 9,219 | 16.1 + 0.187 n | 24 |
| whole, n > 6144 | NCRISC only | 1,410 | -282 + 0.2226 n | 109 |
| big-tile subchunk | NCRISC only | 384 | 0.1083 n + 0.1267 l_sub | 351 |

Fixed per-item cost is small (5-16 us); both RISCs cost ~0.19 us/record. The
legacy LPT priced a big item at n + l_sub, ~1.7x a whole tile of the same
records; measured, it is ~0.6x. NCRISC slots holding one were underloaded and
the rest carried the NCRISC-only work (1.2-1.4x the mean).

## Change

`render/host/sort_mover_split.h`: `MatCostModel` (9 coefficients above, default
on; `GSPLAT_TT_MAT_COST=0` = record counts, or a 9-value override), used for the
one-launch select-off worklist. The LPT picks the eligible slot that finishes
the item first (identical to least-loaded when both RISCs cost the same).
Output is unchanged: items are independent (md5-identical below).

Replay on the captured item times (`replay.py`, us, busiest mover, MPROF
timing): legacy 3,972 / calibrated 3,646 / LPT on measured times 3,479 /
lower bound 3,277 (max item or NCRISC-only work / 110). A busiest-slot
move/swap search after LPT found no improving move: the busiest slot is set by
one item (frames 2-7) or by NCRISC-only load (frames 0, 1, 8-10).

## Device A/B (untraced, 30 views, md5 vs ref)

| round | order | legacy frame | calibrated frame | diff | legacy blend | calibrated blend |
|---|---|---|---|---|---|---|
| r1 (MAT_STATS on) | leg, cal | 19.798 | 19.461 | -0.337 | 9.675 | 9.488 |
| r2 | cal, leg | 19.661 | 19.595 | -0.066 | 9.697 | 9.507 |
| r3 | leg, cal | 19.789 | 19.508 | -0.281 | 9.717 | 9.521 |
| r4 | cal, leg | 19.667 | 19.523 | -0.144 | 9.713 | 9.515 |

All 8 runs ALL_VIEWS_IDENTICAL (30 views). The blend stage (which holds the
materialize) is 0.19-0.20 ms lower in every round; avg frame -0.21 ms mean
(-0.16 over the three clean rounds r2-r4).

## Verdict

Gate (>= 0.3 ms/view) not met: ~0.2 ms/view (1.0%). Not landed. What is
left is set by the largest single item (a 14-16k-record whole tile or a big
subchunk, ~3.0-3.3 ms traced) and by NCRISC-only work (whole tiles above
BRISC's 6144-record buffer). Neither moves with the LPT; both need kernel
work (split a large whole tile across the core's two movers, or a bigger
BRISC buffer).

Files: out/t144-fit.txt, out/t144-replay.txt, out/t144-cap-*.txt (traced
capture), out/t144-r1.log, out/t144-r234.log.
