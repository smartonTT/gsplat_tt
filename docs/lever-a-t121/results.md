# Lever A (task #121): one-launch sort v2, device A/B and default

Board: yyzo-bh-07 (Blackhole p100a, 110 cores), not a p150. Bicycle, 30 views,
1024x1024, untraced unless marked Tracy. md5 reference: md5-r82new.txt.

Code under test: the one-launch sort v2 from #124 / #137 / #138 (2e30c29,
3a3fa83, 68aa90b: PB/RING/PUBOC in sort_bin_onelaunch.cpp, big-tile select
items in sort_subchunk_materialize.cpp, duplicate CB 10/11 fix). This task ran
the device A/B (0f0c365) and made it the default (090fa37, 4e82912).

## Result

| arm (rev) | ms/view | FPS |
|---|---|---|
| legacy multi-launch sort, old tip (0f0c365, 3 rounds) | 29.55 | 33.8 |
| one-launch v2, select on (0f0c365, 3 rounds) | 24.60 | 40.7 |
| default after flip, select on (090fa37, 3 rounds) | 24.69 | 40.5 |
| select off (090fa37, 3 rounds) | 24.55 | 40.7 |
| legacy kill switch `GSPLAT_TT_SORT_ONELAUNCH=0` (090fa37, 3 rounds) | 29.64 | 33.7 |
| **final default, select off (4e82912, 3 back-to-back runs)** | **24.54** | **40.7** |

- Gate passed: -4.95 ms/view (-16.8%) vs the 29.5 ms tip, against a 3 ms bar.
- All 26 runs of 30 views were md5-identical to the reference (hero_vs_ref
  100 dB). `ONELAUNCH_CHECK bad_tiles=0` on every frame checked. No hang in 3
  back-to-back runs, twice (select on: 24.69 / 24.61 / 24.76; final default:
  24.48 / 24.62 / 24.52). A watcher run (TT_METAL_WATCHER=2, 3 views) was clean.
- Defaults now: `GSPLAT_TT_SORT_ONELAUNCH` on (`=0` is the kill switch back to
  the legacy sort); `GSPLAT_TT_OL_MAT_SELECT` 0.

## L1 check (before the run)

- Usable CB space on the remote's tt-metal (e77780fe): MEM_L1_SIZE 1,572,864 B
  minus the unreserved base 111,104 B (MEM_MAP_END 40,448 + 69 KiB kernel
  config) = 1,461,760 B.
- CB total at PB=8 / RING=8 / WIN=1024: 504,896 B per mover x 2 + 22,400 B
  shared = 1,032,192 B. Fits, with 430 KB to spare. No L1 buffers are allocated
  (all buffers are DRAM).
- On the device the program was created and printed
  `[SORT] ONELAUNCH v2 OL_PB=8 OL_RING=8 OL_WIN_PAGES=1024 ... cb_bytes/mover=504896 shared=22400`.
  No fallback was needed.

## A/B rounds (0f0c365, order rotated)

| round | legacy | one-launch v2 |
|---|---|---|
| r1 (base, on) | 29.581 | 24.564 |
| r2 (on, base) | 29.599 | 24.560 |
| r3 (base, on) | 29.466 | 24.667 |
| mean | 29.549 | 24.597 |

Stage view (untraced, ms/view): sort 10.68 -> 5.10, blend 10.69 -> 11.37
(the materialize runs inside the blend stage and now sorts every tile),
project 6.47 and tile_assign 1.44 unchanged.

Single runs on the same build: v1 one-launch (`OL_PB=1 OL_RING=0 OL_MAT_SELECT=0`)
28.20; v1 emit with select 28.48; v2 emit without select 24.47.

## Confirm of the default (090fa37, order rotated)

| round | default (select on) | select off | legacy kill switch |
|---|---|---|---|
| d1 | 24.655 | 24.601 | 29.766 |
| d2 | 24.643 | 24.508 | 29.468 |
| d3 | 24.781 | 24.535 | 29.678 |
| mean | 24.693 | 24.548 | 29.637 |

Select off was faster in every round (-0.05, -0.14, -0.25 ms) and in the A/B's
single run (-0.13 ms). The blend stage, which holds the materialize, was lower
in every round too (11.26-11.31 vs 11.36-11.43). So select is now off by default.
The select code stays behind `GSPLAT_TT_OL_MAT_SELECT=1`.

## Tracy (10 views, traced, ms per view)

| program window | legacy t115 (a03bd1e) | v2 select on | v2 select off (default) |
|---|---|---|---|
| pfwc | 2.46 | 2.46 | 2.46 |
| proj scatter | 3.76 | 3.70 | 3.71 |
| idle before TA | 0.42 | 0.83 | 0.79 |
| tile_assign | 1.42 | 1.40 | 1.40 |
| sort (legacy: hist + host bridge + emit + radix + publish) | 10.51 | 4.67 | 4.67 |
| idle before materialize | 0.08 | 0.32 | 0.33 |
| materialize + cull | 3.17 | 3.94 | 3.77 |
| blend | 7.52 | 7.63 | 7.53 |
| device span | 29.45 | 25.10 | 24.82 |
| all-core idle | 2.00 | 1.29 | 1.27 |

- Emit: sort_ol_emit busiest core 4.04 ms per view (v1 one-launch 8.26;
  legacy emit 4.51 traced / 5.29 host-timed). Well under the 6 ms stop line.
  sort_ol_count 1.03 ms, barrier wait 0.50 ms, prefix 0.05 ms.
- The 0.33 ms idle before the materialize is host work: the totals read-back,
  layout, LPT, worklist and their uploads (layout 0.10 + pub_host 0.15 ms
  host-timed).
- The idle before TA (0.79 vs 0.42 ms) shows only under Tracy; untraced project
  and tile_assign times did not change.

Files: out/t121-ol-*.txt (select on), out/t121-mprof-*.txt (select on, fine
mat zones), out/t121-nosel-*.txt and out/t121-nosel-mprof-*.txt (default).
Captures stay on the remote under /localdev/smarton/gstt2-t121/opt/profiler/.

## Materialize items (GSPLAT_TT_MAT_STATS=1, fine zones GSPLAT_TT_MATCULL_PROF=1)

Cost model (host LPT units, 3 frames):

| | items | mean slot | max item | max item / mean slot | max NCRISC | max BRISC |
|---|---|---|---|---|---|---|
| select on | 1052-1066 | 12.3-12.7k | 15.9-16.4k | 1.27-1.29 | 16.7-18.2k | 10.5-10.7k |
| select off | 1037-1044 | 12.7-13.3k | 31.2-32.9k | 2.34-2.55 | 31.2-32.9k | 10.4-10.5k |

With select on, no item passes 1.5x the mean slot. With select off, the model
says one v1 big-tile item is 2.5x the mean slot. But the measured time says
otherwise: the busiest mover never runs a big-tile item (below). The v1 item
cost (`cnt + l_sub`) overestimates these items.

Measured per item (mat_items.py on the device CSV, 11 frames):

| kind | items/frame | mean us | max us | main zones (mean us) |
|---|---|---|---|---|
| select part (select on) | 60 | 1736 | 2224 | sort 1023, keys 312, cull 213, gather 183 |
| v1 big-tile item (select off) | 35 | 2529 | 3329 | sort 1514, cull 350, keys 303, gather 300 |
| whole-tile item | 1012 | 549 | 3307 | sort 280, perm 248, cull 167, rd 17 |

Busiest mover per frame (select off): 3.0-5.0 ms against a mean of 2.3-3.5 ms.
It is always an NCRISC with only whole-tile items. NCRISC averages 0.2-0.5 ms
above BRISC. For select on it is the same pattern (busiest 3.3-5.0 ms, an NCRISC,
no select parts).

### #132's note (reads in OL_MAT_SELECT)

Per select part, keys (312 us) + gather (183 us) = 495 us of reads, against
1023 us in select_ranks. Reads do not dominate. A gather that fetches only the
pages its candidates touch would not help either: a part's 4096 ranks are
spread over a ~20k-record tile, so nearly every 2 KB page is touched. And the
select parts are not on the critical path at all. Not done.

## What sets the materialize time now

The materialize program (3.77 ms) is set by the per-RISC load imbalance of
whole-tile items: the busiest NCRISC carries 1.2-1.4x the mean. The LPT costs
(`cnt` per whole-tile item, `cnt + l_sub` per big item, same for both RISCs)
do not match the measured times: per-item fixed cost, the NCRISC-only items
above 6144 records and the overestimated big items. Calibrating the costs from
these zones (per kind, per RISC, with a per-item constant) could bring the
busiest mover near the mean: up to ~0.9 ms per view.

## Follow-ups

1. Calibrate the materialize LPT costs (per kind, per RISC, fixed per-item
   cost) from mat_items.py data and re-balance. Up to ~0.9 ms/view.
2. Move the materialize worklist to the device (or build it from the count
   pass) so no totals read-back and host layout sit between the sort and the
   materialize: ~0.3 ms/view of idle.
3. Rework or drop OL_MAT_SELECT: select_ranks costs ~1.0 ms per part; one
   shared tile sort that all parts read would remove the repeat work.
4. Re-rank the deferred levers (#134) against the new 24.54 ms tip.
