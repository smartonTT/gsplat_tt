# Task #418: mover sort / perm in sort_subchunk_mat

#413 showed mat is mover-bound: TRISC1 idles 0.469 ms mean / 0.904 ms max per core per view
between jobs while the movers sort (NCRISC 52%, BRISC 35%) and permute (42% / 34%) the next
subchunk. This task sizes the two and cuts the larger one.

## Step 1: sizing (no device; #413's capture, 30 views x 121 cores, p100a yyzo-bh-04)

From the movers' `fz_mv_*` cycle counters (`docs/mat-job-zones-t413/out/dev-c*.csv.gz`, summed
over all core-views, divided by `fz_mv_recs`; 1350 MHz):

| mover | recs / core-view | jobs / core-view | sort cyc/rec | perm cyc/rec | rd cyc/rec | wr cyc/rec | big-tile cyc/rec |
|---|---|---|---|---|---|---|---|
| BRISC  | 12,112 | 6.3 | **151.5** | 40.2 | 5.4 | 1.6 | 0 |
| NCRISC | 12,100 | 3.7 | **141.1** | 41.7 | 11.5 | 1.5 | 20.9 |

Per core per view (ms, mean / max over cores): sort 1.254 / 1.684 (NCRISC), 1.347 / 1.431
(BRISC); perm 0.370 / 0.465, 0.358 / 0.432. `fz_mv_sort` counts only the whole-tile item
sort (`sort_radix_tile::sort_record_ids`); big tiles (NCRISC) are in `fz_mv_big`.

Per-subchunk `MAT_PZ` zones (views 0-9):

| mover, zone | zones | mean us | p50 | p90 | max |
|---|---|---|---|---|---|
| BRISC mat_ol_sort  | 7,641 | 214.5 | 185.4 | 438.4 | 567.2 |
| BRISC mat_ol_perm  | 7,641 |  56.7 |  47.9 | 124.0 | 186.1 |
| NCRISC mat_ol_sort | 3,701 | 460.1 | 179.9 | 1281.5 | 1934.2 |
| NCRISC mat_ol_perm | 4,351 | 101.9 |  49.7 | 279.9 | 322.9 |

So the sort is 3.5x the perm per record and about 78% of each mover's sort+perm time. The
perm is an 8-word copy per record (about 2.5 cycles per L1 access); the sort is the lever.

Where the sort's ~145 cycles per record go (`sort_record_ids` -> `sort_pairs`): per-tile
depth keys are float bits with a range of about 2^20-2^26, so the plan is mostly P = 3
passes of 9-bit digits (HIST_ENTRIES 1536). Per element that is about 18 L1 accesses: key
gather (load + key and id stores), a min/max scan, a histogram scan, three scatter passes
that each move a key and an id (load 2, store 2; the last stores the id only), and for odd
P a copy of the ids back into v; plus 6 local-memory histogram updates.

## Step 2: packed key|id radix (`GSPLAT_TT_SORT_PACKED`, default on)

`sort_radix_tile_algo.h` `sort_record_ids` / `sort_ids_packed`: same stable LSD radix and
same plan (so the same permutation), but

- the key gather also finds kmin / kmax (no separate scan) and writes no id array;
- pass 0 takes the id from the loop index and writes one packed word
  `((r >> d) << ib) | id` per element (`ib = bitlen(n - 1)`), whenever the remaining key
  bits fit next to the id (`B - d + ib <= 32`; else the old pair sort runs);
- the middle passes move one word, and the last pass writes the id straight into v (no
  odd-pass copy back).

That is about 9 L1 accesses per element instead of 18. `GSPLAT_TT_SORT_PACKED=0` gives the
old pair sort for the A/B. Unit test: `tests/unit/test_sort_radix_tile.cpp` (both settings;
every key width 1..32 for n up to 32768, so the packed path and its fallback both run).

## Step 3: A/B (p100a yyzo-bh-04, measurement reservation IRD job 244892, 11x10)

One build (bb1c9bdd, build #125), bicycle 30 views 1024x1024, untraced, 3 alternating
rounds (`drive.sh` MODE=rounds, one devrun per arm). `xvpk` = defaults (packed sort),
`xvpair` = `GSPLAT_TT_SORT_PACKED=0`. Logs: `out/run-r*-*.log`, `out/round*-*.out`.

| round | xvpk view_total | xvpair view_total | xvpk frame | xvpair frame | xvpk blend | xvpair blend |
|---|---|---|---|---|---|---|
| 1 | 8.223 | 8.673 | 8.263 | 8.711 | 6.573 | 7.028 |
| 2 | 8.220 | 8.676 | 8.259 | 8.712 | 6.561 | 7.034 |
| 3 | 8.248 | 8.708 | 8.297 | 8.756 | 6.524 | 6.995 |
| **mean** | **8.230** | **8.686** | **8.273** | **8.726** | 6.553 | 7.019 |

**-0.456 ms/view (-5.2%)**, 5x the 0.09 ms gate; every round wins by 0.45+ ms. The gain is
all in the fused mat+blend stage (blend 7.019 -> 6.553). md5 906e0435 on 30/30 views in all
6 runs (`out/md5-r*.txt`, per-view md5s identical across arms): bit-exact.

## Step 4: Tracy recapture (`GSPLAT_TT_MATCULL_PROF=1`, same build)

pk = packed, 30 views (c0/c10/c20); pair = `GSPLAT_TT_SORT_PACKED=0` control, views 0-9
(c0). Stitched with `opt/profiler/stitch_device_csv.py`, analysed with
`docs/xvpin-tracy/ana407.py` (section (c); `out/ana418-*.txt`, `out/job-gaps-per-core-*.csv`).
The script stops at the host reconciliation (no untraced rounds in its `--out`), after (c).

Mover cycles per record (`cyc_per_rec.py`, sum of `fz_mv_*` / `fz_mv_recs`):

| mover | sort pair | sort packed | perm pair | perm packed |
|---|---|---|---|---|
| BRISC  | 152.2 | **95.2** (-37%) | 40.2 | 38.7 |
| NCRISC | 140.9 | **88.2** (-37%) | 41.2 | 42.4 |

TRISC1 idle per core per view (ms; mean over views of mean / max over cores):

| | pair (v0-9) | packed (v0-9) | packed (30 v) | #413 (30 v) |
|---|---|---|---|---|
| mat length | 2.284 / 2.835 | **1.831** / 2.539 | 1.842 / 2.531 | 2.299 / 2.839 |
| start gap | 0.596 / 0.788 | **0.425** / 0.575 | 0.435 / 0.578 | 0.611 / 0.805 |
| between-job gaps | 0.477 / 0.919 | **0.335** / 1.058 | 0.326 / 1.040 | 0.469 / 0.904 |
| idle outside jobs | 1.078 / 1.663 | **0.765** / 1.633 | 0.766 / 1.617 | 1.084 / 1.651 |

Both gaps shrank by ~30% on the mean core. The worst core's between-job sum did not
(0.92 -> 1.06): on that core NCRISC's big-tile sort (`fz_mv_big`, not touched here) and the
perm now dominate. During the remaining between-job gaps the movers are in perm 50% / 49%
and sort 42% (NCRISC) / 10% (BRISC), so the next lever is the perm (an 8-word copy per
record) and NCRISC's big-tile sort.

## Device hero (iteration 215)

`opt/metal-screenshots/ttw-215/hero.png` and `hero_diff10.png` (defaults, bb1c9bdd, p100a):
PSNR 42.51 dB vs `benchmarks/reference_v2/hero.png`, golden match, max_lsb 0, pixel-identical
to the iteration-213 hero (md5 86524912). I looked at both images: no tile seams, blocky or
empty tiles; the diff is only thin edge detail (spokes, frame, bench slats, foliage).
Tile-boundary pixel columns mean |err| 0.873 vs interior 0.875 (rows 0.862 vs 0.876).
