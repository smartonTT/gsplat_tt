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
