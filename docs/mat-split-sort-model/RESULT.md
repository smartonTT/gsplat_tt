# t175: split the big-tile key sort across two movers (model only, no device)

**Verdict: SHELVE.** Splitting the sort of big tiles (> 16384 records) into two half-sorts plus a
merge cannot save 0.3 ms/view. The model's best case is -0.18 ms (hero view) and -0.10 ms (view 1)
traced, and it lands exactly on the bound where the big-tile sort costs nothing. Untraced, that is
about -0.05 to -0.1 ms. The window is pinned by the largest **whole** tile (8192 < n <= 16384
records, one item on one NCRISC, 3.2-3.3 ms), not by the big-tile chain.

## Inputs

- Per-tile record counts for views 0 (hero) and 1: `docs/fuse-matblend-t147/out/t147-tc.dprint`.
  These are the only per-tile counts on disk, so the model covers two views, not 30.
- Big-tile costs from the t168 Tracy zones (branch `ttp/t168-shrink-the-largest-materialize-item-big-`,
  9202610, `docs/mat-shared-t168/out/t168-onp|offp/zones.txt`, yyzo-bh-07 p100a, 30 views).
  The mean big tile is N = 19,238 records (views 0 and 1).

| term | Tracy | model, us |
|---|---|---|
| key read | mat_ol_keys 62.0 ms / 211 items | 0.01528 per tile record |
| radix sort | mat_ol_sort: 22.5 ms/view saved by 14.1 fewer sorts/view (off -> on) = 1.595 ms | 0.0829 per tile record |
| publish ids | mat_ol_pub 33.0 ms / 211 | 0.0081 per word |
| gather (re-reads the whole tile) | mat_ol_gather 180.3 ms / 634 | 0.01476 per tile record |
| cull + write, own records | 590.5 ms/view mover total minus the non-nested zones (~8 ms over ~135k records) | 0.059 per own record |
| whole tiles | t144 item-matched fits | NCRISC 5.5 + 0.197n, BRISC 16.1 + 0.187n, n > 6144: -282 + 0.2226n |

- **Merge (assumption):** 0.015 us per merged record (~20 cycles), plus 10 us per gather item to read
  both halves' keys and find its range (merge-path search). Basis: the radix sort costs 0.0829 us per
  record, which is 112 cycles over 3-4 scatter passes plus a histogram pass (`sort_radix_tile_algo.h`
  `choose_plan`, 1536 histogram entries). So one pass is ~25 cycles, and a two-way merge step does
  less work than one scatter pass. Tested at 0.010 and 0.030; for the big-tile split this changes
  nothing.
- **Split design modeled:** half-sort items for records [0, N/2) and [N/2, N). Each one reads its keys,
  sorts them, and publishes keys and ids. A stable merge that takes the left run on ties gives the
  same order as today, so output should stay md5-identical. Gather items wait for both halves, then
  merge their own rank range. A half fits BRISC's CB_BSORT ((2*6144+256) u32 = 12,544 keys), so
  tiles up to 25,088 records can split across one core's NCRISC and BRISC ("pair"). Any two movers
  were also modeled ("any").
- **Schedules:** `host` replays `build_mat_worklist` at a71c61e: LPT on host cost, big items on NCRISC
  only, sort items first and gather items last on each mover. `dep` is an oracle list scheduler on
  true costs that starts a gather only after its sort is done. It is a bound, not host code.

## Predicted materialize window (busiest mover, ms, traced scale)

`python3 docs/mat-split-sort-model/model.py [--sched host|dep] [--mid 8192]` (full output: `out/model.txt`)

| schedule | view 0 host | view 0 dep | view 1 host | view 1 dep | busiest mover is set by |
|---|---|---|---|---|---|
| off (re-sort per subchunk) | 3.350 | 3.337 | 3.218 | 3.218 | one whole tile (16,257 / 15,722 records) |
| t168 (shared sort, tip candidate) | 3.513 | 3.529 | 3.317 | 3.218 | sort chain of 25,019 tile; gathers idle 0.9-1.8 ms |
| **split, pair (NCRISC + BRISC)** | **3.337** | **3.337** | **3.218** | **3.218** | the same whole tile |
| split, any two movers | 3.337 | 3.337 | 3.218 | 3.218 | the same whole tile |
| big-tile sort free (bound) | 3.337 | 3.337 | 3.218 | 3.218 | the same whole tile |
| mean mover load (t168) | 2.880 | | 2.089 | | |

Split vs t168: **-0.176 / -0.099 ms** (host) and **-0.192 / 0.000 ms** (dep). The split shortens the
longest big-tile chain from 2.66 ms to 1.44 ms of sorting, but the busiest mover then becomes the
16k whole tile. That tile costs 3.34 ms by itself in every schedule.

How much to trust these numbers:
- The model overstates t168's gather-wait penalty. It puts t168 0.10-0.16 ms behind off, but on the
  device t168 was 0.08 ms ahead traced and 0.04 ms ahead untraced (t168 RESULT.md). Part of the
  predicted split gain only wins back that overstated penalty, so the real gain is smaller.
- The traced-to-untraced ratio measured in t168 was ~0.5. So the expected untraced gain is ~0.05-0.1
  ms/view, well under the 0.3 gate, with no margin.
- t144 saw the same floor independently on the device (item-matched replay): the busiest mover is set
  by "one item, a 14-16k-record whole tile, ~3.0-3.3 ms traced" or by NCRISC-only load.

## What does move the window (follow-up, model only so far)

Run the same split on whole tiles of 8192 < n <= 16384 records (`--mid 8192`): two half-sorts plus
merge-gather items. That removes the 3.3 ms single-item floor, and the window then drops toward the
mean:

| | view 0 | view 1 |
|---|---|---|
| host scheduler (as is) | -0.25 (merge 0.010) / -0.24 / -0.02 (merge 0.030) | -0.23 / -0.16 / +0.05 |
| dependency-aware scheduler | **-0.49 / -0.45 / -0.35** | **-1.03 / -1.01 / -0.93** |

It only pays with a scheduler that knows about dependencies. With today's "gathers last" order, the
gathers sit idle and the result swings from -0.84 to +0.73 ms depending on the threshold. Costs:
mean mover work goes up ~4% (each tile's records are published and re-read), and two pieces are
needed: a half-sort + merge-gather path for mid tiles, and a dependency-aware worklist built on
t144's calibrated item costs. Before building, it needs a model over all 30 views with real counts
(GSPLAT_TT_MAT_DUMP), using the host's cost estimates rather than the oracle.
