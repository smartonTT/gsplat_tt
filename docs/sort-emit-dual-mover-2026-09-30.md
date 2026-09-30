# Dual-data-mover `sort_bucket_emit` (T-C / R2 pilot, 2026-09-30, task #22)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views,
1024x1024, `python3 render/run.py --no-ref`. Baseline = T-B head `967e973`
(emit already de-stalled: 16.15 ms).

## Result

3 interleaved rounds per build in one device session (base, new, base, ...):

| metric | base (967e973) | T-C | delta |
|---|---:|---:|---:|
| `avg_frame_ms` | 152.99 ± 0.54 (152.4 / 152.9 / 153.7) | **145.42 ± 0.18** (145.2 / 145.4 / 145.7) | **−7.57** |
| FPS | 6.54 | 6.88 | |
| `stage_sort` | 25.97 | 18.76 | −7.21 |
| `stage_sort_bin_emit` | 16.15 | 9.00 | −7.15 (56 % of base) |
| `stage_sort_bin_count` | 0.906 | 0.920 | +0.014 |
| project / tile_assign / blend | 37.00 / 21.83 / 63.59 | 36.97 / 21.77 / 63.58 | ±0.06 |

Clean separation: slowest T-C round (145.7) < fastest base round (152.4).
`hero_vs_ref = 100.00 dB`, hero md5 `e3fefb116d860f99d92bba1ef51d820c`, and all
30 dumped views byte-identical to the base build.

30-view Tracy (profiler on), per-view busiest-core makespan
(`opt/profiler/ttw-149/zones.txt`):

| zone | base | T-C | delta |
|---|---:|---:|---:|
| `sort_bucket_emit` | 15.92 (NCRISC) | 8.43 (BRISC 8.43 / NCRISC 8.43) | −7.49 (1.89x) |
| `NCRISC-KERNEL` | 108.33 | 100.91 | −7.42 |
| `BRISC-KERNEL` | 86.23 | 94.61 | +8.38 |
| `BRISC-FW` | 143.08 | 135.67 | −7.41 |
| `sort_bin_hist` | 0.895 | 0.906 | +0.011 |

BRISC-KERNEL rises by about what NCRISC-KERNEL loses (+8.4 vs −7.4; the
extra ~1 ms is likely the fill-done wait, which both movers' zones include,
plus the empty BRISC launch in the count pass), and BRISC-FW, the program
makespan, falls with the frame: work moved to an idle mover, not shuffled. The 1.89x emit speedup matches T-A's p6
latency-bound dual-mover scaling (1.83x at 110 cores, `docs/hw-ceilings.md`).
Split ratio: BRISC share 45 % / 55 % both measured slower than 50 %
(bin_emit 9.69 / 9.76 vs 9.07 ms).

## Design (differs from the plan's per-(core, mover, tile) widening)

The plan widened the host histogram/prefix to per-(core, mover, tile). That
gives each mover its own page-aligned block per tile in the (key, id) layout,
roughly doubling the per-tile padding the radix sorts. Instead the two movers
of a core share the core's existing layout:

- **Count pass** (NCRISC only, as before) also writes a snapshot of its running
  histogram at `mid` to `buf_bin_h0`: mover 0's kept count per tile.
- **Emit**: BRISC takes pages `[lo, mid)`, NCRISC `[mid, hi)`. Both fill the
  core's one counting-sort region (CB 7/8). NCRISC's cursors start at `h0[t]`,
  so every pair lands in exactly the L1 slot, `buf_l1_recs` slot and overflow
  slot the single-mover pass gave it. Host layout, `bin2d`, `l1_rec_base` and
  the overflow bases are unchanged.
- **Ordering invariant** (the correctness argument): pairs are gaussian-major,
  so `[lo, mid)` precedes `[mid, hi)` in the single-mover order, and the count
  pass snapshots at the same `mid` the emit splits at. A gaussian straddling
  `mid` has its blendrec read by both movers and its 16 B packed-op/color
  chunk written twice with identical bytes (the same idempotent write two
  cores already made at page-range boundaries).
- **Sync**: tile blocks hold both movers' entries, so each mover pre-fills only
  the padding tails of the tiles it writes out, fences, sets its fill-done
  semaphore and waits for the peer's; then mover 0 writes out the tiles whose
  blocks start in the first half of the core's pages, mover 1 the rest.
- Every other staging CB is private per mover (BRISC's at id + 16, created
  after the originals so NCRISC's L1 addresses are unchanged): +42 KB L1.
- Kill switch / A-B in one build: `GSPLAT_TT_SORT_EMIT_MOVERS=1` builds the
  pre-T-C program exactly: no BRISC kernel, mover-0 CBs, semaphores or
  `buf_bin_h0`. `GSPLAT_TT_SORT_EMIT_SPLIT=<permille>` sets BRISC's page share
  (integer 0..1000; anything else warns and uses 500).

Checks: `tests/unit/test_sort_bin_dual_mover.cpp` replays the placement rules
single-mover vs split at every page boundary (21 447 cases, 0 mismatches;
mover-1 cursors from 0 or a full-range pre-fill per mover each fail most
cases), and the on-device 30-view byte-identity above.

Side finding: wrapping the count loop in a by-reference lambda called twice
cost +0.22 ms on `sort_bin_hist` even single-mover; the kernel keeps one inline
loop whose batches stop at `mid`.

## Follow-ups: other single-mover stages (same capture, per-view makespan)

| zone (kernel) | mover | makespan | expected with a split |
|---|---|---:|---|
| `proj_scatter` + `proj_count` (`gather_visible_scatter.cpp`, x2 programs) | BRISC | 20.8 + 15.0 | −12 to −17 ms if latency-bound (1.8x); only ~1.1x if the 32 B scatter is NoC-write-bound (T-A p6) |
| `ta_bucket_scatter` + `ta_gauss_aabb` (`tile_assign_*`, 7 kernels) | NCRISC | 10.5 + 9.2 | −8 to −9 ms |
| `sort_subchunk_mat` (`sort_subchunk_materialize.cpp`) | NCRISC | 12.4 | −5 to −6 ms |
| `sort_tile_depth` (radix, `sort_radix_tile.cpp`) | NCRISC | 7.2 | −3 ms (LPT tile lists split per mover) |

These ranges assume the 1.8–1.9x measured here; each needs its own
ordering/disjointness argument. Audit each for soft-float first (T-B found
~15 ms of libgcc float in the emit).

## How to reproduce

```
# base tree = 967e973, new tree = this branch, both built on the box with
# their own TT_METAL_CACHE_RENDER (the JIT cache does not tell trees apart).
devrun.sh --no-verify --timeout 580 --tag t22-ab3 -- "... for v in base new base new base new; do
  cd /localdev/smarton/<tree>; TT_METAL_CACHE_RENDER=<per-tree cache> python3 render/run.py --iter-dir t22-ab3 --no-ref; done"
GSTT2_REPO=<tree> TT_METAL_CACHE_RENDER=<per-tree profiler cache> bash opt/profiler/capture_tracy.sh ttw-149
```

Logs: `~/dev/gstt2/.ttw/logs/t22-ab3-*.log` (A/B + Tracy), `t22-ab2-*.log`
(count-loop fix), `t22-d-*.log` (30-view dumps), `t22-ab-*.log` (split ratios).
