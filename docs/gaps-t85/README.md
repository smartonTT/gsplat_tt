# Inter-program host gaps at tip (task #85, 2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views, 1024x1024.
Base: 9cced7e (tip, 44.6 ms/view). Task #83's Tracy capture showed 5.57 ms/view of
all-core idle between programs. This task measured the same boundaries untraced with the
host stage timers (`STAGES`, `*_STAGES`) and a new per-view breakdown
(`GSPLAT_PER_VIEW_STAGES=1` in `render/run.py`, prints `VIEW_STAGES` per view).

## Finding

The largest gap was not host dispatch. The tile_assign `publish` bucket (1.33 ms/view)
came from 4 of the 30 views (orb_07..orb_10, 9-12 ms each). On those views P was larger
than on any earlier view, so the grow-only pair buffers (gid/tid/keep) were reallocated,
and each reallocation re-filled the all-ones keep mask: a ~13-19 MB host vector, H2D and a
`Finish`. These are the 59-61 ms views in the per-view list (neighbours ~46-50 ms). The
Tracy chunk (views 0-9) averaged two of these spikes into the ta_bucket_scatter ->
sort_bin_hist gap, which is why it read 2.56 ms instead of ~1.3 ms.

## Gap attribution (untraced, ms/view, mean of 30 views)

| gap (traced t83 value) | untraced | attributed to |
|---|---:|---|
| proj -> ta_gauss_aabb (1.39) | ~0.21 | host: gather ProjectResult build 0.11 + TA setup/rtargs/enqueue 0.10. The traced 1.39 is profiler-inflated. |
| ta_bucket_scatter -> sort_bin_hist (2.56) | 1.33 -> **~0.07** | keep-mask refill on buffer regrowth (1.31, **fixed**); rest is sort pread 0.02 + count-pass launch ~0.05 |
| sort_bin_hist -> sort_bucket_emit (1.38) | ~1.3-1.4 | host bridge: hist D2H 0.25 + layout 0.24 + upload enqueue 0.07 + device pull of ~5.4 MB H2D tables before the emit (~0.8, from bin_emit 8.84 minus emit busy) |
| sort tile_depth enqueue | ~0.2 | sort publish_host 0.18-0.21 |
| sort -> blend chain | 0.03 | blend_setup |
| **total before** | **~3.4** | |
| **total after** | **~2.0** | hist->emit bridge is the only piece > 0.25 ms |

## Change

`render/host/tile_assign_device.cpp`: the pair buffers are allocated at
`max(P_pad, pair_ceiling)` (4,718,592 pairs, the existing static ceiling with ~27 % margin
over the 30-view peak). The keep mask is then filled once, on the warmup view. A P above
the ceiling still grows the buffers as before. Device work and output bytes are unchanged.

## Result

3 interleaved rounds, base tree 9cced7e vs fix, same device session, untraced:

| round | base ms/view | fix ms/view | base max | fix max |
|---|---:|---:|---:|---:|
| 1 | 44.6 | 43.4 | 60.7 | 50.2 |
| 2 | 44.5 | 43.1 | 61.1 | 50.3 |
| 3 | 44.8 | 43.3 | 62.6 | 50.1 |
| mean | **44.63** | **43.27** | | |

- **-1.37 ms/view (-3.1 %), 23.1 FPS.** tile_assign 8.89 -> 7.58 ms, publish 1.33 -> 0.001 ms.
- Worst view 61-63 ms -> 50 ms (the spikes are gone).
- All 30 views md5-identical to `md5-r82new.txt` in every round; hero_vs_ref 100 dB.

Tracy on the fix (views 0-9, `tracy-fix.txt`): all-core idle between programs
**5.57 -> 2.46 ms/view**. ta_bucket_scatter -> sort_bin_hist 2.56 -> 0.06 ms,
proj -> ta_gauss_aabb 1.39 -> 0.82 ms (traced only; the profiler adds ~0.75 ms to the
traced project stage), sort_bin_hist -> sort_bucket_emit 1.35 ms (unchanged).

## What is left

About 2.0 ms/view of untraced device idle remains, below the 3 ms gate. The only piece
worth a task is the hist -> emit bridge (~1.3 ms): a device-side prefix sum over the bin
histogram would remove the D2H, host layout and most of the H2D tables. On its own it
does not pass the gate; do it only together with other sort-side work.

Other grow-only buffers were checked with the per-view breakdown: after this fix no view
shows a one-off stage spike (every stage tracks the view's P).

## Files

`remote_run.sh` (30-view run + md5 check), `remote_ab.sh` (interleaved rounds), `ab.log`,
`remote_tracy.sh` + `tracy-fix.txt` (program gaps). Trace:
`yyzo-bh-07:/localdev/smarton/gstt2-t85/opt/profiler/t85-fix/chunks/0-10/render.tracy`.
Remote trees: `/localdev/smarton/gstt2-t85` (fix), `/localdev/smarton/gstt2-t85b` (base).
