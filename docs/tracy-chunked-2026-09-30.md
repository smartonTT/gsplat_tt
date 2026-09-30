# Chunked 30-view Tracy device capture + per-core occupancy (2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (the ledger reference board, not a p150).
Build: `smarton/tt-project-opt` @ 76ffb66 (iter-142, task #18 sort cuts), rebuilt in the
isolated box tree `/localdev/smarton/gstt2-t16` (render_clean .so md5 `86aa1d2e…`).
Scene: bicycle, 30 views, 1024x1024, `render/run.py --no-ref`, device profiler on.

## Tooling

```
# from the Mac repo root; one devrun job per 10-view chunk, then an off-device stitch
GSTT2_REPO=/localdev/smarton/gstt2-t16 DEVRUN_FLAGS=--no-verify DEVICE_LOCK="ttp lock p150 --" \
  bash opt/profiler/capture_tracy_chunked.sh ttw-t16 10 480
```

- `render/run.py --view-range A:B` times `order[A:B]` (the warmup still renders the hero).
- `opt/profiler/capture_tracy.sh <iter> [A:B]` writes `opt/profiler/<iter>/chunks/A-B/`.
- `opt/profiler/stitch_device_csv.py` finds frames, drops each chunk's warmup, checks that
  every frame has one `proj_count` start per core and that every zone's START/END pairs
  stay inside one frame, then writes one CSV. The first version split frames at the widest
  timeline gap. That split was wrong: `sort_bucket_emit` has a ~30 ms stretch with no
  markers, and the warmup has multi-second JIT gaps. Frames now start at the first marker
  of the contiguous run that leads into each anchor cluster
  (`tests/test_stitch_device_csv.py` covers this case).
- `opt/profiler/zone_occupancy.py` prints the per-zone and per-core occupancy table, the
  stage windows, and the device timeline below. The full output is in
  `opt/profiler/ttw-t16/zone_occupancy.txt`.

## Row counts

| chunk | rows | frames | warmup rows | kept rows | job wall |
|---|---:|---:|---:|---:|---:|
| 0:10 | 218,614 | 11 | 19,906 | 198,708 | 27 s |
| 10:20 | 218,748 | 11 | 19,906 | 198,842 | 14 s |
| 20:30 | 218,506 | 11 | 19,906 | 198,600 | 14 s |
| **stitched** | **596,150** | **30** | | | |

The stitched CSV has exactly as many rows as task #18's single-job 30-view capture
(616,056 rows) minus its warmup frame (19,906), and all its zone tables agree with that
capture to within 0.01 ms. That is **~19.9k rows per view**. The old "~65.6k rows per
view, ~1M rows total" figure came from ttw-043, before the kernel fusions, and no longer
applies.

**The premise did not hold for this build.** One full 30-view capture already fits in a
single devrun job: task #18's took about 33 s of trace time with a 480 s limit. Chunking
is kept as a fallback for when a build adds many zones (for example, per-batch zones
inside the emit). It is not needed today.

Stitched CSV (82 MB; kept on the box, not committed):
`yyzo-bh-07:/localdev/smarton/gstt2-t16/opt/profiler/ttw-t16/profile_log_device.csv`

## Top device zones (per view, mean over 30 views)

The device frame span (first to last device marker) is 162.3 ms (131.5–198.9).
`max_ms` is the busiest core's time. `occ_%` is the busiest core's time as a share of the frame span.

| zone | RISC | cores | mean ms/core | busiest core ms | balance | occ % |
|---|---|---:|---:|---:|---:|---:|
| BRISC-FW | BRISC | 110 | 146.27 | 153.87 | 0.95 | 94.9 |
| NCRISC-KERNEL | NCRISC | 110 | 113.66 | 120.39 | 0.94 | 74.1 |
| BRISC-KERNEL | BRISC | 110 | 79.30 | 84.66 | 0.94 | 52.3 |
| TRISC-KERNEL | TRISC0-2 | 330 | 48.06 | 51.98 | 0.92 | 32.0 |
| sort_bucket_emit | NCRISC | 110 | 28.78 | 29.87 | 0.96 | 18.4 |
| tile_blend_sfpu | TRISC | 330 | 25.75 | 29.65 | 0.87 | 18.3 |
| tile_blend_load | NCRISC | 110 | 25.01 | 28.97 | 0.86 | 17.9 |
| tile_mb_mask | TRISC | 330 | 20.35 | 21.63 | 0.94 | 13.4 |
| tile_l1_cull_rd | NCRISC | 110 | 19.97 | 21.27 | 0.94 | 13.1 |
| proj_scatter | BRISC | 110 | 17.97 | 20.10 | 0.89 | 12.5 |
| proj_count | BRISC | 110 | 13.26 | 14.46 | 0.92 | 9.0 |
| sort_subchunk_mat | NCRISC | 110 | 9.48 | 11.93 | **0.79** | 7.3 |
| ta_bucket_scatter | NCRISC | 110 | 9.78 | 10.18 | 0.96 | 6.3 |
| ta_gauss_aabb | NCRISC | 110 | 8.86 | 8.88 | 1.00 | 5.5 |
| sort_tile_depth | NCRISC | 110 | 6.75 | 6.97 | 0.97 | 4.3 |
| pfwc | TRISC | 330 | 1.95 | 1.99 | 0.98 | 1.2 |
| sort_bin_hist | NCRISC | 110 | 0.85 | 0.86 | 0.99 | 0.5 |

Reading: NCRISC runs 74% of the frame on the busiest core, and TRISC (SFPU) runs 32%.
BRISC-FW's 95% is mostly dispatch wait, not work. Load is evenly spread across cores
(balance ≥ 0.86) except `sort_subchunk_mat` (0.79): its busiest core runs 2.5 ms longer
than the mean core.

## Device timeline vs the host stage timers

Each kernel zone's window (first START on any core to last END), measured from the frame's
first device marker. Host column: `TTW_TIMING stage_*` for the same build, from task #18's
A/B runs (166.6 ms/view).

| host stage (ms) | device zones | device window |
|---|---|---:|
| project 36.96 | pfwc 0.00→1.99, proj_count 2.03→16.49, proj_scatter 16.60→36.70 | 36.70 |
| tile_assign 21.79 | ta_gauss_aabb 36.91→45.79, ta_bucket_scatter 47.18→57.36 | 20.44 |
| sort 40.00 (bin_emit 30.26, publish_wait 8.02) | sort_bin_hist 58.70→59.56, sort_bucket_emit 60.93→90.79 (29.86), sort_tile_depth 90.83→97.81 | 39.11 |
| blend 63.60 (one Finish drains the CQ) | sort_subchunk_mat 99.04→110.97, cull 110.98→132.61, blend 132.63→162.28 | 64.47 (97.81→162.28) |
| d2h + assemble + head 4.2 | — (host only) | — |

The device timeline matches every host stage within about 1.3 ms. The device frame span
is 162.3 ms, and adding the 4.2 ms of host-only tail gives 166.5 ms, against 166.6 ms
measured. Between programs, the device sits idle for about **5.8 ms per view** in total.
The biggest gaps are aabb→scatter 1.39, scatter→hist 1.34, hist→emit 1.37 (the host
histogram D2H plus layout) and tile_depth→mat 1.23. These gaps are the host-bridge floor
between dependent programs.
