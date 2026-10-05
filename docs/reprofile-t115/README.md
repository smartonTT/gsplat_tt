# Re-profile of the tip after iter-177, and the next levers (task #115, 2026-10-02)

Board for every number: **yyzo-bh-07, Blackhole p100a, 110 cores** (not a p150). Bicycle,
1024x1024, 30 views. Code: tip a03bd1e; the remote tree was 298139f (tip + these scripts,
no render changes). Every run was md5-identical to `md5-r82new.txt` over 30 views,
hero_vs_ref 100.00 dB. GPU reference: 10.75 ms/view (published, not measured).

## How it was run

- `drive.sh` (Mac): each device step is its own `ttp lock p100 -- devrun ...` job.
- `remote_time.sh`: untraced 30-view `render/run.py` runs. Arms: `base`, `hp`
  (`GSPLAT_TT_HOST_PROFILE=1 GSPLAT_PER_VIEW_STAGES=1`), `dc` (`GSPLAT_TT_DUMP_CULL` +
  `dead_pairs.py`).
- `remote_tracy.sh`: 30-view Tracy (`opt/profiler/capture_tracy.sh`), then
  `stitch_device_csv.py`, `program_gaps.py`, `zone_occupancy.py`, `analyze_zones.py` and the
  new `opt/profiler/risc_roofline.py` (per program and RISC: mean core vs busiest core).
- Raw outputs: `out/` (`chain.log`, `run-r*.log`, `hp-r1.txt`, `gaps.txt`, `roofline.txt`,
  `zones.txt`, `zone_occupancy.txt`).

## Frame time (untraced)

| round | ms/view | FPS |
|---|---:|---:|
| r1 | 29.47 | 33.9 |
| r2 | 29.53 | 33.9 |

Per-view range 25.2-34.0 ms. Stage timers (ms/view): project 6.46, tile_assign 1.44,
sort 10.66, blend 10.66-10.72, d2h 0.21-0.28. Sort: bin_count 0.91, hist_d2h 0.25,
layout 0.23, upload 0.07, bin_emit 5.29, publish_host 0.19-0.21, publish_wait 3.57-3.59.
Project: pfwc_finish 2.48, gather_wait 3.77, gather_result 0.10. TA: k2_finish 1.40.

## Device timeline (Tracy, 30 views)

Device span 29.45 ms = 27.46 ms of programs + 2.00 ms with every core idle. Programs run
back to back on all 110 cores (one in-order queue).

| # | program | window ms | idle before | mean / busiest core ms (RISC) |
|---|---|---:|---:|---|
| 0 | pfwc | 2.46 | 0 | TRISC 2.39 / 2.44 (SFPU-bound, balanced) |
| 1 | proj_vis_scan + proj_scatter | 3.76 | 0.04 | scan 0.40 on 1 core, serial; scatter BRISC 3.11 / 3.35 |
| 2 | ta_bucket_scatter | 1.42 | 0.42 | 1.35 / 1.42 |
| 3 | sort_bin_hist | 0.87 | 0.06 | NCRISC 0.86 / 0.87 |
| 4 | sort_bucket_emit | 4.51 | **1.34** (host hist->emit bridge) | 4.30 / 4.51 |
| 5 | sort_tile_depth (radix) | 2.59 | 0.04 | NCRISC 1.95 / 2.43, BRISC 2.22 / 2.55 |
| 6 | sort publish | 1.15 | 0.01 | NCRISC 1.06 / 1.15 |
| 7 | sort_subchunk_mat + mat_cull_mask | 3.17 | 0.08 | TRISC 2.90 / 3.17, NCRISC 2.80 / 3.15 |
| 8 | blend | 7.52 | 0.01 | TRISC 6.90 / 7.45 (SFPU-bound) |

- Load imbalance (busiest minus mean) adds up to about 2.1 ms. Largest: blend 0.55, radix
  0.49, mat 0.36, scatter 0.24 plus the 0.40 serial scan.
- Per core, summed over a frame (mean / busiest): BRISC 23.2 / 24.1, NCRISC 24.3 / 25.4,
  TRISCs 12.2 / 12.8. Mover time includes CB waits, so 25.4 ms is a loose bound, not a
  floor. The TRISCs (SFPU/FPU) are busy 44% of the frame.
- Host time only shows up in two gaps: the hist->emit bridge (1.34 ms: hist D2H 0.25,
  layout 0.24, upload 0.07, the rest is launch) and gather->TA (0.42 ms: gather result 0.10
  plus TA enqueue). Other host work overlaps device time.

## Record facts

- Hero view (`dc` arm): 3,369,033 records, **664,773 (19.7%) have an all-zero microblock
  mask** (dead). In tiles with more than 8192 records (40.2% of records) 18.5% are dead.
  Live records cover 3.06 of 32 microblocks on average. Max tile 25,699 records, mean 3,290.
- pfwc's tile rect is `ceil(3 * sqrt(cov2d diagonal))` per axis (`pfwc_compute.cpp:14`). It
  ignores opacity: for opacity below about 0.35 the alpha >= 1/255 ellipse is smaller than
  3 sigma, and corner tiles of the box often miss the ellipse. Those are the dead records.
- `[OVERFLOW-DIST]`, 30 views: 69-100 tiles with 8193-16384 records are pre-packed and
  sorted in L1 by the materialize. Only **7-13 tiles per view with more than 16384 records
  (140k-243k records, 5.5-7.2%)** read `sort_sorted_ids`.
- Code read (`sort_subchunk_materialize.cpp:519-611`): in-budget and in-cap overflow tiles
  are sorted in L1 by the materialize itself. Still, the radix (2.59 ms) and the publish
  (1.15 ms) run over every tile, and the emit writes the keys/ids layout for every pair.

## Levers ranked by measured upper bound

Gate: deep-tier work needs >= 3 ms/view, or must compound.

| # | lever | upper bound ms/view | basis | tier |
|---|---|---:|---|---|
| A | One-launch sort v2: port #100's PB/RING into `sort_bin_onelaunch`, split big-tile materialize items | **3.5-4.6** | t114 measured -1.07 with its emit at 8.26 ms vs 5.29 legacy (+2.97) and mat +0.58 from one 31-33k item vs a 12.7-13.3k mean slot | deep |
| B | Fuse the visible-gaussian compaction (and TA pair emission) into pfwc's writer | **3.8 gross** (gather + gap), **5.6** with TA and its gap; minus the writer's added cost (not measured) | Tracy: gather 3.76, gaps 0.04 + 0.42, TA 1.42 | deep, probe first |
| C | Conservative dead-record pre-cull (opacity-aware rect + ellipse-tile test) | 2.0-2.7 x removable share | 19.7% dead x (TA + hist + emit + radix + publish + mat); 2.0 after A | standard after a probe; compounds with A |
| D | Radix and publish only for tiles over 16384 records (fallback if A fails) | ~1.5-3, likely ~2 | publish 1.15 -> ~0.1; radix limited by the largest tile; keys/ids writes in emit | standard |
| E | Blend bundle (coefficient staging, const hoist, big-tile split) | <= 2.5-3, realistic <= 1.5 | t111 2.0, t78 0.9, imbalance 0.55 | defer |
| F | Trace replay / launch overhead | <= 0.3 once A and B land | A removes the 1.34 bridge, B the 0.42 gap | defer |
| G | Chunk frustum cull before pfwc | <= 1.5-2 | 17-31% of 6.13 M gaussians visible; md5 tie-order risk if reordered | defer, after B |

**Plain answer:** two levers clear the 3 ms gate on measured data (A and B). C is
cheap to probe and compounds with A. Everything else is under 2 ms alone and should stay
deferred. If A, B and C land near their realistic values (3.5 + 3.5 + 1.5) the tip goes from
29.5 to about 21 ms/view (~48 FPS) on the p100a. That is still about 2x the published GPU
reference. After that, blend (7.5 ms, SFPU-bound) and mat+cull (3.2 ms) are most of the
frame, and the FPU blend / tile-owner L1 pipeline ideas come back.

## Task specs (ready to queue)

1. **A: one-launch sort v2** (deep, device, no dependencies). Start from the
   `GSPLAT_TT_SORT_ONELAUNCH` code already on the tip (#106, default off). Port the legacy
   emit's PB batching and RING write coalescing (`GSPLAT_TT_EMIT_PB=8`, `RING=8`, PUBOC) into
   `sort_bin_onelaunch.cpp`, within L1 (RING needs ~256 KB per mover next to the count page
   cache; RING=16 already overflows with dual movers). Then split big-tile materialize items
   (cache a per-tile sorted index in DRAM, or split items by subchunk) so no item exceeds
   ~1.5x the mean slot (`GSPLAT_TT_MAT_STATS=1`). Gate: md5-identical over 30 views,
   `ONELAUNCH_CHECK bad_tiles=0`, >= 3 ms/view better than the tip in 3 interleaved rounds.
   Measure the one-launch emit zone against the legacy 5.29 ms first; stop if the port leaves
   it above ~6 ms.
2. **B: fuse compaction into pfwc** (deep, device, no dependencies; touches pfwc, gather,
   TA and the gid-indexed consumers). Step 1, probe: give pfwc's writer the compaction work
   (copy visible lanes, using the per-tile visibility mask, into per-core segments plus the
   64 B blendrec) without switching consumers, and measure the pfwc window. Continue only if
   pfwc grows by less than ~1 ms. Step 2: delete `proj_vis_scan` and `proj_scatter`; keep the
   gaussian order (per-core or per-tile segments in gid order) so the stable depth sort's tie
   order and the md5 stay the same. Step 3, optional: emit TA pairs from the same writer and
   delete `ta_bucket_scatter`. The pfwc program is ~66 KB of the 70.6 KB kernel config
   buffer: move the RECHECK soft-float path out of line before growing it.
3. **C: dead-record pre-cull probe** (standard, one short device run for dumps, then host
   analysis). From the `GSPLAT_TT_DUMP_CULL` records plus per-gaussian conic, opacity and
   mean, count the dead records removed by (a) a per-axis rect of
   `min(3, sqrt(2 ln(255 * opacity))) * sigma` and (b) an ellipse-vs-tile test, each with a
   rounding margin, and confirm that no live record is removed in 30 views. Implement in
   pfwc (rect) or TA (ellipse test) only if the removable share times the stage cost is
   >= 1.5 ms. Run after A lands so the gain is measured on the new sort.
4. **D: radix and publish only for over-cap tiles** (standard, device). Queue only if A
   fails. Build the radix and publish tile lists from tiles with more than 16384 records,
   and stop writing keys/ids for the other tiles in the emit. md5-safe by construction
   (only the over-cap gather reads `sort_sorted_ids`). Measure the radix makespan first: if
   the largest tile alone takes ~2.4 ms, only the publish and emit savings (~1-2 ms) remain.

## Notes

- ssh to yyzo-bh-07 warned that its host key changed (`known_hosts` line 68, new ECDSA
  fingerprint SHA256:aCKm7NxYiqDQSFnWQZvLGOQTsTuM9HusRgrRsa/KTBQ). ssh still connected
  with key auth because the existing ssh config does not enforce strict checking for this
  host. Nothing in `known_hosts` or the ssh config was changed. The key should be verified.
- `devrun.sh` warns when `--timeout` is above the 400 s reservation command limit. Measured
  device-held times: ~46 s for the 30-view Tracy, ~75 s per timing round.
