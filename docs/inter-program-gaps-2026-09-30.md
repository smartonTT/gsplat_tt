# Inter-program device idle (task #31, 2026-09-30)

Board: **yyzo-bh-07, Blackhole p100a** (not a p150). Scene: bicycle, 30 views, 1024x1024.

## What the "5.8 ms/view of idle" really was

`docs/tracy-chunked-2026-09-30.md` measured the gaps between *named kernel zones*.
Programs with no `DeviceZoneScoped` zone (the tile_assign scans, sort's `unnamed`
1.2 ms program before the materialize) looked like idle time there. The new
`opt/profiler/program_gaps.py` uses the firmware zones instead: every program
launch opens a `*-FW` zone on every core it runs on, so the union of all FW
intervals is "some core is running a program", and a gap in that union is time
when **no core runs anything**.

On the ttw-149 capture (T-C build, 54567ad):

```
python3 opt/profiler/program_gaps.py <ttw-149 profile_log_device.csv> --skip-first --min-gap-us 1
 #    start      end    busy gap_before cores  program
 0    0.000    1.988   1.988      0.000   110  pfwc
 1    2.028   36.695  34.667      0.040   110  proj_count+proj_scatter
 2   36.848   46.285   9.437      0.152   110  ta_gauss_aabb      (+ scan1 + scan_bases)
 3   46.331   57.288  10.957      0.046   110  ta_bucket_scatter  (+ scan2)
 4   58.575   59.452   0.877      1.287   110  sort_bin_hist
 5   60.801   69.009   8.208      1.349   110  sort_bucket_emit
 6   69.039   76.013   6.974      0.031   110  sort_tile_depth
 7   76.024   77.178   1.153      0.011   110  <unnamed 110 cores>
 8   77.239   89.171  11.932      0.061   110  sort_subchunk_mat
 9   89.178  110.821  21.643      0.007   110  tile_mb_mask+tile_l1_cull_rd
10  110.827  140.476  29.649      0.007   110  tile_blend_sfpu+tile_blend_load+rd_l1_bulk
total all-core idle between programs: 2.990 ms/view
```

Real all-core idle is **2.99 ms/view**, not 5.8. Per gap:

| gap | zone-gap (old doc) | all-core idle | cause |
|---|---:|---:|---|
| ta_gauss_aabb -> ta_bucket_scatter | 1.39 | 0.05 | not idle: scan1 + scan_bases + scan2 run there (no zones). The one host drain (scan Finish + 64 B P read) costs 0.05 ms. |
| ta_bucket_scatter -> sort_bin_hist | 1.34 | **1.29** | tile_assign's `Finish` after K2. Host Tracy: the host enqueues the count pass 40 us after that Finish returns, and the count pass itself starts within ~50 us of its enqueue, so the idle is the Finish returning ~1.2 ms after K2's last core ended. The host `tile_assign` timer shows the same ~1.2 ms excess over the device window in untraced runs. |
| sort_bin_hist -> sort_bucket_emit | 1.37 | **1.35** | host bridge: blocking 1.8 MB histogram D2H (0.24 ms), host layout (~0.25-0.35 ms), then ~5.4 MB of H2D tables (bin2d bases, l1_rec_base, l1_ov_base 1.8 MB each + small per-tile tables) that the device pulls before the emit can start (~0.75 ms). |
| sort_tile_depth -> sort_subchunk_mat | 1.23 | 0.07 | not idle: the unnamed 1.15 ms program runs there. |

## Change (gap 2: K2 -> count pass)

- `tile_assign_device.cpp`: in the production resident-pairs, no-cull path, skip the
  `Finish` after scan2 + K2. Nothing later in tile_assign reads K2's output on the host.
- `sort_bin.cpp` count pass (mode 0): new args 30..32 (ta_pairs_P ctrl page,
  num_cores, split permille). When the ctrl address is set, the kernel reads P and
  P_pad from the ctrl page and computes its page range and T-C mover split with the
  host's `split_pages` / `mover_mid` formulas.
- `sort_device.cpp`: enqueue the count pass first, then read P (the count pass's
  Finish has drained the CQ, so this read no longer waits), then run the overflow
  check, the P == 0 exit, and the host work split for the emit. The emit gets the
  same ranges as before, so every output byte is unchanged.

Timer note: `tile_assign` no longer includes K2's device time; that time now shows
up in sort's `bin_count`.

## Result

A/B, 3 interleaved rounds per build in one device session on **yyzo-bh-07 (Blackhole p100a,
not a p150)**, bicycle 30 views 1024x1024, base = 7df44de (iter-150 tree):

| round | base frame | new frame | base tile_assign+sort | new tile_assign+sort |
|---|---:|---:|---:|---:|
| 1 | 145.8 | 145.3 | 40.606 | 40.564 |
| 2 | 145.4 | 145.3 | 40.528 | 40.488 |
| 3 | 147.1 | 145.2 | 40.762 | 40.517 |

- tile_assign + sort: 40.63 -> 40.52 ms/view (**-0.11 ms**), smaller in every round.
- avg_frame_ms: base median 145.8 -> 145.3 (base round 3 is an outlier).
- All 30 dumped views byte-identical to base; hero_vs_ref 100.00 dB.

Dropping the host `Finish` removed only ~0.1 ms of the 1.29 ms gap. So the idle is
not mainly the host round trip: the count pass still starts late on the device even
when it is already queued behind K2. Not verified with Tracy on the new build (budget).

**Decision: rejected, not landed.** ~0.1 ms/view (0.07%) is far below the 1% bar and the
change adds a device-side P read and split to `sort_bin.cpp`. The patch is kept at
`docs/inter-program-gaps/k2-count-chain.patch` (against 7df44de).

## What is left and what to try next

- Real remaining all-core idle is ~3.0 ms/view (2% of frame), of which ~2.6 ms sits in
  the two sort-side gaps above. Worth pursuing only as part of a larger sort rework.
- K2 -> count pass: take a Tracy capture of the patched build and look at what the
  device does in the 1.2 ms (dispatch of the count program, a late core in K2, or a
  CQ wait) before trying anything else there.
- hist -> emit (1.35 ms): the host bridge is D2H 0.24 + layout 0.25-0.38 + ~5.4 MB H2D.
  Moving the layout (a prefix sum over the bin histogram) onto the device would remove
  the D2H and most of the H2D; expected gain is at most ~1.3 ms/view.

