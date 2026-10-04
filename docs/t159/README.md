> **Note (task #182, 2026-10-04):** this describes the lever-2 chain (GSPLAT_TT_PFWC_FUSE=0: proj_vis_scan / proj_scatter / ta_bucket_scatter), not the default pipeline (PFWC_FUSE=1 since #122: pfwc, K2, sort_ol, mat, blend). See docs/vis-scan-t180/RESULT.md (commit e13e4f8 on ttp/t180-device-build-a-b-of-fused-multi-core-pro) and docs/rerank-17ms.md.

# t159: zone/gap/deep re-analysis of t142-pc (PRECULL) capture, yyzo-bh-07 p100a

Source: yyzo-bh-07:/localdev/smarton/gstt2-t142/opt/profiler/t142-pc/chunks/0-10/profile_log_device.csv
(11 frames = warmup + 10 views; traced, so absolute times exceed the untraced ~18.2 ms/view tip).
Scripts at 307a9ff: program_gaps uses segment_frames (9a3a33f); analyze_zones and deep_zones now drop
orphan START/END markers via stitch_device_csv.kept_rows. zones/deep normalize by 11 frames (incl. warmup);
gaps skips the warmup.

- NCRISC-KERNEL busiest-core makespan: 21.13 ms/view (device window 24.34 ms) -> one frame, fixed.
  Before the filter, deep_zones showed NCRISC busy 1067 ms/view, occupancy 113% on the fixture.
- Programs per view (ms): pfwc 2.73 | proj_vis_scan+proj_scatter 3.70 | ta_bucket_scatter 1.36 |
  sort_ol (count/barrier/prefix/emit) 4.51 | materialize+cull+blend 10.80. Device window 24.34.
- All-core idle between programs: 1.245 ms/view, mostly 0.812 before ta_bucket_scatter and 0.308 before
  materialize (traced; per #155 check the untraced host bridge before targeting these).
- Surprises: proj_vis_scan runs on ONE core (1 pair/view, 0.40 ms) inside the proj program, a serial step;
  deep_zones' "sort" bucket swallows mat_cull_mask ("mat_" matched first), so its "cull" row reads 0;
  blend TRISC sfpu 7.05 vs NCRISC load 6.66 / rd_l1_bulk 6.60 per-view makespan: the reader and compute
  are nearly balanced, so blend is not purely compute-bound.
