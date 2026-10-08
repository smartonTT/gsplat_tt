# xvpin Tracy capture on bh-30 (task #407) — data landed, analysis pending

Status: capture and stitch done; the full analysis and the critical-path write-up
follow in a continue task. Nothing below is a conclusion yet.

- Build: ttp/t407-xvpin-tracy 275798b4 (base origin/smarton/tt-project-opt 14f42fda,
  xvpin = GSPLAT_TT_XVIEW_OVERLAP=1 + GSPLAT_TT_OUT_PINNED=1, both default on), bh-30 p150,
  11x10 worker grid, aiclk 1350 MHz. ETH 12x10 was not captured: the GSPLAT_TT_DISPATCH /
  opt/eth overlay is not on this base.
- Untraced 3x30 views (out/r1..r3.log): avg_frame_ms 9.188 / 9.169 / 9.135, md5 906e0435
  over all 30 views in every round (golden match), XVIEW_HITS 29/30 each round.
  Stage means r1: project 1.162, sort 1.066, blend 6.423, d2h 0.299, xview 0.132 ms.
- Tracy: 3 chunks of 10 views (+1 warmup each), XVIEW_HITS 9/10 per chunk, no
  "Profiler DRAM buffers were full". GSPLAT_TT_PROFILE_READ_EVERY=11 (new, render.cpp) reads
  the device profiler once per chunk process instead of after every render, so the per-view
  ReadMeshDeviceProfilerResults stall (finish() on all CQs) does not break the overlap.
  --dump-device-data-mid-run stays on: render_clean never closes the device, so that one
  read is the only one that reaches profile_log_device.csv.
- xvpin-stitched-30v.csv.gz: the 30 timed views. stitch_device_csv.py was fixed for xview
  (no host gap between views): anchor clusters try a 2 ms gap first, and a frame never starts
  before the last *-FW ZONE_END ahead of its anchor cluster.
- Hero (out/hero.png, view00 of r1): 42.512 dB vs benchmarks/reference_v2/hero.png, 10x diff
  in out/hero-diff10x.png. Looked at: no tile seams or tile artifacts; differences sit on thin
  edges (spokes, frame) as in earlier iterations.
- Viewer downtime: stopped 02:02:11Z, READY + localhost:8091 200 at 02:05:08Z (~3 min).

First look (out/fill_zones.txt, out/matblend_cores.txt): per view the device runs pfwc
(~1.8 ms) -> k2 (~0.85) -> sort (~1.34) -> mat+blend (~6.65) back to back; on a sample core
the program gaps are ~0.06-0.2 ms and pfwc N+1 starts ~0.09 ms after mat+blend N ends.
In the mat phase TRISC1 (MATH) is busy in band_batch 1.19 ms of 2.28 ms (mean core; max core
1.34 of 2.82), so ~1.1-1.6 ms of TRISC1 idle per core per view sits inside mat. The capture has
only whole-phase zones plus fz_* totals, so a per-gap list needs per-job zones.
