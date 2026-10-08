# xvpin Tracy capture on bh-30 (task #407) — capture + analysis (#412)

Status: capture (#407) and analysis (#412, ana407.py) done. Conclusions are under
"Analysis" below; numbers there are traced (profiler on, ~13% slower than untraced).

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

First look from #407, superseded by the analysis below (out/fill_zones.txt, out/matblend_cores.txt): per view the device runs pfwc
(~1.8 ms) -> k2 (~0.85) -> sort (~1.34) -> mat+blend (~6.65) back to back; on a sample core
the program gaps are ~0.06-0.2 ms and pfwc N+1 starts ~0.09 ms after mat+blend N ends.
In the mat phase TRISC1 (MATH) is busy in band_batch 1.19 ms of 2.28 ms (mean core; max core
1.34 of 2.82), so ~1.1-1.6 ms of TRISC1 idle per core per view sits inside mat. The capture has
only whole-phase zones plus fz_* totals, so a per-gap list needs per-job zones.

## Analysis (#412)

Script: `python3 docs/xvpin-tracy/ana407.py <gunzipped xvpin-stitched-30v.csv> --zone-csv
docs/xvpin-tracy/out/ana407-zones-per-view.csv > docs/xvpin-tracy/out/ana407.txt`. It reads the
stitched device CSV, the Tracy `-u` host dumps (out/T*.tracy-u*) and the run logs (out/r*.log,
out/T*.log). Device clock 1350 cyc/us, 110 cores; all times ms per view, traced. View N -> N+1
transitions are measured only inside a chunk (27 of 30 views); each chunk has its own time base.

### Critical path

The device is the bottleneck and is almost never idle. Per view it runs four programs back
to back, with no overlap between them (no core starts a program before it finishes the
previous one):

| program | window | per-core mean | per-core max |
|---|---|---|---|
| pfwc (project+fill+count) | 1.891 | 1.766 | 1.891 |
| k2 (pairs/rows) | 0.829 | 0.774 | 0.829 |
| sort | 1.391 | 1.324 | 1.391 |
| mat+blend | 6.243 | 6.099 | 6.243 |
| period (view N start -> N+1 start) | 10.387 | | |

- Time when no core runs anything: 0.041 ms/view in total (mat+blend->pfwc 0.013, k2->sort
  0.013, sort->mat+blend 0.008, pfwc->k2 0.006). The xview hook works: pfwc N+1 is queued
  behind blend N, so the host's image read, Python and setup are hidden.
- The host is not on the critical path: per view it spends 2.09 ms in host_cq1_proj_m
  (waiting in finish for pfwc), ~0.22 ms setting up k2/bridge/blend, and then 6.53 ms in
  host_wait_blend_event waiting for the device.
- What the device loses inside its own programs:
  1. End-of-program load imbalance: each program ends when its slowest core ends. A mean core
     sits idle 0.43 ms/view (max-minus-mean summed over the 4 programs ~0.39 ms; mat+blend
     alone 0.14).
  2. Inside mat (2.29 ms mean, 2.82 max per core) the TRISCs wait on the BRISC/NCRISC movers
     about half the time (see (a)). The movers spend ~1.25-1.35 ms sorting and ~0.37 ms
     permuting per view each (fz_mv_sort/perm).
  3. Blend is the largest block: tile_blend_sfpu 3.68 ms mean, 4.09 max per core; it starts
     right after mat on each core (T1 mat end -> blend start 0) and the NCRISC loader
     (tile_blend_load 3.62) keeps up with it.
- So the frame is set by: pfwc 1.89 + k2 0.83 + sort 1.39 + (mat ~2.8 + blend ~4.1 on the
  slowest cores, overlapping to 6.24) + ~0.04 gaps. The levers are blend per tile, mover
  speed inside mat, and balance of the per-core work, not host work or launch gaps.

### (a) TRISC1 (MATH) idle inside mat

From the fz_* totals (the capture has no per-job zones), mean / max over cores:

- TRISC1 in mat: 2.285 / 2.822 ms; band_batch 1.191 / 1.345; copy 0.012.
- Idle bound (total - band - copy): 1.082 / 1.641 ms per core per view, i.e. ~47% of mat.
  TRISC0's pick_job wait is 1.098 / 1.656, the same amount: the TRISCs wait for the movers
  to hand them jobs. Mailbox wait (fz_m_mbw) reads 0.
- In blend units: tile-blend time per job = tile_blend_sfpu / jobs per core = 0.371 ms
  (max 0.513; 10.02 jobs per core), per 32x32 tile 0.395 ms. The idle equals ~2.96 blend
  jobs per core (max 5.1), 0.30 of the blend busy time. Spread over 192 mat batches it is
  ~5.8 us per batch (max 12 us).
- Upper bound: if every core filled all its mat idle with its own blend work, the slowest
  core would go 6.243 -> 5.288 ms, at most 0.955 ms/view (traced). This needs blend jobs to
  run while mat waits on movers; today each core starts blend only after its mat phase.
- Where exactly the gaps sit (start of mat, between jobs, at the end) needs per-job zones; see
  the follow-up (per-job DeviceZoneScopedN in mat_cull_compute.cpp + recapture).

### (b) Device idle between sort emit and mat

Last sort_ol_emit end -> first mat_cull_mask start: 0.009 ms per view on average (0.008 in
29 views, 0.030 in view 1). Time with no core busy inside it: 0.008 (view 1: 0.029). This gap
is not worth chasing. For comparison, mat+blend N end -> pfwc N+1 start is 0.013 ms (all
cores idle 0.013), but a mean core idles 0.157 ms there because it finished mat+blend early
(imbalance, not launch latency).

### Untraced stages vs traced device span

| | ms/view | source |
|---|---|---|
| untraced avg_frame_ms (r1-r3 mean) | 9.164 | r*.log |
| untraced per-view mean, same 27 views | 9.159 | r*.log `[run] view=` |
| untraced stages: project 1.157, tile_assign 0.008, sort 1.073, blend 6.416, d2h 0.288, xview 0.129, head+tail 0.011, other 0.078 | 9.164 | STAGES |
| traced host per-view mean | 10.274 | T*.log |
| traced device period = pfwc 1.889 + 0.006 + k2 0.829 + 0.013 + sort 1.391 + 0.008 + mat+blend 6.238 + 0.013 | 10.387 | stitched CSV, 27 views |
| profiler overhead (traced device - untraced) | 1.228 (~13%) | |
| traced host - traced device | -0.113 | chunk edges, clock offsets |

The untraced stages are host timers, not device windows: under xview pfwc N+1 runs during
N's d2h/xview, so untraced "project" (1.16) is shorter than the pfwc device window, and the other host stage
boundaries do not line up with device programs either. Treat device numbers above
as shares of the frame (traced), and the 9.16 ms untraced frame as the real speed.

Host Tracy caveat: device zones in the -u dumps sit on the host clock with a varying offset, so
host zones were analysed on their own, anchored on host_cq1_proj_m. EnqueueProgram starts
(view-relative) are 3.05, 3.17, 10.34 and 10.42 ms; the xview hook, EventSynchronize,
gather_wait, bin_layout and publish_host are stage timers only (TTW_TIMING), not Tracy zones.

Known facts unchanged: md5 906e0435 on 30/30 views in all 3 untraced rounds, XVIEW_HITS 29/30,
hero 42.512 dB vs benchmarks/reference_v2/hero.png, no tile artifacts.
