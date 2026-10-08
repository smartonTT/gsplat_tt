# ETH 12x10 Tracy + scaling/bridge check on bh-30 (task #427, iter-215 stack)

Status: done. Verdict: **one scaling loss over the 0.15 ms/view line: pfwc** (per-core load
imbalance). The emit->mat bridge is not a lever. All numbers are traced (profiler on) unless
marked untraced; ms/view at 1350 MHz.

## Setup
- Branch ttp/t427-eth-tracy (base best-iter-215 ff9bd5ce + #426/#431 zone_hash_check). Remote
  tree bh-30:/localdev/smarton/p150bench/tree392, bh-30 p150 under the viewer exception
  (IRD job 135630; no reserve/extend/release). Viewer stopped 03:44:13Z, READY again and
  localhost:8091 page 200 / websocket 101 at 03:48:55Z (~4.7 min).
- **ETH Tracy enablement.** With the profiler on, the idle-ERISC cq_prefetch is 0x4668 bytes,
  over the 24 KB link-time bound (0x4490 left) in the overlay's `kernel_ierisc.ld`.
  `opt/eth/make_overlay.sh` gained `ETH_IERISC_KB` (default 32 = byte-identical output; 36 for a
  profiler-only overlay `ttm-eth12p36` with its own JIT cache). No tt-metal rebuild.
- Zone-hash pre-check (opt/profiler/zone_hash_check.py at the remote tree): 149 zones, no 16-bit
  collision (out/zone_hash_check.txt).
- md5 neutrality, untraced 30 views, ETH 12x10: default 32 KB overlay (U) and the 36 KB profiler
  overlay (UP) both give 39d84b28 on 30/30 (out/md5-U.txt, out/md5-UP.txt, identical). No
  render code changed, so no new hero/PSNR card is needed; out/hero-U.png is the device hero.
  Untraced U: avg_frame_ms 7.941.
- Tracy: GSPLAT_TT_MATCULL_PROF=1, GSPLAT_TT_PROFILE_READ_EVERY=11, 3 chunks x 10 views
  (+1 warmup each), arm E12 (ETH 12x10) and arm E11 (ETH, GSPLAT_TT_GRID_X=11) on the same
  stack, so the scaling check does not mix stacks. #412's 11x10 trace (pre-ETH, older stack) is
  in the table too, as the spec asked.

Reproduce: `stitch_device_csv.py -o raw/stitched-E12.csv raw/dev-E12-c{0,10,20}.csv` (gunzip
out/dev-*.csv.gz first), then `python3 docs/xvpin-tracy/ana407.py raw/stitched-E12.csv --out
ana-E12 --zone-csv ana-E12/zones-per-view.csv` (same for E11), and `pfwc_cores.py`.
ana-E*/ hold symlinks to the run logs and tracy -u dumps that ana407.py expects. The
reconciliation block in ana-E11/ana.txt uses the 12x10 untraced log (only U was run untraced).

## Per-program windows, 12x10 vs scaled 11x10 (ms/view, mean of 30 views)

| program | E12 (12x10) | E11 (11x10, same stack) | E11 x 110/120 | E12 miss | #412 11x10 | #412 x 110/120 |
|---|---|---|---|---|---|---|
| pfwc | 1.899 | 1.892 | 1.734 | **+0.165** | 1.889 | 1.732 |
| k2 | 0.758 | 0.829 | 0.760 | -0.002 | 0.829 | 0.760 |
| sort | 1.312 | 1.395 | 1.279 | +0.033 | 1.391 | 1.275 |
| mat+blend | 5.389 | 5.793 | 5.310 | +0.079 | 6.238 | 5.718 |
| 4 program gaps | 0.097 | 0.056 | 0.056 (not scaled) | +0.041 | 0.040 | 0.040 |
| device period | 9.455 | 9.958 | 9.128 | +0.327 | 10.387 | 9.521 |

(#412's mat+blend is ~0.4 ms higher than this stack's E11 because of iter 213-215; compare
against the same-stack E11 column.)

### pfwc (the one miss >= 0.15)
Per-core pfwc zone (max over RISCs), out of pfwc_cores.txt:

| | E12 | E11 | E11 x 110/120 |
|---|---|---|---|
| mean core | 1.607 | 1.718 | 1.575 (+0.032) |
| max core | 1.800 | 1.834 | 1.681 (+0.119) |
| max / mean | 1.120 | 1.067 | |
| per-core idle pfwc -> k2 (mean core) | 0.251 | 0.132 | |

The average core scales almost ideally; the slowest core does not. The fused pfwc deals tiles
strided (core c owns c, c+C, ...), so the data split is even; the slow cores are specific
locations (E12: (14,3) slowest in 14/30 views, (11,5) in 8, then (3,2), (6,5); E11: (15,2),
(15,5), (3,5)), not a whole column. That points at NoC/DRAM placement contention, not work
count. Every other core waits ~0.25 ms for them before k2. Upper bound if the max core came
down to ~1.03 x mean: ~0.14 ms/view traced (~0.12 untraced, ~1.5%).

### mat+blend (+0.079, below the line)
Blend scales (max 3.732 vs 4.007 x 0.917 = 3.673). The mat phase's slowest core does not
(2.505 vs 2.525): it is the big-tile core (fz_mv_big max 2.12 ms on one NCRISC). #425's
packed big-tile sort (iter 216, landing) already targets that core.

### sort (+0.033) and host
Sort's window is the emit/town phase (~1.1-1.2 ms) and scales to within 0.03. Host Tracy zones
stay off the device critical path: host_blend_setup 0.087 (E12) vs 0.078 (E11), k2_rows
0.126 vs 0.122, finish bridge 0.032 vs 0.028; all of them fall inside device work. The
#423 untraced growth of sort host mat / publish_host (~0.07 ms) is not visible as device idle.

## Emit -> mat bridge
All-core idle between the last sort emit end and the first mat start: mean 0.059 ms/view
(E12), 0.020 (E11). In steady state it is 0.010-0.011 ms in both grids; the mean is raised only
by chunk-start views (E12 views 0/1/2/10/20/21: 0.60/0.31/0.30/0.07/0.18/0.06; E11 views 1/2:
0.11/0.18), i.e. the first views after each chunk's warmup, which the live viewer does not
see. Well under the 0.15 line. All-core idle per view period: 0.101 (E12) / 0.056 (E11).

## Verdict (decision rule)
- Bridge idle 0.059 mean, 0.011 steady < 0.15: no device bin_layout/publish task.
- pfwc misses ideal scaling by 0.165 ms/view (traced) >= 0.15: propose a pfwc scaling-loss task
  (load balance of the slowest cores at 12x10). It is borderline untraced (~0.14), so the task
  should first confirm the per-core spread untraced-equivalent and find why those cores are slow
  (reader DRAM wait vs writer, per-RISC zones on the slow vs median core) before changing the
  split. It overlaps #429 (chunk cull before pfwc), which changes pfwc's per-core work; run it
  after #429 decides.
- No other program misses by >= 0.15 (sort +0.033, mat+blend +0.079, k2 -0.002).

## Files
- out/: run logs (U, UP, TE12-*, TE11-*), md5 lists, per-chunk device CSVs, tracy -u dumps,
  hero-U.png / hero-UP.png, zone_hash_check.txt, overlay-p36.log.
- ana-E12/ana.txt, ana-E11/ana.txt (+ zones-per-view.csv): full ana407 output.
- pfwc_cores.py / pfwc_cores.txt: per-core pfwc balance.
- bench427.sh, drive427.sh, sync_bh30.sh, vstart.sh: driver (Mac) and remote bench.
