# t528: per-stage back-to-back device profile at best-iter-224 (structural plan input)

Task #528, 2026-10-12 (PDT). Build best-iter-224 (ad647d8e, MAT_EARLY_CULL on) plus this
task's profiling scripts only (b26e530e / 08c6d358: no render code changed). Bicycle, 30 views,
`render/run.py --back-to-back`, defaults. Box: bh-33 (IRD 138447, existing measurement
reservation; **card p100a**, 11x10 = 110 worker cores, CHIP_FREQ 1350 MHz in every profiler
header). No reserve/extend/release, no bh-30, all ssh with StrictHostKeyChecking=yes.

## Headline: b2b ms/view (untraced)

| round | passes | ms/view median | raw_md5 |
|---|---|---|---|
| r0 | 20 | 9.134 | 5a438f5a |
| r1 | 20 | 9.130 | 5a438f5a |
| r2 | 20 | 9.132 | 5a438f5a |
| r3 | 20 | 9.131 | 5a438f5a |

Median of 4 rounds: **9.131 ms/view** (109.5 FPS), identical images across rounds.
Device timeline steady period (profiler, 18 steady views, device clock): **9.130 ms**, the
same number. The device profiler does not change the image: untraced chunk U0 and profiled
chunk P0 (views 0-4) both give raw_md5 06fbaf11.

## 1. Where the 9.13 ms goes (ranked, ms per view, steady views)

Window = first core start to last core end of a program. Mean core busy = average over the
110 cores of that core's own program length. Source: `out/ana/steady.txt`, `out/ana/ana.txt`.

| rank | what | ms/view | share | bound by |
|---|---|---|---|---|
| 1 | blend (`tile_blend_sfpu`, inside mat+blend) | 3.67 mean core / 3.94 max | 40% | SFPU (TRISCs), per-tile fixed cost + per-pair work |
| 2 | pfwc (project + cov2d + visibility + compaction, one program) | 1.65 window / 1.55 mean core | 18% | TRISC SFPU over **all N = 6.13 M** gaussians (t484); DRAM ~70% of achieved (derived, sec. 3) |
| 3 | sort (one-launch bin: count, prefix, emit/town) | 1.46 window / 1.36 mean core | 16% | scalar RISC-V movers (`sort_ol_emit` 1.24, `sort_ol_town` 1.12 busy) |
| 4 | mat (L1 sub-chunk sort + cull, inside mat+blend) | 1.42 mean core / 2.04 max | 16% of core time, on the critical path only via the slowest cores | mover scalar sort (`fz_mv_sort` 0.85-0.88 ms per mover) + TRISC1 band_batch 1.08 |
| 5 | k2 (pair/row emit) | 0.65 window / 0.54 mean core | 7% | scalar pair loop on TRISCs + movers; worst imbalance (1.19) |
| 6 | end-spread (load imbalance) at the 4 program boundaries | 0.49 per core | 5.4% | each program waits for its slowest core |
| 7 | all-core idle (dispatch gaps, no core busy) | 0.04 | 0.4% | 6-13 us go-signal gaps |
| | host | 0 on the critical path | | host residue `py_resid` 0.06 ms runs while the device works |

Steady period split: windows pfwc 1.654 + k2 0.647 + sort 1.459 + mat+blend 5.330 + all-core
gaps 0.040 = 9.130 ms. Mean core busy sums to 8.635 ms, so the mean core is idle 0.495 ms
per view (5.4%).

## 2. Per-program device time, imbalance and idle

Program windows over all 30 views (`ana.txt`; mean over views):

| program | window | mean core | slowest core | end spread (idle per core) | max/mean core |
|---|---|---|---|---|---|
| pfwc | 1.659 | 1.554 | 1.659 | 0.105 | 1.07 |
| k2 | 0.653 | 0.550 | 0.653 | 0.103 | 1.19 |
| sort | 1.468 | 1.376 | 1.468 | 0.092 | 1.07 |
| mat+blend | 5.368 | 5.222 | 5.368 | 0.146 | 1.03 |

All programs start on every core together (start spread 0.000: go-signal multicast), so
the idle inside a window is all end spread. No two programs overlap on a core (0 per-core
overlaps): a core that finishes early waits for the slowest core and then for dispatch.

Boundaries (steady views):

| boundary | all-core idle mean / max | per-core idle mean / worst core |
|---|---|---|
| pfwc -> k2 | 0.006 / 0.007 | 0.109 / 0.175 |
| k2 -> sort | 0.013 / 0.013 | 0.117 / 0.257 |
| sort -> mat+blend | 0.009 / 0.051 | 0.101 / 0.236 |
| mat+blend N -> pfwc N+1 | 0.013 / 0.016 | 0.160 / 0.263 |
| **sum** | **0.042** | **0.486** |

No late mat start: the worst all-core sort -> mat+blend gap is 0.051 ms (view 4).

Inside mat+blend (per core, mean / min / max over cores, `ana.txt` tail):

| part | mean | min | max |
|---|---|---|---|
| mat, TRISC1 (`mat_cull_mask`) | 1.420 | 1.163 | 2.039 |
| blend, TRISCs (`tile_blend_sfpu`) | 3.667 | 3.037 | 3.938 |
| blend loader, NCRISC | 3.661 | 2.968 | 3.985 |
| mat movers, NCRISC / BRISC | 1.358 / 1.349 | | 2.039 / 1.582 |
| blend jobs per core | 10.0 | 7.3 | 12.0 |

TRISC1 idle inside mat: 0.33 ms mean, 1.17 ms on the worst core (TRISC0 `pick_job` wait 0.34 /
1.19). Filling every core's TRISC1 mat idle with its own blend work would cut the slowest
mat+blend core from 5.368 to 5.079 ms: **upper bound 0.29 ms/view** (same estimate as t506).
The critical cores are the ones whose NCRISC holds one >8192-record item (`fz_mv_big` max
1.91 ms on one mover, mean 0.15).

Per-view Gantt, views 6-8, all 110 cores: `out/ana/gantt-v6-8.png`.

Host stage timers (b2b, ms/view, `out1/run-b2b-r*.log` B2B_STAGES): project 2.24 (of which
`gather_wait` 2.28: host waits on the device), sort 0.98 (`sort_bin_emit` 1.18,
`sort_cont_prep` 0.65, `sort_publish_host` 0.31, all overlapped with device work), blend 5.73,
xview 0.10, `py_resid` 0.06. The host enqueues ahead and waits; it is not on the critical
path (device all-core idle is 0.04 ms/view).

## 3. Bytes moved per stage and roofline position

There is no NoC/DRAM byte counter in the existing tools. The table below is **derived** from
the kernels' DRAM access code (tile and record sizes) times the measured per-view counts
(N = 6.13 M, M = 1.76 M visible median (1.27-2.09), P = 2.72 M pairs median (2.05-3.19); `out1/run-counts.log`), not
measured with counters. Ceilings: p100a DRAM 448 GB/s published, **420 GB/s achieved** with
both NoCs and 215 GB/s on one NoC (`docs/hw-ceilings.md` p7, measured).

| stage | DRAM reads | DRAM writes | total | time | achieved | % of 420 | bound |
|---|---|---|---|---|---|---|---|
| pfwc | 15 fp32 streams x 5989 tiles x 4 KiB = **368 MB** (all N, every view: 9 on NCRISC, 6 on BRISC) | ~76 B x M = ~134 MB (record + depth/offs/aabb) | ~0.50 GB | 1.65 | ~300 GB/s | ~70% | TRISC SFPU wall (t484: reader input wait ~0) |
| k2 | ~12 B x M = ~21 MB | gid/tid 8 B x P = ~22 MB | ~0.04 GB | 0.65 | ~70 GB/s | ~16% | scalar pair loop |
| sort | gid/tid 8 B x P = 22 MB + record fold (bulk) | 32 B x P = ~87 MB | ~0.1-0.3 GB | 1.46 | <200 GB/s | <50% | scalar movers, small scattered writes |
| mat+blend | 32 B x P = ~87 MB (into L1; blend reads L1, `rd_l1_bulk`) | image 3 MB | ~0.09 GB | 5.33 | ~17 GB/s | ~4% | SFPU + scalar L1 sort |

Notes:
- Only pfwc moves enough data to matter: its DRAM floor at 420 GB/s is ~1.19 ms, close to
  its N-proportional fixed term (1.15-1.20 ms, sec. 4). A pfwc made 2x faster on the SFPU
  would hit DRAM next unless the scene streams shrink (bf16/fp16 or quantized cov3d, fewer
  streams) or off-view gaussians are skipped before they are read.
- Compute units: every compute kernel on the default path is SFPU. The FPU (matrix engine)
  runs only `copy_tile` (unpack to DEST) in blend and mat; no matmul or FPU eltwise op is
  used anywhere. Data movers do the sort, emit and mat-sort work in scalar RISC-V code.
- FPU vs SFPU vs data-movement share of busy time (mean core, per view): SFPU-bound TRISC
  work (pfwc 1.55 + blend 3.67 + mat TRISC1 1.42 shared with the movers) about 6.2 ms;
  scalar data-mover work on the critical path (sort 1.36, k2 0.54, mat mover sort ~0.85 where
  it is the critical unit) about 2.2-2.8 ms; FPU ~0.

## 4. View-independent vs view-dependent work

Per-view windows regressed on that view's counts (24 steady views, `out/ana/regress.txt`):

| program | fit (ms, M and P in millions) | r | fixed part of the mean |
|---|---|---|---|
| pfwc | 1.195 + 0.274 x M | 0.947 | **72%** (~1.2 ms scales with N, not with the visible set) |
| k2 | -0.021 + 0.255 x P | 0.993 | ~0 |
| sort | 0.269 + 0.455 x P | 0.994 | 18% |
| mat+blend | 1.355 + 1.521 x P | 0.998 | 25% (per-tile fixed cost of 1024 screen tiles) |
| sum of windows | 2.754 + 2.424 x P | 0.998 | 30% |

- pfwc's fixed ~1.2 ms is the per-gaussian projection, cov2d and visibility test over all
  6.13 M gaussians plus their 368 MB of reads. The inputs are view-independent (world means,
  cov3d, opacity, colours: uploaded once and pinned) but the math is per view. Only 29%
  (M/N) of it produces a visible gaussian. Over all 30 views only 2.45 M gaussians (40%)
  are ever visible, 0.84 M are visible in every view and 3.68 M (60%) never are (`xview.py`):
  a scene-level pre-filter would cut pfwc's N-sized work and reads by 60% for this benchmark,
  but not for a free-moving viewer camera. The #433/t493 chunk cull skipped ~39% of pfwc tiles but
  pfwc did not speed up and the Morton reorder slowed blend; with pfwc SFPU-bound per tile,
  a skip only pays if it removes whole tiles from the SFPU passes and keeps the gaussian order.
- The sort/K2 cost is entirely view-dependent in the data: k2 has no fixed part and sort only
  0.27 ms. Cross-view coherence (CPU model `xview.py`, same 30 views in run order, visibility
  by opacity, depth and a 3-sigma circle; M within 1% of the device's, P ~1.7x the device's
  because it uses the bounding box):

| metric, view N vs N-1 (orbit views 2-29; hero -> orbit_00 excluded, its visible Jaccard is 0.60) | mean | range |
|---|---|---|
| visible-set Jaccard | 0.935 | 0.897-0.978 |
| (gaussian, tile) pair-set Jaccard | 0.150 | 0.081-0.297 |
| pairs of view N also in view N-1 | 0.256 | 0.148-0.452 |
| adjacent depth-sorted pairs still in order | 0.660 | 0.643-0.684 |

  The visible set is stable between neighbouring views, but the pair set is not (most pairs
  move to another tile) and only about two thirds of adjacent sorted pairs keep their order,
  so re-using the previous view's sort or bins would still need most of today's sort/k2 work.
  The 30-view orbit moves the camera far between views; a live viewer at 60+ Hz with small
  steps would be much more coherent, but that is not the benchmark.

## 5. Roofline-bound vs overhead

- **DRAM-roofline bound: none.** pfwc reaches ~70% of achieved DRAM bandwidth (derived) but
  its limit is the SFPU wall. Everything else uses under half of DRAM bandwidth, mat+blend ~4%.
- **Compute bound on the unit doing the work: ~94.6%** of the period (mean core busy 8.635 of
  9.130 ms). This is "bound by the current algorithm on its unit" (SFPU or scalar RISC-V),
  not bound by a hardware peak: the FPU is idle and the scalar movers are doing sort work.
- **Imbalance: 5.4%** (0.495 ms, end spreads at 4 program boundaries; pfwc/k2/sort each ~0.1,
  mat+blend 0.15).
- **Dispatch/host overhead: 0.4%** (0.04 ms all-core idle). Host is off the critical path.

So structural levers have to cut work on the SFPU or move scalar work to faster units; idle
and overhead together are worth at most ~0.53 ms, and only with program overlap.

## Profiling flags that did not work at this commit

- `GSPLAT_TT_MATCULL_PROF=1` (per-job mat zones) does not build with the device profiler:
  `TT_FATAL: Program size (108384) too large for kernel config buffer (103424)` (`out/m0.out`).
  The per-job TRISC1 gap table is therefore missing; the per-core idle bound above comes from
  the always-on `fz_*` totals.
- `GSPLAT_TT_K2_PROF=1 GSPLAT_TT_OL_EMIT_PROF=1` (step X0) hung the device; devrun timed out at
  380 s and reset the board (`tt-smi -r`, re-initialized; the board listed normally afterwards).
  Sort and k2 are split here only to the always-on zone level.
- `python -m tracy` cannot run in the remote venv (no loguru/click/pandas): the first pass
  (`out1/T-*.log`) failed for that reason; `bench.sh` now runs the device profiler directly.

## Reproduce

- Driver: `drive.sh <rev>` (one `ttp lock p100`; `STEPS=prof2` for the 5-view profiler chunks),
  remote side `bench.sh` (b2b rounds, counts, untraced and profiled chunks).
- Analysis (Mac):
  `python3 docs/b2b-timeline-p150-223/prep506.py -o p.csv.gz out/dev-P{0,5,10,15,20,25}.csv.gz`
  `python3 docs/b2b-timeline-p150-223/steady506.py p.csv.gz --mhz 1350 --gantt g.png --gantt-views 6,7,8`
  `python3 docs/b2b-timeline-p150-223/ana506.py p.csv.gz --mhz 1350 --out out/ana`
- Cross-view model: `venv/bin/python docs/profile-t224/xview.py scenes/bicycle.ply`
  (`out/xview.txt`).

Raw: `out1/` (first pass: 4 b2b rounds, counts, warm-up), `out/` (profiler chunks
`dev-P*.csv.gz`, step logs, `T-*.log`), `out/ana/` (analysis output).
