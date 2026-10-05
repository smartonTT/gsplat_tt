# Task #197: where pfwc's ~2.8-3.0 ms/view goes (default chain, PFWC_FUSE=1)

Board: yyzo-bh-07 (Blackhole p100a). Bicycle 1024x1024, views 0:4 (5 pfwc launches:
JIT warm-up of view 0, then views 0-3). Code: dd59fca/5446d5e (tip of the t197 branch,
pfwc code = default chain plus the counters below, which are off by default).

## Method

`GSPLAT_TT_PFWC_STEPCYC=1|2` (default 0) adds wall-clock counters (1350 MHz) to the three
pfwc kernels. Each RISC sums the cycles between step boundaries over its chunks and, at
kernel end, records them as Tracy timestamped data (`pfwc_pc` on TRISC0/1/2, `pfwc_pr`
on NCRISC, `pfwc_pw` on BRISC; value = (index << 32) | cycles).
`GSPLAT_TT_PFWC_STEPRISC` picks the RISC(s) (0-2 TRISC, 3 reader, 4 writer, 9 all; default 9).
`=2` also splits cov_cam by call type (copy_tile, mul_unary, add_binary, acquire, pack).

DPRINT was tried first and does not fit: with DPRINT on, the fused pfwc program is
71.6-77.4 KB against a 70656 B kernel config buffer, even with only one instrumented RISC
(`out/failed-allriscs/`). The Tracy capture uses `GSPLAT_TT_KCFG_EXTRA_KB=10`.

Files: `remote_prof.sh` (on the board, under `ttp lock p100` via `drive.sh`),
`pc_split.py` (per-RISC means + slowest core), `core_crit.py` (per launch, per core),
`out/pc2-r9-dev.csv.gz` (device CSV), `out/pc2-r9-pc_split.txt`.

## Results (110 cores, 54.4 chunks of 1024 Gaussians per core per view)

pfwc zone makespan per view: 3.006 (view 0), 2.738, 2.738, 2.755 ms. Per-core zone
mean 2.63-2.80 ms. The program ends with the slowest core.

### Per RISC (mean over cores and launches, ms per view)

| RISC | wall | busy | waiting |
|---|---:|---:|---:|
| NCRISC reader | 2.600 | DRAM barrier 0.098 | **CB reserve (compute back-pressure) 2.467** |
| BRISC writer | 2.739 | classify 0.729, records 1.270, rest 0.036 | compute tiles 0.704 |
| TRISC1 math | 2.703 | see below | — |

The reader is idle 95% of the time: DRAM input is not a cost. The writer is busy 2.04 ms
on average (1.6-2.6 ms per core, scales with pairs). The math thread is the floor.

### Math thread (TRISC1) by step, ms per view

| step | ms | note |
|---|---:|---|
| 1 transform (R·means + t) | 0.191 | |
| 2 recip, 3 depth | 0.063 | |
| 4 means (mean_x/y) | 0.188 | includes writer back-pressure (see below) |
| 5 cov_cam (6 entries) | **0.664** | copy_tile 0.223 (36 copies), mul_unary 0.232 (36), add_binary 0.155 (30) |
| 6-8 cov2d a, b, c | 0.496 | 0.173 / 0.187 / 0.136 |
| 9 conic | 0.073 | |
| 10-11 radii x, y (recompute a, c) | 0.414 | 0.208 / 0.206 |
| 12 visibility + tile rect + precull + pops | **0.613** | |
| total | 2.703 | |

The math wall minus steps 3-4 is 2.46-2.54 ms on every core: a flat compute floor.
Steps 3-4 are where the output CBs fill when the writer is behind: 0.09 ms on light
cores, up to 0.23 ms on the heaviest. Per core, wall correlates with pairs (r = 0.87-0.93);
the slowest core (2-5) has ~19% more pairs than the mean.

Unpack and pack threads show the same totals; their step times mostly wait on math
(e.g. TRISC2 spends 0.547 of its 0.586 ms cov_cam waiting in pack, TRISC0 0.610 in copy).

## Answer: dominant cost

**SFPU-side compute on the math thread (~2.5 ms/view of ~2.75)**, spread over many small
tile ops: each value goes DEST→L1 scratch→DEST between steps, cov_cam copies each input
tile 6 times (36 copy_tile, ~150 cycles each), and radii recompute cov2d a and c.
Reader stalls are nil. The writer (scalar classify + record emission, ~2.0 ms) is the
second limit: on heavy cores it already stalls compute by up to ~0.2 ms, and it caps any
compute-only gain.

## Levers (upper bounds, traced ms/view; gate 0.3 untraced)

1. **Single-pass cov_cam** (6 copy_tile + one SFPU kernel doing the 6x6 per-view linear
   map with 36 MADs per element in LREGs, in place in DEST). Replaces 36 copies + 66
   tile ops (0.66 ms) with ~0.15-0.2 ms. Saves up to ~0.45 ms of math; critical path
   gains ~0.3-0.45 because heavy cores turn writer-bound (writer busy up to 2.2 ms on
   views 1-3). Smallest, md5 risk: fp32 MAD order changes, so check md5 or PSNR.
2. **Fused SFPU cov2d kernel** (cov_cam + a + b + c + conic + radii in one SFPU pass,
   T = J·W then T Σ Tᵀ, ~40 MADs per element, no recompute; stage through DEST rows when
   8 LREGs are not enough). Replaces ~1.65 ms of math with ~0.2-0.3. Alone it is capped by
   the writer: pfwc ~2.75 → ~2.2 (saving ~0.5-0.6 ms).
3. **Split the writer's scalar work onto the idle NCRISC** (reader waits 2.47 ms of 2.60):
   e.g. NCRISC classifies/records odd chunks, BRISC even ones, each with its own record
   staging. Alone ≤ ~0.2 ms (heavy-core back-pressure only). With lever 2 it removes the
   writer cap: pfwc floor becomes max(compute ~1.0-1.2, writer/2 ~1.1-1.3), i.e. up to
   ~1.4-1.5 ms/view saved for 2+3 together.

Order: 1 first (cheap, measurable), then 2 as its extension, 3 once compute drops below
the writer. All estimates come from the counters above and SFPU instruction counts. None
has been measured yet.

## Caveats

- The counters add ~2 wall-clock reads per step and ~200 per chunk in cov_cam (=2), which
  is small but non-zero. The pfwc makespans here (2.74-3.01) match t194's 30-view 2.98 mean.
- p100a, not p150. Views 0-3 only; view 0 is heavier (3.0 ms) than views 1-3 (2.74-2.76).
- The 5 launches include the warm-up (identical to view 0), so view 0 counts twice in the
  means above.
