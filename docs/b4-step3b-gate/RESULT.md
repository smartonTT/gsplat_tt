# B4' step 3b gate (#550): TRISC 2 idle during fused mat+blend vs NT=1 town sort

**Verdict: GO** (TRISC 2 side). Projected b2b saving about 1.3 ms/view (median), conservative lower
bound about 0.7 ms/view, both above the 0.4 ms/view build gate. One part stays unmeasured: the sort's
BRISC/NCRISC (mover) phases. Step 3b should measure that first (see Risks).

bh-33 (p150, IRD measurement reservation), 2026-10-12, all device runs inside `ttp lock p100`.
Code: daa86bf7 on ttp/t550 (from ttp/t544 @bedc35ac). Default config b2b sanity: 9.000 ms/view
(`out/v1/b2b.out`, pass 13).

## Method
- `GSPLAT_TT_T2_PROF=1` (profiler builds only) adds TRISC 2 (PACK) wait counters to the fused blend
  (`render/kernels/compute/alpha_blend_compute_mb.cpp`, zone `t2_pc`).
- v1 was wrong: `tile_regs_wait()` on PACK is a Tensix SEMWAIT, so the RISC-V never blocks there. The
  real wait for MATH hid in `pack_tile` as instruction-FIFO backpressure and counted as busy (v1: busy
  3.69 of 3.73 ms, idle 0.17 ms). That would have been a false NO-GO. v1 data kept in `out/v1`.
- v2 (`T2_MATHWAIT`): `tensix_sync` drains the pipe (WDRN), then the RISC-V polls the MATH_PACK
  semaphore (WMATH). So the RISC-V now waits where the hardware waits, and the timers see it.
- Default arm: 3 Tracy device captures (views 0-9, 10-19, 20-29; warmup frame dropped) =
  30 views x 110 cores = 3300 (view, core) launches.
- NT=1 arm: `GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1`, town zone `town_pc` on TRISC 2,
  net work = proc - wait-for-fill - wait-for-queue. Same 30 views, same 110 cores.
- `ana.py` compares per (view, core): the NT=1 town work vs TRISC 2 idle in blend(N) (MATH waits,
  optionally only waits >= 1K/4K/16K/64K cycles, plus the tail until the last core ends). Core c
  ends late by max(0, sort_c - hide_c); extension = max over cores per view.

## Numbers (ms per core per view, 1.35 GHz)
| quantity | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| blend TRISC 2 total | 2.024 | 3.115 | 3.731 | 4.278 | 4.745 |
| blend TRISC 2 busy (issuing) | 0.008 | 0.010 | 0.038 | 0.081 | 0.141 |
| blend TRISC 2 wait for MATH (wmath) | 1.995 | 2.980 | 3.651 | 4.205 | 4.553 |
| blend TRISC 2 wait, all kinds | 2.013 | 3.072 | 3.679 | 4.225 | 4.677 |
| of which in waits >= 16K cycles | 2.005 | 3.048 | 3.655 | 4.190 | 4.608 |
| of which in waits >= 64K cycles | 1.916 | 2.786 | 3.360 | 3.918 | 4.408 |
| tail (idle until last core ends) | 0.000 | 0.047 | 0.137 | 0.250 | 0.312 |
| **per-core idle (wait+tail), median over views** | **3.246** | **3.698** | **3.813** | 3.857 | 3.912 |
| **NT=1 town net work** | **2.046** | **2.533** | **3.270** | 3.822 | 4.045 |
| NT=1 town per 16-record batch (us) | 15.9 | 16.6 | 16.9 | 17.1 | 17.3 |

Counts per core per view (median): 231 MATH waits, 9 tiles, 43 record batches.
Load imbalance: blend idle per core is tight (3.25-3.91 ms); town work per core spreads 2.0-4.0 ms.
The per-(view, core) projection covers that: no core's town work exceeds its own idle time.

Sort program span (default NT=3, first start to last end): median 1.484, min 1.202, p90 1.663 ms.

## Projection
| idle usable by sort | extension of mat+blend(N), ms/view median (max) | hidden fraction |
|---|---|---|
| all MATH waits | 0.000 (0.000) | 1.00 |
| only waits >= 16K cycles (12 us) | 0.000 (0.000) | 1.00 |
| only waits >= 64K cycles (49 us) | 0.163 (0.456) | 1.00 |
| none (serial) | 3.433 (4.045) | 0.00 |

- Saving bound = removed sort program span - extension: 1.484 - 0.163 = **about 1.3 ms/view** (median,
  using only the long waits).
- Conservative: also assume the sort's mover phases (fill 0.14 + prefix 0.09 + barrier 0.15 =
  0.37 ms span) do not hide, and every pack is late by one yield quantum (231 packs x 1 us = 0.23 ms):
  1.48 - 0.16 - 0.37 - 0.23 = **about 0.7 ms/view**. Still above the 0.4 gate.

## Risks for 3b
1. **Pack latency.** TRISC 2 issues pack only when MATH is done. The town must yield often: poll the
   MATH_PACK semaphore every few records (one record is about 180 cycles), not per 16-record batch
   (17 us; 231 x 17 us would cost too much). The dst double buffer absorbs short delays.
2. **Mover side not measured.** The town feeds BRISC/NCRISC emit (span 1.38 ms) and needs fill/prefix.
   During blend the movers run rd_l1_bulk / tile_blend_load. Their spare time is unknown. First step
   of 3b: per-core BRISC/NCRISC busy counters in blend, same method.
3. **L1 and ordering.** Step 0 (#543) showed one sort stream fits L1 with a phased layout. The pipeline
   needs pfwc(N+1)+k2(N+1) before mat+blend(N). L1 port contention from the town's scalar loads is not
   measured.

## Correctness and image
Profiling off: 30-view sweep md5 **906e0435** (30/30, 0 views differ), hero golden_match=true,
max LSB 0, PSNR **42.51 dB** vs benchmarks/reference_v2/hero.png (`out/v1/shot.out`).
Device hero `opt/metal-screenshots/ttw-t550/hero.png`, diff `hero_diff10.png`: checked by eye, no tile
seams or block artifacts; the diff shows only thin edges.

## Raw data
- `out/v2/dev-def-c{0,10,20}.csv.gz` (default + T2_PROF, v2 counters), `out/v2/dev-nt1-c*.csv.gz`
  (NT=1 town, from v1 run 1520), `out/v2/t2-per-core.csv`, `out/v2/town-nt1-per-core.csv`,
  `out/v2/ana.out` (full analysis), logs `out/v2/def-*.{log,out}`.
- v1 (wrong T2 attribution, kept for the record): `out/v1/`.
- Drivers: `drive.sh`, `remote_tracy.sh`; analysis `ana.py`.
