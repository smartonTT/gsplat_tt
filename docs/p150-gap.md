# p150 vs p100a gap attribution (tasks #350 + #356, 2026-10-07)

Iter 207 (best-iter-207, b28a6c76) on bh-30 p150b vs yyzo-bh-04 p100a. All bh-30 runs: tree, venv and
tt-metal build of #346 (`/localdev/smarton/p150bench/tree`, viewer's tt-metal 437bc366 read-only),
30-view bicycle bench, `GSPLAT_PER_VIEW_STAGES=1`, md5 906e0435 on 30/30 views in every run.
Scripts: `docs/p150-gap-t350/{bench_gap.sh,drive.sh}`; logs, md5 lists, mutagen samples: `docs/p150-gap-t350/out/`.

## Measured so far (bh-30, ms/view, 3 rounds per arm, interleaved)

| arm | frame (mean ± sd) | sort bin_emit | d2h | publish_host |
|---|---:|---:|---:|---:|
| U unpinned (r1-r3) | 12.924 ± 0.10 (12.977 / 12.806 / 12.990) | 2.525 (2.519 / 2.526 / 2.529) | 0.443 (0.505 / 0.317 / 0.506) | 0.332 |
| P `taskset -c 18-23` (r1-r3) | 12.867 ± 0.13 (12.740 / 13.000 / 12.860) | 2.454 (2.451 / 2.511 / 2.400) | 0.394 (0.309 / 0.478 / 0.396) | 0.326 |
| U again (window c) | 13.006 ± 0.11 (13.045 / 12.880 / 13.093) | 2.540 | 0.512 | 0.330 |
| C `GSPLAT_TT_MAT_CQ1=0` (window c) | 13.390 ± 0.06 (13.355 / 13.359 / 13.457) | 3.054 | 0.523 | 0.328 |
| p100a reference (iter 207, #346 doc) | 10.923 | 0.571 | 0.243 | 0.169 |

Findings:
1. **mutagen-agent**: busy the whole time (125-138 % CPU in every run, never idle; the box used ~5 of 96
   CPUs). Pinning our bench to 6 cores: -0.06 ms (inside noise, sd 0.1). Mutagen threads run on all 24 of
   our cores, so pinning cannot fully avoid it. Measured effect ~0 ms (±0.1).
2. **Host / PCIe**: bh-30 = 2x AMD EPYC 7352 (Zen 2, 24 cores visible, 2.3 GHz max, governor performance),
   card tenstorrent!0 on NUMA node 0, our cpuset 0-23 = node 0 (local), hugepage on N0.
   **PCIe link runs at Gen4 16 GT/s x16 (card max 32 GT/s)**. yyzo-bh-04 = Ryzen 5 7600X (Zen 4, 5.45 GHz max,
   1 NUMA node), p100a link **Gen5 32 GT/s x16**. Host-side stages run ~2x slower (publish_host 0.33 vs 0.17).
   numactl/lspci are not installed on bh-30; values come from /sys (current_link_speed, numa_node, cpulist).
3. **bin_emit is a device wait, not host work**: with CQ1 it is the blocking CQ1 read of the K2 count rows
   (`host_cq1_k2_rows`, sort_device.cpp ~2475), i.e. it ends when the tile-assign K2 finishes on device.
   With `GSPLAT_TT_MAT_CQ1=0` it waits for the whole one-launch sort instead: 3.05 ms, frame +0.38 ms.
   So the extra ~1.9 ms is device time (K2 / earlier queued programs) or dispatch stalls, still to split by Tracy.
4. **Firmware**: bh-30 card 19.13.2.0, aiclk 1350 MHz (same clock as p100a 19.12.0, 1350 MHz).
5. **Build**: viewer's tt-metal build has ENABLE_TRACY=ON (same commit 437bc366 as the p100a build).
6. **Mover speed table** (`render/host/sort_mover_speed.h`, `kMoverSpeedP150`) was measured on a **p100a**
   (t166-p2f, yyzo-bh-07) by physical NoC (x, y); on the p150 the 11x10 grid maps to other physical columns,
   so the table may mis-weight the emit/K2 split. Not yet measured (needs the EMIT=1 capture).

Not done yet (part 2): Tracy capture (failed: `python -m tracy` cannot import loguru from the viewer venv),
device-zone comparison with opt/profiler/ttw-207, emit_cores.py --weights on p150, final attribution table.

# Part 2 (task #356): Tracy attribution

Tracy fixed by installing loguru, click, pandas, pyyaml, seaborn into the bench venv only
(`/localdev/smarton/p150bench/tree/.venv`; the viewer venv was not touched). Two captures on bh-30
(iter 207, same tree): `drive.sh tracy T` (31 frames) and `REMOTE_ENV=EMIT=1 drive.sh tracy E` (per-mover
emit counters). Reference: `opt/profiler/ttw-207` (p100a). Profiler on, so absolute ms are ~0.1-0.2 high on
both boards; only deltas are used.

## Result

| cause | measured ms/view | ours to fix? | proposed fix |
|---|---:|---|---|
| Emit imbalance: `kMoverSpeedP150` was measured on a p100a; on the p150 rows y=2-4 are slow (emitB 2.2-2.7 ms vs 1.3-1.9 for y>=5), last mover mostly (11,3) BRISC | **+0.98** | yes | per-board mover table re-derived on bh-30 (`out/reweight-E.txt`), selected by board type; predicted emit 2.78 -> 1.74 ms (**-1.0**) |
| emit -> mat idle gap: the CQ1 read of the K2 count rows (`host_cq1_k2_rows`) is queued behind the emit traffic and ends only when the emit ends (p150 5.79 ms vs p100a 3.73 ms), then host totals + CQ1 bridge run while the device idles | **+0.82** (gap 0.84 vs 0.016); of it host CPU 0.70 vs 0.29 = +0.41 | yes | issue the K2-rows read right after the K2 event (merge it into the `host_cq1_proj_m` read or read before emit ramps), or compute the totals on device so the mat does not wait on the host |
| Uniform per-mover emit slowdown (mean per mover 1.78 vs 1.18 ms) | **+0.60** | unclear | part may go away with rebalancing (less NoC contention at the slow rows); rest: NoC/DRAM placement or firmware 19.13.2. Needs an A/B after the table fix |
| pfwc + tile-assign K2 faster on the p150 | **-0.26** | n/a | none |
| mat + blend windows | 0.00 (2.83 and same blend on both) | n/a | none |
| **device span total** (pfwc start -> blend end) | **+2.15** (12.66 vs 10.51) | | |
| host finish after device (d2h/publish, slower CPU, PCIe Gen4) | ~-0.1 to +0.1 (frame end - device end: 0.15 vs 0.11) | partly | see part 1 |
| mutagen-agent, pinning, build flags, firmware clock | ~0 | n/a | none |
| **sum** | **~2.1** (measured frame gap 2.0) | | |

Ranked fixes: (1) p150 mover table, ~-1.0 ms; (2) early K2-rows read / device totals, up to -0.8 ms;
(3) find the residual +0.6 ms emit slowdown (A/B after 1). Together 1+2 should bring the p150 to about
the p100a (~11 ms/view).

## Evidence

Device timeline, ms from the first pfwc start, mean over frames 1-30 (`docs/p150-gap-t356/timeline.py`):

| stage | p100a ttw-207 | bh-30 p150 T |
|---|---:|---:|
| pfwc end | 1.940 | 1.834 |
| K2 (k2_rows) end | 2.982 | 2.726 |
| emit start-end (window) | 3.095-4.432 (1.338) | 2.844-5.764 (2.920) |
| mat start (gap after emit) | 4.448 (0.016) | 6.604 (0.840) |
| mat window | 2.83 | 2.83 |
| blend end | 10.510 | 12.664 |

Host timeline, ms from the first EnqueueProgram:

| zone | p100a | bh-30 p150 |
|---|---:|---:|
| host_cq1_proj_m | 0.139-3.020 | 0.313-2.819 |
| host_cq1_k2_rows | 3.255-3.734 (0.48) | 3.415-5.785 (2.37) |
| bridge end | 4.021 | 6.481 |
| finish_blend end | 10.622 | 12.817 |

- Emit: per-mover mean 1.18 (p100a) vs 1.78 (p150); window - mean (imbalance) 0.16 vs 1.14.
  `sort_ol_town` tracks the emit (2.75 vs 1.18 makespan).
- `program_gaps.py` (T): one idle gap per view between town end and mat start, 0.819 ms.
- Core map: the p150 movers sit at physical x=1-6, 11-15, y=2-11, the same coordinates as
  `kMoverSpeedP150`, so the table applies coordinate-wise but its speeds are wrong for this board.
  `emit_cores.py --weights` divides by record counts, which are 0 in iter-207 EMIT captures (ZeroDivisionError);
  `reweight.py` derives the weights from per-core emit time instead (v = s/T, mean 1000).
- tt-metal build: bh-30 viewer build and yyzo-bh-04 both at 437bc366439; CMakeCache identical
  (Release, -O3, g++-12, ENABLE_TRACY=ON) apart from SITE (`out/cmakecache-bh30-vs-bh04.diff`). Not a cause.
- Screenshot: r1-U hero (device, p150, iter 207) vs `benchmarks/reference_v2/hero.png`: PSNR 42.51 dB,
  max abs diff 46, 0.82 % of pixels > 8. Diff image checked by eye: no tile seams or 32-px block artifacts,
  differences only on fine edges (spokes, foliage). `shot/{hero.png,hero_diff10.png,psnr.txt}`.

Files: `docs/p150-gap-t356/` (`timeline.py`, `reweight.py`, `out/` zone, gap, emit, timeline and
CMakeCache outputs, `prof/` gzipped device CSVs and csvexport dumps, `shot/`).
