# p150 vs p100a gap attribution (task #350, part 1 of 2, 2026-10-07)

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
