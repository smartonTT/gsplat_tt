# B4' step 3b, step 1 (#553): BRISC/NCRISC spare time in fused mat+blend vs the NT=1 sort's mover work

**Verdict: GO** (mover side). In fused mat+blend each mover idles on CB waits for 3.2 ms (BRISC) and
3.6 ms (NCRISC) per core per view (median). The NT=1 sort's net mover work is 0.64 ms per mover per
core per view. Every one of the 6600 (view, core, RISC) cases fits: projected extension 0.000 ms/view,
worst per-core margin +2.43 ms. Together with #550 (TRISC 2 side, GO), step 2 (the fusion) may start.

bh-33 (p150, IRD measurement reservation), 2026-10-12 15:55-16:00 PDT, inside `ttp lock p100`.
Code: e71bdf29 (counters 2106468b) on ttp/t544-b4-step-1-measure-fused-kcfg-size-on-dev.

## Method
- `GSPLAT_TT_MV_PROF=1` (profiler builds only, off by default; `render/kernels/dataflow/mv_prof.h`)
  wraps every blocking call of the fused mat+blend movers (`matblend_brisc.cpp`, `matblend_ncrisc.cpp`,
  `reader_alpha_blend_mb_devcull.cpp`, `writer_alpha_blend.cpp`) and reports zone `mv_pc` per launch:
  total, mat phase (pre), CB waits (wcb: idle on compute or the other mover), NoC round trips (wnoc),
  bulk slab barrier (wbulk), mat-ready poll (wrdy), long-wait buckets (wcb in waits >= 1K/4K/16K
  cycles), wait count. Busy = total - pre - all waits. These are RISC-V polls (cb_wait_front,
  cb_reserve_back, noc barriers), so unlike PACK's tile_regs_wait (#550 v1) the timers see real waits.
- Default arm: 3 Tracy device captures (views 0-9, 10-19, 20-29; warmup frame dropped) =
  30 views x 110 cores = 3300 (view, core) launches per mover.
- Sort side: #550's NT=1 captures (`GSPLAT_TT_OL_TOWN_NT=1 GSPLAT_TT_OL_EMIT_PROF=1`,
  `docs/b4-step3b-gate/out/v2/dev-nt1-c*.csv.gz`), same views and cores. Sort mover gross = the
  sort_ol_{count,fill,fillb,prefix,emit} zones on that RISC; net = gross - emit's waits for the town's
  slots (wfl) - its wait for the final batch (proc).
- `ana_mv.py`: per (view, core, RISC) late = max(0, sort net - hide), hide = blend wcb (+ tail until
  the last mover on the chip ends); extension per view = max over (core, RISC).

## Numbers (ms per core per view, 1.35 GHz; median, with p10/p90)
| quantity | BRISC | NCRISC |
|---|---|---|
| fused launch total | 5.28 (4.40 / 6.03) | 5.10 (4.14 / 5.86) |
| mat phase (pre) | 1.37 | 1.36 |
| blend phase | 3.90 (3.29 / 4.40) | 3.73 (3.09 / 4.28) |
| blend busy | 0.645 (0.580 / 0.782) | 0.038 (0.031 / 0.050) |
| **blend CB wait (wcb)** | **3.229 (2.617 / 3.782)** | **3.602 (2.889 / 4.172)** |
| of which in waits >= 16K cycles | 3.225 | 3.601 |
| NoC round trips (wnoc) | 0.003 | 0.017 |
| bulk slab barrier (wbulk) | 0.000 | 0.013 |
| tail until last mover ends | 0.141 | 0.340 |
| waits per launch | 29 | 109 |
| **NT=1 sort mover net work** | **0.648 (0.521 / 0.748)** | **0.636 (0.510 / 0.726)** |
| sort mover gross (incl. town waits) | 3.44 | 3.45 |
| of which emit wait for town slots (wfl) | 2.66 | 2.71 |
| sort fill / prefix / barrier | 0.074 / 0.061 / 0.081 | 0.085 / 0 / 0 |

Projection (hide = wcb + tail, or only waits >= 16K cycles): extension 0.000 ms/view on all 30 views,
0 of 6600 (view, core, RISC) late. Serial (no hiding): 0.707 ms/view median. Tightest cores
(BRISC): (11,3) sort 0.681 vs idle 3.114, margin +2.43 ms; the 8 tightest are all BRISC, all > +2.4 ms.

What this means:
- Both movers spend almost all of blend waiting on compute. NoC traffic in blend is tiny (wnoc + wbulk
  < 0.03 ms), so the sort's emit writes should not fight blend for the NoC.
- The sort's mover gross time (3.4 ms) is mostly waiting for the town. Fused, those waits must not
  block: the movers have to poll the sort queue and the blend CBs in one loop. The binding resource
  is TRISC 2 (town 3.27 ms net vs 3.81 ms idle per core, #550), not the movers.

## Step 2 (not started in this run)
The fusion (town sort(N+1) on TRISC 2 inside mat+blend(N), mover fill/prefix/emit interleaved with the
blend reader/writer loops, host pipeline order pfwc(N+1)+k2(N+1) before mat+blend(N), flag off by
default, independent of #552's B4a flag) is a multi-run kernel change. Handed off as a `continue:`
follow-up of #553 with this design.

## Correctness
Counters are compiled only with `MV_PROF=1` in profiler builds; default builds are unchanged (every
macro empty). No new image: step 1 changes no rendering path. The image acceptance (md5 906e0435
30/30 or PSNR >= 42.4 dB, device hero + diff by eye) applies to step 2.

## Raw data
- `out/dev-def-c{0,10,20}.csv.gz` (default + MV_PROF), `out/mv-per-core.csv`,
  `out/sort-mover-nt1-per-core.csv`, `out/ana553.txt` (full analysis), logs `out/def-*.{log,out}`.
- Drivers `drive.sh`, `remote_tracy.sh`; analysis `ana_mv.py`.
