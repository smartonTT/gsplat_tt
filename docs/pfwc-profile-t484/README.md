# t484: what bounds pfwc at iter 222 (b2b profile, analysis only)

**Answer.** pfwc is TRISC-bound on every core, and the TRISC work is the same on every core
(slowest-core TRISC wall minus writer back-pressure = 1.678 ms = mean core 1.679 ms). The single
biggest TRISC step is the **visibility + precull SFPU pass (0.62 ms, 37% of the TRISC wall)**,
then cov2d (0.46), projection (0.34) and cov_cam (0.26). Reader input wait is ~0. The writers
(survivor classify + record scatter) are idle 0.43 ms per launch on the mean core, but on the
high-survivor cores they push back on the TRISCs and end last: that costs about 0.15 ms of the
1.83 ms window (0.104 back-pressure + 0.049 writer tail). Core imbalance comes only from survivor
counts (max/mean 1.46 per core), not from compute.

## Setup

- Box: yyzo-bh-04, card p100a, worker dispatch, 11x10 = 110 compute cores (existing measurement
  reservation; no ird reserve/extend/release; bh-30 not used). All ssh with StrictHostKeyChecking=yes.
- Tree: dfb73a74 = best-iter-222 (43ff5b19) + t481 (`GSPLAT_TT_PFWC_SKIP_RGB`, default off) + this
  task's profiling tooling. Bicycle, 30 views, `render/run.py --back-to-back`.
- Drivers: `drive.sh` (one `ttp lock p100` around sync, build and all arms), `remote_prof.sh`
  (arms). Analysis on the Mac: `ana.py` (per-core windows and STEPCYC split), `bp.py` (writer
  back-pressure). Raw: `out/run-*.log`, `out/pfwc-*.csv.gz` (device CSVs cut to the pfwc rows; ana.py output is identical to the full CSVs), `out/ana.txt`, `out/bp-*.txt`.

| arm | what | result |
|---|---|---|
| u | untraced b2b, 1 check + 1 warm-up + 20 passes | **9.40 ms/view** (passes 2-21: 9.402-9.423) |
| z | Tracy device profiler, kernel zones only | 60 pfwc launches (2 passes x 30 views) |
| s | z + `GSPLAT_TT_PFWC_STEPCYC=1`, all RISCs (`STEPRISC=9`) | in-kernel split per RISC |
| sr | s + `GSPLAT_TT_PFWC_SKIP_RGB=1` | the #481 lever, same split |

STEPCYC costs nothing measurable: window s - z = +0.004 ms. The sr - z window delta (-0.056 ms)
matches #481's untraced b2b saving (-0.058 ms/view), so traced window deltas carry over to b2b
here (pfwc runs on the device critical path in b2b).

## Breakdown (arm s, ms per pfwc launch = per view; mean over 110 cores x 60 launches)

**Window:** first start -> last end **1.831 ms**, mean core end 1.722, max/mean core end 1.063.

| part | unit | ms | share of window |
|---|---|---:|---:|
| reader input wait | TRISC_0 step 0 | 0.034 (T1: 0.001) | ~0 (not a bound) |
| projection: xform, recip, depth, means | TRISC_1 steps 1-4 | 0.338 | 18% |
| cov_cam | TRISC_1 step 5 | 0.256 | 14% |
| cov2d a, c + radii | TRISC_1 step 6 | 0.290 | 16% |
| cov2d b + conic | TRISC_1 step 7 | 0.170 | 9% |
| **visibility + precull + scratch pops** | TRISC_1 step 12 | **0.621** | **34%** |
| = TRISC wall (critical unit, uniform across cores) | | **1.679** | 92% |
| writer back-pressure on high-survivor cores | TRISC step 4 excess, slowest core | +0.104 | 6% |
| writer tail after the last TRISC ends | BRISC ends last in 56/60 launches | +0.049 | 3% |

Writers (BRISC role 0 / NCRISC role 1, each takes every other chunk, 27 chunks per core):

| part | BRISC | NCRISC | note |
|---|---:|---:|---|
| wait for compute output (idle) | 0.433 | 0.448 | p90 core 0.63; slowest core only 0.17-0.20 |
| classify (cls) | 0.422 | 0.408 | ~20.5k cycles per chunk |
| survivor record scatter (rec) | 0.755 | 0.747 | 112-117 cycles per survivor; slowest core 0.98 / 1.27 |
| prefix + open + tail | 0.087 | 0.100 | |
| busy total | 1.264 | 1.254 | 0.42 ms under the TRISC wall on the mean core |
| (in the above) reader polls rd / flushes fl | 0.340 / 0.100 | 0.412 / 0.097 | |

Survivors per core: mean 15.4k, max 22.5k (max/mean 1.46); corr(writer wall, survivors) 0.62.

Notes on reading the TRISC split: the pfwc compute kernel runs with full-sync DEST
(`dst_full_sync_en`), so unpack, math and pack run each section strictly in turn and all three
TRISC walls are equal (1.667-1.679 ms). The math thread (TRISC_1) is the one whose steps are its
own work; long T0/T2 steps (T2 xform 0.745) are neighbouring sections' pack/unpack landing there
(t226). On slow cores all three TRISCs stall ~0.27 ms more in steps 3-4: that is the first
output-CB reserve of a chunk waiting on its writer. `bp.py` measures that excess against the
step-4 floor (p5, 0.070 ms): mean 0.028, p90 0.087, max 0.297 ms; 22% of core-launches wait
more than 0.05 ms.

### SKIP_RGB (arm sr) changes only the writer side

| | s | sr | delta |
|---|---:|---:|---:|
| window | 1.831 | 1.772 | -0.059 |
| mean core end | 1.722 | 1.697 | -0.025 |
| max/mean core end | 1.063 | 1.044 | -0.019 |
| TRISC_1 wall, mean core | 1.679 | 1.655 | -0.024 |
| back-pressure on slowest core | 0.104 | 0.047 | -0.057 |
| core-launches waiting > 0.05 ms | 22.3% | 6.2% | |
| writer busy (BRISC / NCRISC) | 1.264 / 1.254 | 1.210 / 1.202 | -0.05 |

So #481's 0.058 ms came from the high-survivor cores' writers, not from TRISC. With SKIP_RGB,
the window is 1.772 = TRISC 1.655 + back-pressure/tail 0.117.

## Dominant cost

The TRISC compute wall, 1.68 ms of 1.83, and inside it the **visibility + precull SFPU pass**
(0.62 ms, ~15.4k cycles per 1024-gaussian chunk). From the t226 cost model (copy_tile 154
cycles, one pack/section ~222 cycles), the step's 9 copies and 2 packs are ~2k cycles; the other
~13k cycles are SFPU instructions of the precull and vis passes (~400 per 32-lane row). The
writers have ~0.4 ms of headroom on the mean core, so TRISC cuts up to ~0.4 ms pay in full on
most cores; past that, the writers' 1.21-1.26 ms busy time becomes the floor.

## Levers (each >= 0.1 ms/view b2b expected; estimates, not measurements)

1. **Cut the visibility + precull pass** (pfwc TRISC, expected 0.12-0.25 ms/view).
   First add a STEPCYC mark between precull and vis (one device run) to split the 0.62 ms.
   Then: fold `pfwc_precull_tile` into `pfwc_vis_one` so radii, opacity and a/b/c stay in LREGs
   (drops one full DEST sweep and 3 copy_tiles), stage the vis constants in LREGs once per tile
   instead of `dst_reg[DR_VP + ...]` loads per row, and drop `noinline,noipa` on
   `pfwc_vis_one` if TRISC_1 code size allows. Must stay bit-identical (edge-exact predicate;
   tests/unit/test_vis_lever2.cpp is the lane model). A 20% cut of 13k cycles is 0.12 ms.
   **A/B: SKIP_RGB on in both arms; also report base (SKIP_RGB off) to show the combined gain.**
   Expected combined with SKIP_RGB: 0.18-0.3 ms/view.
2. **Fuse the projection steps into one SFPU section** (t226 lever X, pfwc TRISC, modeled
   0.07-0.14 ms; measured target 0.338 ms on T1). xform + recip + depth + means today are four
   sections with scratch packs and copy-backs between them. Same A/B rule: SKIP_RGB on.
3. **Cheaper survivor record scatter on the writers** (pfwc writer side; compounds with 1-2).
   rec costs 112 cycles per survivor and sets the window on high-survivor cores (back-pressure +
   tail: 0.15 ms without SKIP_RGB, 0.12 with it). Today that is under 0.1 ms on its own, so do
   it only after 1 and/or 2 land: every 0.1 ms cut from TRISC moves more cores onto the writer
   floor (writer busy 1.21 ms with SKIP_RGB vs TRISC 1.655). Options: batch consecutive
   survivors' 32 B records into one 64 B (or larger) NoC write per staged page instead of a write
   per record, and deal chunks to BRISC/NCRISC by running survivor count instead of parity.

Not levers (data says no): reader/input bandwidth (T0 input wait 0.034 ms, T1 0.001); TRISC
core imbalance (none: 1.678 vs 1.679); LPT tile deal (t439, shelved, same finding).

## Reproduce

```
ttp detach t484 -- docs/pfwc-profile-t484/drive.sh <rev>     # all arms, one ttp lock p100
python3 docs/pfwc-profile-t484/ana.py out/pfwc-z.csv.gz out/pfwc-s.csv.gz out/pfwc-sr.csv.gz
python3 docs/pfwc-profile-t484/bp.py out/pfwc-s.csv.gz
```

Profiler arms must run under `python -m tracy --dump-device-data-mid-run` (render_clean never
closes the device, so the CSV is only written by the mid-run dump), and keep the host default
`GSPLAT_TT_KCFG_EXTRA_KB` (40 KB overflows the sort program's static CBs by 22,400 B).
