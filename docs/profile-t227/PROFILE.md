# t595 (F0): b2b device profile of best-iter-227 on bh-33, and the step-0 bounds of plan-3 specs B and C

Task #595, 2026-10-13 (PDT). Build best-iter-227 (445f9e40: MAT_EARLY_CULL, opacity prune, B4a
fold, BLEND_EXP1 all on) plus this task's profiling scripts only (`render/run.py` gains the
`GSPLAT_VIEW_COUNTS=1` print from t528; no render code changed). Bicycle, 30 views,
`render/run.py --back-to-back`. Box: bh-33, existing measurement reservation, **card p100a**,
11x10 = 110 worker cores, 1350 MHz. No reserve/extend/release, no bh-30, every ssh with
StrictHostKeyChecking=yes, every device step under `ttp lock p100`.

Replaces the "Where the 8.45 ms go (estimate)" table in docs/structural-plan-3/PLAN.md.

## Headline: b2b ms/view (untraced, defaults)

| round | passes | ms/view median | raw_md5 |
|---|---|---|---|
| r0 | 20 | 8.444 | 1d372b94 |
| r1 | 20 | 8.457 | 1d372b94 |
| r2 | 20 | 8.439 | 1d372b94 |
| r3 | 20 | 8.457 | 1d372b94 |

Median of 4 rounds **8.451 ms/view** (118.3 FPS). 30-view `--dump-views` sweep md5 **077caaba**
(30/30, the golden; `out/md5-dump.txt` without the hero_clean line). Untraced chunk U0 and
profiled chunk P0 give the same raw_md5 (57fe026f): the profiler does not change the image.

## How the profile was taken (and what did not fit)

The default build cannot be device-profiled: with the B4a fold the traced sort/matblend program
is 106528 B, over the profiler's kernel-config buffer (103424 B). `GSPLAT_TT_KCFG_EXTRA_KB=36`
fits it, but then the static circular buffers overflow L1 by 1920 B. So the P chunks ran with
**`GSPLAT_TT_B4A_FOLD=0`** (K2 as its own program: the t224 program shape, 4 programs per view).
The fold only moves K2 into the mat+blend launch; the other programs' code is the same.

The `GSPLAT_TT_MATCULL_PROF=1` chunks (per-item `mat_*` zones that `bound508.py` reads) do not
fit on 227 at all: 107632 B > 103424 B at the default kcfg, and at +37 KB the static CBs overflow
L1 by 2944 B (`out/T-M0.log`). Spec B's rebound therefore uses the always-on `fz_*` fill-zone
markers (`bounds595.py`), which the plan allows ("or its equivalent on F0's bh-33 capture").

Six 5-view chunks P0..P25 (`out/dev-P*.csv.gz`), stitched by `prep506.py` (last 5 views of
each chunk), analysed by `ana506.py` (`out/ana-P.txt`), `steady506.py` (`out/steady-P.txt`,
`out/gantt-P.png`), `bounds595.py` (`out/bounds-P.txt`), `fit595.py` (`out/fit-P.txt`).

**Traced vs untraced.** Traced steady period (FOLD=0): mean 8.831 ms. Untraced default (FOLD=1):
8.451. The difference, 0.38 ms, is the fold's measured gain (iter 226 A/B: -0.375 ms/view), so
the profiler adds ~0 here, as at t224 (traced 9.130 vs untraced 9.131).

## 1. Measured split (ms per view, steady views, device clock)

Window = first core start to last core end. Mean core = average over 110 cores of the core's own
program length. t224 column: docs/profile-t224 (best-iter-224, same box and method).

| program | window 227 | mean core | slowest core | end spread | window t224 | change |
|---|---|---|---|---|---|---|
| pfwc | **1.523** | 1.419 | 1.525 | 0.103 | 1.654 | -0.131 (opacity prune) |
| k2 (own program, FOLD=0) | **0.643** | 0.549 | 0.650 | 0.095 | 0.647 | -0.004 |
| sort | **1.436** | 1.351 | 1.447 | 0.083 | 1.459 | -0.023 |
| mat+blend | **5.189** | 5.033 | 5.235 | 0.157 | 5.330 | -0.141 (EXP1, prune) |
| all-core gaps (4 go signals) | 0.040 | | | | 0.040 | 0 |
| **steady period** | **8.831** | sum 8.353 | | idle/core 0.478 | 9.130 | -0.299 |

With the fold on (the shipped default), K2 runs inside the mat+blend launch and the period is
8.451: K2's net cost there is about 0.643 - 0.38 = **~0.27 ms** (derived from the A/B, not
traced). Shipped split, rounded: **pfwc 1.52, sort 1.44, mat+blend incl. folded K2 ~5.46,
gaps ~0.03 = 8.45 ms/view**.

Inside mat+blend (per core, mean / min / max over cores, mean over views):

| part | mean | min | max |
|---|---|---|---|
| mat (TRISC0/1/2, `mat_cull_mask`) | 1.420 | 1.162 | 2.040 |
| blend (TRISCs, `tile_blend_sfpu`) | 3.514 | 2.881 | 3.781 |
| blend reader NCRISC (`tile_blend_load`) | 3.512 | 2.817 | 3.839 |
| mover NCRISC (`sort_subchunk_mat`) | 1.358 | 1.041 | 2.040 |
| mover BRISC (`sort_subchunk_mat`) | 1.349 | 1.090 | 1.581 |
| writer BRISC (scm end -> kernel end) | 3.728 | 3.550 | 4.115 |
| blend jobs per core | 10.0 | 7.3 | 12.1 |

Mover detail (`fz_mv_*`, NCRISC / BRISC mean): sort 0.854 / 0.880, permute 0.211 / 0.213,
reads 0.111 / 0.050, done-waits 0.010 / 0.183 (BRISC max 0.408). Big items (>8192 records)
on ~9.7 cores per view.

TRISC1 in mat (`fz_m_*`): total 1.420 mean / 2.039 max, **band_batch 1.077 / 1.237**, copy_tile
0.011, mailbox wait 0; idle bound (total - band - copy) 0.331 mean. T0 pick_job wait 0.341.
Filling every core's T1 mat idle with its own blend work would shorten the slowest core's
mat+blend 5.235 -> 4.937 (upper bound 0.298 ms/view; same lever as spec B).

Boundaries (steady views): all-core idle 0.006 / 0.013 / 0.008 / 0.013 ms
(pfwc->k2, k2->sort, sort->mat+blend, mat+blend->pfwc N+1); per-core idle at those boundaries
0.108 / 0.108 / 0.091 / 0.171 (end spread, sum 0.479). No program overlap on any core.

## 2. M/P fits (t224 method: 24 views, first view of each chunk dropped; M visible, P pairs, in millions)

| program | fit 227 (ms) | r | fixed share | fit t224 |
|---|---|---|---|---|
| pfwc | 1.063 + 0.274 x M | 0.972 | 70% | 1.195 + 0.274 x M |
| k2 | -0.012 + 0.250 x P | 0.992 | ~0 | -0.021 + 0.255 x P |
| sort | 0.329 + 0.424 x P | 0.995 | 23% | 0.269 + 0.455 x P |
| mat+blend | 1.421 + 1.443 x P | 0.998 | 27% | 1.355 + 1.521 x P |
| sum of windows | 2.754 + 2.310 x P | 0.998 | 31% | 2.754 + 2.424 x P |

Mean M 1.675 M, P 2.614 M per view (N 6.13 M before the prune). pfwc's fixed part fell 0.13 ms
(the static opacity prune drops gaussians before the per-view math); its per-visible slope is
unchanged. mat+blend's per-pair slope fell 5% (EXP1 cut blend SFPU ops 27 -> 24); its fixed
part (per-tile cost of 1024 screen tiles) did not move. Still ~30% of the frame is fixed cost.

## 3. Step-0 bounds of plan-3 specs B and C (`out/bounds-P.txt`)

### Spec B (mat fill jobs + big-tile split + mover-cap rebalance): **killed**

Mat makespan per view (max over cores, mean over 30 views): today 2.040 ms; with every mover
done-wait removed and big items free (combined ceiling, t508's "no-waits+big-free") 1.592.

| form of the combined ceiling | ms/view b2b |
|---|---|
| t508 model ms 0.448 x 0.28 (t335 factor) | 0.125 |
| t508 model ms 0.448 x 0.40 (t508 factor) | 0.179 |
| balanced: freed mat time spread over all cores by pick_job (mean per-core saving) | 0.129 |
| direct: slowest core's mat+blend end, each core shortened by its own saving | 0.020 |
| fill jobs alone (done-waits removed, big items as today) | 0.000 |

Kill line: combined ceiling < 0.15 ms/view. Three of the four forms are under it; only the
generous x0.40 factor clears it, by 0.03. The ceiling also shrank 31% since t508 (0.652 -> 0.448
model ms) because early-cull shortened the mat. Judgment: **kill B**. The realized fraction of
earlier mat levers was well under their ceilings, so a 0.13-0.18 ceiling cannot reach the 0.1
build gate after three coupled changes.

### Spec C (mat band cull on the FPU): **killed**

Saving bound = min(band_batch on the critical core, critical-core TRISC1 - mover time), the
critical core being the one whose mat+blend ends last (a big-item core in 47% of views).

| quantity (ms, mean over views) | critical core | mean core | max core |
|---|---|---|---|
| TRISC1 mat (`mat_cull_mask`) | 1.684 | 1.420 | 2.040 |
| band_batch MATH (`fz_m_band`) | 1.008 | 1.077 | 1.237 |
| mover work (max of the two, span - done-waits) | 1.647 | 1.350 | 2.040 |
| **saving = min(band, T1 - mover)** | **0.037** | 0.071 | |

Makespan bound with that saving on every core: 5.235 -> 5.227 (0.009 ms/view). Kill line < 0.2.
band_batch is a large share of TRISC1 (60% on the critical core, 76% on the mean core), but the movers (scalar sort + gather) end within
0.04 ms of TRISC1 on the critical cores, so a faster band test leaves TRISC1 waiting for them.
**Kill C.** The mat's real bound is the mover sort (0.85-0.88 ms per mover), not the band test.

## 4. What this says for the next levers

- mat+blend is 59% of the frame (5.19 of 8.83 traced; ~5.46 of 8.45 shipped incl. K2). Its blend
  part (3.51 mean core) is SFPU-bound with a 1.42 ms fixed per-tile share; the mat part is
  mover-bound on the slowest cores.
- pfwc (1.52) still has 70% fixed cost over all gaussians: spec A (survivor compaction) is the
  plan's next candidate and is not touched by this profile's kills.
- sort (1.44) is scalar-mover-bound (`sort_ol_emit` 1.22 / `sort_ol_town` 1.11 busy).
- Load imbalance (end spread) is 0.48 ms per core per view; all-core dispatch gaps are 0.04.

## Files

`drive.sh` (run 1: b2b, counts, dump, U0), `drive2.sh` (run 3: P chunks, FOLD=0), `drive3.sh`
(run 4: M chunks at +37 KB, failed on L1), `bench.sh`, analysis scripts above, `out/` logs and
per-chunk device CSVs. Regenerate the stitched CSV with
`ANA_PROGS=pfwc,k2,sort,mat+blend python3 prep506.py -o out/stitched-P.csv.gz out/dev-P{0,5,10,15,20,25}.csv.gz`.
