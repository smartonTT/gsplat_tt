# #428: mat stop check after #425 (CPU only, no device)

Question: after #425 (mover perm cut + NCRISC big-tile sort, iter 216, 8.043 vs 8.237 ms/view on
p100a, branch ttp/t425-cut-mover-perm-and-ncrisc-big-tile-sort- at 70645735) does the mat
phase still have a lever worth >= 0.15 ms/view?

**Verdict: stop the mat stream.** The stop rule's numbers are not met (mean idle 0.594 > 0.3,
worst-core idle excess 0.607 > 0.15), but no unshelved lever reaches 0.15 ms/view:

- The idle is the movers' own work (mostly radix sort) running ahead of TRISC1, not a
  scheduling gap that a pipeline change can close.
- The worst-core excess does not reach the frame. Blend claims tiles dynamically, so the
  mat+blend core length spread is only 0.099 ms.
- The one new lever checked here, a 2-deep per-mover job queue, replays at -0.040 ms/view.

## Data

#425's own capture: `docs/perm-noc-t425/out/dev-xvn7-c0.csv.gz`.

- Views 0-9 of the bicycle 30-view set, p100a yyzo-bh-04, 11x10 grid.
- Captured with GSPLAT_TT_MATCULL_PROF=1 and PERM_NOC=7 (the #425 default).
- Holds 152,568 mj_ zones.
- There is only one chunk (c0), so these numbers cover 10 views, not 30. #418 c0 versus #418 over
  30 views moved idle by 0.001 ms, so 10 views are enough.

## ana407 section (c): TRISC1 in the mat phase, ms/view, mean / max over cores

| metric | #413 | #418 (30 v) | #418 pk (v0-9) | **#425 (v0-9)** |
|---|---|---|---|---|
| idle outside jobs | 1.096 / 1.661 | 0.766 / 1.617 | 0.765 / 1.633 | **0.594 / 1.201** |
| worst core - mean (idle) | 0.565 | 0.851 | 0.868 | **0.607** |
| band_batch busy | 1.191 | | 1.042 | **1.026 / 1.187** |
| mat length | | | | 1.644 / 2.121 |
| start gap | | | | 0.389 / 0.538 |
| between-job gaps | | | | 0.200 / 0.664 |
| tail | | | | 0.005 / 0.020 |
| in-job idle | | | | 0.011 |

Other numbers:
- Worst-core mat length minus mean is 0.477 ms.
- Mat+blend core length is 5.414 mean / 5.513 max, so the excess is **0.099 ms**. The blend
  tile counter absorbs mat imbalance, and the frame pays roughly the mean-core idle, not the
  worst core.
- Section (a) bound if all of TRISC1's mailbox wait were filled: 0.424 ms/view.

What the movers do while TRISC1 waits:
- Start gap: sort is 69% (BRISC) and 67% (NCRISC); NCRISC rd is 25%.
- Between-job gaps: NCRISC sort 51%, BRISC perm 42%, NCRISC perm 36%.

Mover busy per view, mean ms (#425 vs #418):

| counter | NCRISC #425 | BRISC #425 | NCRISC #418 | BRISC #418 |
|---|---|---|---|---|
| sort | 0.772 | 0.852 | 0.775 | 0.840 |
| perm | 0.199 | 0.203 | 0.373 | 0.342 |
| rd | 0.101 | 0.048 | | |
| wait for TRISC (dw) | 0.377 | 0.329 | | |

Total mover busy is about 1.26 ms (NCRISC) and 1.13 ms (BRISC), against a 1.64 ms mat phase.
Each mover is busy for most of the phase, so the movers are the critical path.

## #419 gain model on the #425 trace

Files:
- Model: `model419.py`, a copy of t419 `docs/blend-interleave-model/model.py` with only env
  overrides for source, chunk and prefix added.
- Output: `out/model-t425n7/`.
- #418 control: `out/model-t418pk/`.
- The "mean of 30 views" header line in model.txt is left over from the original; here it is
  10 views.

Calibration: simulated 5.333 ms vs measured 5.446 ms.

| case (ms/view) | #425 | #418 pk c0 |
|---|---|---|
| blend interleave, best buildable (fit, wr deferred), s=1 | 0.045 | |
| blend interleave, upper bound (sub-slab + DEST spill), s=1 | 0.115 | 0.171 |
| mover gaps cut 30% (s=0.7), mover_only | 0.186 | 0.253 |
| mover gaps cut 50% (s=0.5), mover_only | 0.311 | 0.414 |

The model held up on #425:
- #425 cut idle by 22% (s~0.78), and the model predicted about 0.19 ms/view at that s.
- #425 measured -0.194 ms/view untraced.

The blend interleave (#419, shelved) has fallen further below the gate: its upper bound is now
0.115. The only route to >= 0.15 is cutting another ~30% of the mover gap time.

## The new lever checked: 2-deep per-mover job queue

Today each mover keeps one cull job in flight:

> perm i -> post i -> rd i+1 -> sort i+1 -> wait job i -> emit i -> perm i+1 (into the same slab) -> post i+1

The perm of i+1 sits between the end of cull i and the next post. The idea: permute i+1 behind
i in CB_SLAB when both fit (NCRISC 8192 rec, BRISC 6144 rec), post it, and wait for job i one
block later. This adds no L1.

`replay2deep.py` replays each core's measured mover zones and TRISC1 mj_job lengths:
- A post follows each perm/gather. Posts equal jobs on all 1100 core-views.
- TRISC1 serves jobs in arrival order, with a 1 us per-job overhead.

| replay, ms/view | mean core | worst core |
|---|---|---|
| measured T1 mat_cull_mask | 1.644 | |
| replay, today (1-deep) | 1.646 | 2.119 |
| replay, 2-deep when both slabs fit | 1.605 | 2.106 |
| replay, 2-deep, no slab cap | 1.605 | 2.106 |

Gain: **-0.040 ms/view**, and the slab cap is not the limit. Only the perm part of the
between-job gap moves; the start gap (first sort) and sort time do not. Below the gate, so not
proposed.

## Why nothing else reaches the gate

| lever | status / estimate |
|---|---|
| faster mover radix sort | Already adaptive per tile (kmin/kmax, B bits, P passes, `sort_radix_tile_algo.h`); #418 packed key\|id already landed. A further 30% sort cut is worth ~0.11 ms/view. |
| mat ramp (start gap) | #417, -0.030 measured, shelved |
| blend interleave | #419, upper bound now 0.115, shelved |
| microblock shape | #408, shelved |
| mover perm | done in #425 (perm halved) |
| 2-deep job queue | this doc, -0.040 |
| worst-core balance | frame excess only 0.099 (dynamic blend claim) |

The remaining mat idle is the price of sorting on two RISC-V movers. Removing it would need a
different sort design, such as moving the per-tile sort out of the mat phase entirely. That is a
pipeline restructure, not a mat-phase lever, and it belongs in a fresh planning pass with its
own model, not this stream.

## Reproduce

Run from the worktree root. tmp/ is not committed.

```
mkdir -p tmp/t428 && gunzip -c docs/perm-noc-t425/out/dev-xvn7-c0.csv.gz > tmp/t428/dev-c0.csv
python3 opt/profiler/stitch_device_csv.py -o tmp/t428/n7_10.csv tmp/t428/dev-c0.csv
python3 docs/xvpin-tracy/ana407.py tmp/t428/n7_10.csv --out docs/perm-noc-t425/out \
    --job-csv docs/mat-stop-check-t428/out/job-gaps-per-core-n7.csv   # sections a-c; then rc=1 in reconcile(): no untraced rounds, harmless
T419_SRC=docs/perm-noc-t425/out T419_CHUNKS=c0 T419_PREFIX=dev-xvn7- T419_CACHE=tmp/t428/n7.pkl \
    python3 docs/mat-stop-check-t428/model419.py --out docs/mat-stop-check-t428/out/model-t425n7
python3 docs/mat-stop-check-t428/replay2deep.py tmp/t428/n7_10.csv --cache tmp/t428/rp.pkl
T428_CAP_N=999999999 T428_CAP_B=999999999 python3 docs/mat-stop-check-t428/replay2deep.py tmp/t428/n7_10.csv
```
