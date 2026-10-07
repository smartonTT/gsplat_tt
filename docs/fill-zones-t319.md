# t319: fill zones on the MATCULL_TRISC_FILL path (iter 207 defaults)

Gating measurement from `docs/checkpoint-t311.md`: where does the mat phase go with
`GSPLAT_TT_MATCULL_TRISC_FILL` on? Is TRISC0 the bound (H1), or do the movers wait on coarse
jobs (H2)?

- Code: `fz_*` counters under `PROFILE_KERNEL` only, in `sort_subchunk_materialize.cpp` (movers)
  and `mat_cull_compute.cpp` (TRISC0/1/2). Report: `opt/profiler/fill_zones.py`
  (test `tests/test_fill_zones.py`).
- Box: yyzo-bh-04 p100a (IRD job 244892), commit d09e37d4, bicycle 30 views at 1024x1024,
  defaults. All device work ran under one `ttp lock p100`.

## Untraced run (bit-identity check)

| run | ms/view | md5 list | views |
|---|---:|---|---|
| untraced r1 | 10.988 | 906e0435 | 30/30, the same as iter 207 |
| screenshot run | 10.903 | 906e0435 | 30/30 |

The untraced build is unchanged: md5 matches iter 207 (#315). 10.99 / 10.90 ms/view against
iter 207's 10.907 on the same box is within run noise (d2h 0.319 vs 0.237).

Screenshot: `opt/metal-screenshots/t319-fill-zones/hero.png` (rendered on device), diff
`hero_diff10.png`. PSNR 42.51 dB against `benchmarks/reference_v2/hero.png`, the same as iter
207. I checked the hero and the diff by eye: no tile seams and no blocky or empty tiles. The
diff (x10) shows only edge detail (spokes, foliage). The golden match (8-bit,
max 0 LSB) is a separate badge.

## Tracy capture (defaults, 30 views)

It completed without a hang (no ARC timeout, no reset needed). `opt/profiler/t319-fz/`:
`zones.txt`, `fill.txt` (per view: mean/max over 110 cores, then mean over 29 views) and
`percore.csv` (per core, mean over views).

| metric (ms/view unless noted) | mean | max |
|---|---:|---:|
| mat end (= mat length) | 2.292 | 2.834 |
| NCRISC mover busy | 2.289 | 2.833 |
| NCRISC mover done-wait | **0.355** | 0.618 |
| NCRISC mover read / sort / permute | 0.102 / 1.250 / 0.370 | |
| NCRISC mover big-tile item (>8192 records) | 0.188 | **2.305** |
| BRISC mover busy | 1.962 | 2.179 |
| BRISC mover done-wait | 0.188 | 0.412 |
| BRISC mover read / sort / permute | 0.049 / 1.343 / 0.358 | |
| TRISC0 fill busy | 1.142 | 1.295 |
| TRISC0 idle (pick_job wait) | 1.109 | 1.665 |
| TRISC0 busy fraction | **0.518** | 0.612 |
| TRISC0 fill cycles/record | **64.5** | 76.3 |
| TRISC1 band_batch busy | 1.187 | 1.339 |
| TRISC1 idle | 1.104 | 1.661 |
| TRISC2 busy (pack + patch) | 0.340 | 0.397 |
| TRISC2 waits (mostly patch wait) | 0.876 | 0.982 |
| TRISC2 patch cycles/record | 18.7 | 18.7 |
| pfwc tail (max − mean core end) | 0.191 | |
| mat tail (max − mean core end) | 0.542 | |

Mover fill cost before the change: t304 measured 75 cycles/record for `cull_slab` (fill and patch
together; t304 did not split them). On the TRISCs it is now 64.5 (fill) + 18.7 (patch) = 83
cycles/record, a ratio of 1.11 against the 0.83 that t304 assumed. Fill alone is probably about the
mover's speed, not faster.

### Critical cores (per-core mean over views, `percore.csv`)

| core | mat end | NC busy | NC big-tile | NC done-wait | BR busy | BR done-wait | TRISC0 busy | end without done-wait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-2 | 2.675 | 2.675 | 2.304 | 0.303 | 1.820 | 0.054 | 0.38 | 2.372 |
| 1-2 | 2.669 | 2.669 | 2.298 | 0.303 | 1.827 | 0.058 | 0.38 | 2.366 |
| 14-2 | 2.631 | 2.627 | 0.642 | 0.452 | 1.804 | 0.043 | 0.44 | 2.175 |
| 5-3 | 2.578 | 2.578 | 0 | 0.549 | 1.970 | 0.154 | 0.48 | 2.030 |
| 6-3 | 2.549 | 2.549 | 0 | 0.541 | 1.979 | 0.184 | 0.49 | 2.008 |

The NCRISC mover sets the mat end on every critical core. "End without done-wait" is
max over the two movers of (busy − done-wait). Over all 110 cores, removing the done-wait moves
the mean core end from 2.292 to 1.936 ms and the slowest core from 2.675 to 2.372 ms
(**0.30 ms upper bound**). After that, the bound is the five row-2 cores, where the NCRISC mover
carries a 2.3 ms big-tile item while its BRISC partner finishes at 1.82 ms.

## Reading

- **H2, not H1.** TRISC0 is busy only 52 % of mat on average (61 % at most), far from being the
  bound. The movers do wait on the done word: NCRISC 0.355 ms on average and 0.5-0.55 ms on the
  critical non-big-tile cores. The fill jobs are whole slabs and one queue serves both movers, so
  a mover waits whenever its slab's fill is queued behind the other mover's slab or is longer than
  its next read+sort.
- The 0.34 ms shortfall against t304's model is mostly that done-wait (about 0.36 ms on the
  critical NCRISC mover), plus TRISC fill+patch costing 1.11x the mover per record instead of
  0.83x.
- NCRISC waits about twice as long as BRISC (0.355 vs 0.188). This is unexplained: it could be
  `pick_job` order or NCRISC posting its job later in the item.
- Lever A: TRISC0 and TRISC1 idle is about 1.1 ms on average, but at least 0.88 ms on some cores. It
  is spread over about 10 pick waits per view, not one window. That is barely at the (c) threshold
  and too fragmented for a blend lane.

## Decision

| rule | condition | measured | holds |
|---|---|---|---|
| (a) fill rebalance | TRISC0 ≥ 85 % busy and done-wait ≥ 0.3 | TRISC0 52 % (max 61 %) | no |
| (b) finer fill jobs | done-wait ≥ 0.3 and TRISC0 < 85 % | NCRISC 0.355 mean, 0.5-0.55 on critical cores; TRISC0 52 % | **yes** |
| (c) reopen lever A | every TRISC idle ≥ 1.0 ms | TRISC0 1.11 / TRISC1 1.10 mean, min 0.88; fragmented | no (marginal) |
| (d) radix on TRISC1 model | TRISC0 fill cyc/rec ≤ mover's and TRISC1 idle ≥ 2 ms | 64.5 vs 75 (fill+patch), but TRISC1 idle 1.10 | no |

**Decision: (b) finer fill jobs.** Use per-512-record jobs, per-mover queues (or NCRISC-first
`pick_job`) and a streamed emit, so a mover waits only for the 512-record chunk it emits next.

Expected value: at most 0.30 ms of mat makespan (all done-wait removed). After that the
big-tile cores are the bound, so (b) only pays in full if the big-tile item is also split across
the two movers of its core (row-2 cores: NCRISC 2.67 vs BRISC 1.82 ms busy). In the fused
mat+blend stage, #306 turned about 40 % of its mat-side model into frame time, so the realistic
frame gain is about 0.1-0.2 ms (1-2 %). That is at the charter's ~1 % line, so check it with a
no-device model before building.
