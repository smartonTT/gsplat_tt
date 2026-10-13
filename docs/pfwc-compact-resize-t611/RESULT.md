# t611: next lever after the RECFAST/VIS_FAST shelving — lever list exhausted

Task #611, 2026-10-13 09:20-10:00 PDT. CPU only, no device run (spec step 1: none qualifies).
Base: best-iter-227, 8.451 ms/view b2b (t595), md5 golden 077caaba (bh-33 p100).

## Lever picked: plan-4 spec 2, A' (pfwc survivor compaction on top of RECFAST)

It is the only open lever left in docs/structural-plan-4/PLAN.md (copied here from
ttp/t603 @bf425d81). Its spec has a step 0 on the CPU: re-calibrate the pfwc model with
spec 1's measurement and kill A' if its gain over the then-best is < 0.15 ms/view at a
repack cost of 40 cyc/survivor.

## Step 0 re-size (`model611.py`, outputs `model611-out.txt`, `model611-M3-out.txt`)

Measurements used (all bh-33, best-iter-227, b2b ms/view):

| arm | b2b gain (t599/t606) | pfwc window, profiler, FOLD=0 (t608) |
|---|---:|---:|
| base | 0 (8.440) | 1.466 |
| RECFAST | 0.071 | 1.432 (-0.034) |
| VIS_FAST | 0.022 | 1.416 (-0.050) |
| RECFAST + VIS_FAST | 0.070 | 1.389 (-0.077) |

The bundle shows the key fact: VIS_FAST still cuts TRISC on top of RECFAST (profiler:
slowest-core TRISC busy 1.376 -> 1.337, window -0.043), but b2b gains nothing from it
(0.070 vs 0.071). model603 (2 parameters, 2 points) predicted +0.12..0.26 for the bundle.

Refit of the same per-core model with all four b2b points:

| model | params | fit error | A' gain over RECFAST at 30 / 40 / 50 cyc |
|---|---|---:|---|
| M1 no floor (model603 form) | L, o, dV fitted | 0.024 | +0.351 / +0.283 / +0.214 |
| M2 window floor F | L, o, dV, F fitted | 0.024 | +0.161 / +0.148 / +0.122 (lowest F allowed) |
| **M3 floor F, dV fixed to the measured TRISC cut (0.04 / 0.05)** | L, o, F fitted | 0.023 | **+0.000 / +0.000 / -0.002** |

- M1 and M2 fit only by setting VIS_FAST's TRISC cut dV to 0, which the t608 profile
  contradicts. The floor in M2 is not pinned by the data: any F from 1.29 to 1.45 fits. Even
  its most favorable value gives 0.148 at 40 cyc, under the 0.15 kill line.
- M3 uses the measured TRISC cut. It fits all four points as well, but only with a floor at
  F = 1.452 ms, which is the window after RECFAST. Every further pfwc TRISC cut, A' included,
  then gains 0 in b2b. A' gets its gain from a TRISC cut (slowest core 1.41 -> 0.79 ms) and adds
  repack work on the movers.

**Verdict: A' is killed at step 0** (< 0.15 at 40 cyc in every form consistent with the data;
~0 in the form that uses the measured TRISC cut). No branch code, no device run.

## Ranked list and why each lever is out (sizes ms/view b2b)

| lever | status | why out |
|---|---|---|
| RECFAST + VIS_FAST bundle (plan-4 rank 1) | shelved #606 | 0.070 measured, md5 off golden (VIS_FAST) |
| A' compaction on RECFAST (rank 2) | **killed here** | step 0: <= 0.148 at 40 cyc, ~0 with the measured TRISC cut |
| RECFAST alone | shelved #599 | 0.071 measured |
| VIS_FAST alone | shelved t488/t494/#606 | 0.022-0.068 measured |
| A. compaction without RECFAST | shelved #596 | < 0 at measured mover rates |
| pfwc mover floor (batch writes, survivor deal) | shelved #598 | 0.084-0.109 bound, ~0.06 realistic; and M3 says pfwc cuts beyond RECFAST convert to ~0 |
| pfwc byte cut (fp16 cov3d, packed rgb) | open, unranked | 0 until pfwc nears its DRAM floor |
| B4 cross-frame overlap | shelved #591 | +0.07 best design |
| B. mat fill jobs + big split; C. mat band cull on FPU | killed #595 | < 0.1 / ~0 |
| L4 persistent program | shelved #579 | kcfg 2.4x |
| precision variants | no-go #580 (degree-1 exp landed) | 0.45 dB margin |
| sort / mat mover sort / blend | no lever (plan-4 "stages checked") | no profile supports >= 0.1 |
| 2-chip AFR | measured (t582) | not a single-chip lever |

So: **lever list exhausted** for single-chip levers of >= 0.1 ms/view b2b that we know of.

## Open question worth one profile (not a lever yet)

b2b gains do not follow the profiled pfwc window: RECFAST gains 0.071 b2b for a 0.034 window cut,
VIS_FAST gains 0.022 for 0.050, and VIS_FAST on top of RECFAST gains 0 for 0.043. The t608
profile runs with FOLD=0 (profiler kcfg limit), while b2b runs with the B4a fold on. A device
timeline of the b2b configuration (FOLD=1, pfwc and its neighbours) would show what sets the
~1.45 ms floor M3 needs. That could be a DRAM/NoC contention with the folded K2, a dispatch
wait, or something outside pfwc. It is the only place a hidden >= 0.1 lever could still be.
