# t596: spec A (pfwc survivor compaction in L1, `GSPLAT_TT_PFWC_COMPACT`): gate passes, shelved before build

Spec: docs/structural-plan-3/PLAN.md, spec A (origin/ttp/t592-structural-plan-3 @3326d8be).
Base: best-iter-227 line. No device run, no kernel code: the mover budget below rules the
lever out before the build, from measured t484 costs. No ird reserve/extend/release, no ssh.

## Step 0: CPU gate (passes)

`gate.py` (repo root: `python3 docs/pfwc-compact-t596/gate.py <ply>`; output `gate-out.txt`).
B3-pruned bicycle (N 5,559,909), pfwc deal over 110 cores (bh-33 p100, 11x10, max 50 tiles per
core), 30 views of `benchmarks/cameras_v2.json`. Gate = z > 0.01 and the screen rect of a static
radius `3 sqrt(tr cov3d) f/z sqrt(1 + min(|t/z|, lim)^2) + 3 sqrt(0.3) + 1` meets the image
(conservative bound on pfwc's `ceil(3 sqrt(a))`; needs only the mean and the cov3d diagonal).

| | mean over views | max view |
|---|---:|---:|
| visible (#140/#169 model) | 0.306 | 0.377 |
| gate survival | **0.309** | 0.380 |
| gate survival, worst core | 0.368 | 0.443 |
| phase-2 tiles on the critical core / today's 50 | 0.377 | 0.460 |
| gaussians the gate drops that the model keeps | **0** (all 30 views) | |

Kill rule (mean survival > 0.6) not met. The gate is tight: 0.309 vs 0.306 visible.

## Why it is shelved anyway: the movers, not TRISC, set the window once TRISC drops

t484 (pfwc b2b profile, 110 cores, `docs/pfwc-profile-t484`) measured the two pfwc movers
(writer split: BRISC even chunks, NCRISC odd chunks and the reader):

- classify `cls` ~20.5k cycles per 1024-lane chunk (~20 cycles/lane), on every lane;
- survivor record scatter `rec` 112-117 cycles per survivor (~13 L1 words read and staged);
- busy 1.26 / 1.25 ms on the mean core; on the slowest core only 0.17-0.20 ms idle, so its
  movers are busy ~1.65 ms of the 1.83 ms window (TRISC 1.68 + 0.10 back-pressure + 0.05 tail).

Spec A cuts TRISC to about 0.34 (projection) + 0.05 (gate) + 0.377 x 1.34 (cov_cam, cov2d,
conic, vis+precull) ~ 0.9 ms. The window then sits on the slowest core's movers: ~1.65 ms + the
repack's net cost. So even a free repack gains at most ~0.18 ms, and the repack is not free.

Net mover cost per lane at core survival s (repack moves 10 input words + the lane index per
survivor; classify then runs on dense survivor lanes only, its full-lane scan replaced by the
repack's mask walk): `d = c_rep * s - c_cls * (1 - s)`, c_cls = 20, 51.2k lanes per core,
worst core s = 0.44. For a >= 0.1 ms gain the busiest mover may grow by <= 0.08 ms (~108k
cycles at 1.35 GHz):

| repack on | break-even c_rep (cycles / survivor) |
|---|---:|
| NCRISC only (as specced; the classify saving splits over both movers) | <= 17.5 |
| both movers, half each | <= 35 |

The movers' measured rate on this exact kind of work (`rec`: survivor lane words out of compute
tiles into staging) is ~8.6 cycles per word, i.e. ~95 cycles for an 11-word repack. That puts
NCRISC on the worst core at +0.9 ms (window ~2.5 ms, a loss of ~0.7) and the split variant at
+0.4 ms each (window ~2.05, a loss of ~0.2). Even a pure load/store loop (~2 cycles/word + ctz,
~30 cycles/survivor) only reaches the split break-even. The 0.1 ms keep gate needs a repack 3-5x
faster than any measured mover loop in this kernel, and the realistic outcome is a slower pfwc.

## Verdict

**Shelved before build.** CPU gate passes (mean 0.309, 0 drops), but the gain is capped by the
pfwc movers (~0.18 ms with a free repack) and the repack, at measured mover rates, costs more
than it saves. No code behind `GSPLAT_TT_PFWC_COMPACT`, no device run, nothing to land.

Reopen only with new evidence: (a) a measured repack <= ~30 cycles/survivor on a Blackhole
data-movement RISC, or (b) the pfwc mover floor drops first (t484 lever 3: cheaper record
scatter, chunks dealt to BRISC/NCRISC by running survivor count). Either raises A's ceiling;
after (b), recompute the table above with the new mover busy times.
