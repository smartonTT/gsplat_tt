# Structural plan 4: what is left after the 2026-10-13 shelvings

Task #603, 2026-10-13 02:30 PDT. No device use; desk re-check of docs/structural-plan-3/PLAN.md
(ttp/t592 @3326d8be) against today's results, plus one CPU model (`model603.py`).
Headline metric: `render/run.py --back-to-back`, 30 bicycle views, median ms/view (b2b).

Base: best-iter-227, **8.451 ms/view b2b** (t595, bh-33 p100 11x10), md5 golden **077caaba**,
hero 42.85 dB vs `benchmarks/reference_v2/hero.png`.
Measured split (t595, docs/profile-t227/PROFILE.md): pfwc 1.52, sort 1.44, mat+blend incl.
folded K2 ~5.46, gaps ~0.03.

## Answer

The list is **not** exhausted. One cheap lever is open, with new measured evidence:
**RECFAST + VIS_FAST bundle** (two pfwc switches that already exist, default off).
Each was shelved alone under the 0.1 gate (0.071 and 0.051). The evidence shows each one was
capped by the bottleneck the other removes. Calibrated model: **+0.12 to +0.26 ms/view b2b** for
the pair. A second lever, spec A (pfwc survivor compaction), is reopened on top of it. t596's
own reopen condition (b) ("the pfwc mover floor drops first") is now met by t599's measured
record loop. It is gated on a repack microbench. Everything else in plan 3 is landed, shelved
or killed.

## Why the two pfwc switches are worth trying together (new evidence)

pfwc ends at `max(TRISC + tail, slowest mover + lead)` on each core.

- t599 RECFAST (record loop 117 -> 67 cyc/survivor, measured) cut mover busy by ~0.35 ms on the
  slowest core, but b2b only by 0.071. So after RECFAST, pfwc is bound by TRISC again.
- t488 VIS_FAST (vis+precull pass 25 -> 9 `v_if`s; that pass is 0.62 ms of pfwc TRISC, t484)
  gained only 0.051 (p100) / 0.068 (p150, t494). t598's model shows why: at that time a
  TRISC-free pfwc could gain at most ~0.06 because the movers were the floor.
- `model603.py` re-fits the t598 per-core model (t484 STEPCYC capture, 6600 core-launches) with
  two parameters calibrated on two iter-227 measurements: pfwc window 1.523 (t595) and the
  RECFAST gain 0.071 (t599). Result: lead 0.0375 ms, TRISC offset -0.010 ms. Slowest mover
  after RECFAST is 1.30 ms (was 1.64). The TRISC-free floor is 1.17 ms, 0.28 under today's
  window.
- Independent check: the same fit says that without RECFAST any TRISC cut gains at most 0.058
  (column 1 below). That matches VIS_FAST's measured 0.051 / 0.068 without being fitted to it.

Gain vs best-iter-227 for a TRISC cut dT (`model603-out.txt`):

| TRISC cut dT (ms) | without RECFAST | with RECFAST (RECFAST's 0.072 included) |
|---:|---:|---:|
| 0 | 0 | +0.072 |
| 0.05 | +0.017 | **+0.122** |
| 0.10 | +0.032 | +0.172 |
| 0.15 | +0.045 | +0.218 |
| 0.20 | +0.054 | +0.260 |

VIS_FAST's TRISC cut is not measured directly. Its measured 0.051 near the 0.058 cap suggests
dT >= ~0.1. The instruction count gives at most ~0.33 (0.62 ms pass, ~85% SFPU, 64% fewer
`v_if`s). The bundle clears 0.1 if dT >= ~0.04.

## Lever table (plan-3 levers plus today's results)

Sizes in ms/view b2b. "Bound" = model or profile ceiling; "realistic" = expected measured gain.

| rank | lever | status | bound | realistic | source |
|---|---|---|---|---|---|
| **1** | **RECFAST + VIS_FAST bundle** (`GSPLAT_TT_PFWC_RECFAST=1 GSPLAT_TT_PFWC_VIS_FAST=1`) | **open: reopened on new evidence** (each switch alone shelved) | 0.35 (TRISC cut 0.5) | **0.12-0.26** | t599 docs/pfwc-rec-loop-t599, t488 docs/pfwc-visfast-t488, t494, model603 |
| **2** | **A'. pfwc survivor compaction on top of rank 1** (`GSPLAT_TT_PFWC_COMPACT`) | **open: reopened** (t596 condition (b) met) | +0.36-0.45 over rank 1 (free repack) | +0.09-0.25 over rank 1 *if* repack <= 40 cyc/survivor split over both movers; <= 0 at >= 60 | t596 docs/pfwc-compact-t596 (gate 0.309 survival, 0 drops), model603 |
| - | B4 cross-frame overlap | shelved (#591) | 0.383 ceiling | +0.07 best design | docs/b4-overlap/RESULT.md |
| - | A. pfwc compaction without RECFAST | shelved (#596) | 0.18 (mover-capped) | < 0 at measured mover rates | docs/pfwc-compact-t596/README.md |
| - | B. mat fill jobs + big-tile split + mover cap | killed (#595 step 0) | 0.02-0.18 | < 0.1 | docs/profile-t227/PROFILE.md §3 |
| - | C. mat band cull on FPU | killed (#595 step 0) | 0.037 critical core (0.009 makespan) | ~0 | docs/profile-t227/PROFILE.md §3 |
| - | pfwc mover floor (batch writes + survivor deal) | shelved (#598) | 0.084-0.109 | ~0.06 | docs/pfwc-mover-t598/README.md |
| - | RECFAST alone | shelved (#599), **part of rank 1** | | 0.071 measured | docs/pfwc-rec-loop-t599/README.md |
| - | VIS_FAST alone | shelved (t488/t494), **part of rank 1** | <= 0.058 at 227 (model) | 0.05-0.07 measured | docs/pfwc-visfast-t488/README.md |
| - | pfwc byte cut (fp16 cov3d, packed rgb) | open, not ranked | 0 until pfwc nears its DRAM floor (~0.80 ms) | 0 today | plan 3 #5; only after ranks 1+2 bring pfwc near 0.9 |
| - | L4 persistent program | shelved (#579) | | code 2.4x kcfg | docs/persistent-step0/RESULT.md |
| - | precision variants (alpha floor, T_EPS, coarser T) | no-go (#580); degree-1 exp landed (iter 227) | | | docs/precision-model/RESULT.md |
| - | p150 120-core box as headline | not a code lever | | | plan 3 #6 |
| - | 2-chip AFR | measured (t582, bh-50): 4.226 ms/view on 2 chips; reported next to the headline | | | plan 3 |

### Stages with no new lever (checked)

- **sort (1.44 ms, 23% fixed, `sort_ol_emit` 1.22 / `sort_ol_town` 1.11 busy).** No profile
  supports a lever of 0.1 or more. The emit loop already has the register-locals fast path
  (t160). Coalesced emit was a measured no-go (t534: runs too short). Sort reuse across views
  saves 0 on the bench (t529). The t599 trick (loads in registers, grouped before the stores)
  is already in the radix code: `sort_radix_tile_algo.h` unrolls and groups its loads. Not
  ranked.
- **mat mover sort (0.85-0.88 ms per mover).** Already packed and unrolled (t418, 88-95 cyc/rec).
  The critical core is bound by NCRISC's big-tile sort (`fz_mv_big` max 1.91). That is B's
  "big items free" part (killed) and t175's split big-tile sort (shelved). Not ranked.
- **blend (3.51 mean core, SFPU).** FPU variants were measured losses (t111/t511/t148). Precision
  margin is 0.45 dB (t580). Lane packing <= 0.09 (t538). Not ranked.

After ranks 1 and 2, pfwc would be ~0.9-1.3 ms and near its mover/DRAM floor. The single chip
would then be at its known SFPU floor for blend. If both are shelved, the project's stop
conclusion holds: no single-chip lever of 0.1 ms or more is left that we know of.

## Spec 1 (rank 1): RECFAST + VIS_FAST bundle — device A/B

Ready to queue. No new kernel code: both switches are runtime JIT defines
(`render/host/env_config.h` `pfwc_recfast()`, `pfwc_vis_fast()`; `render/host/pfwc_device.cpp`).
VIS_FAST's code (`render/kernels/compute/project_pfwc_compute.cpp`) is unchanged since t488.

Binding rules (restated from the charter):
- Own git worktree of `~/dev/gstt2` on a new `ttp/<task>` branch from
  **ttp/t603-pick-the-next-device-lever-0-1-ms-view-b** (best-iter-227 + t599 RECFAST code,
  default off). Never edit `~/dev/gstt2` itself. No PRs, no force push, no rebase/amend of
  pushed commits, no tag or branch deletion or move; main moves forward only.
- Device: only the existing **bh-33 measurement reservation** (p100). Never `ird reserve`,
  extend or release, never use bh-30 or the viewer reservation. Run
  `"$TTP_PROJECT/harness/bin/ssh-preflight" bh-33` before any sync. Use
  `-o StrictHostKeyChecking=yes` on every ssh/rsync: the user's ssh config sets `no` for bh-*,
  so override it there, and never edit that file or known_hosts by hand. No scanning, no
  exploratory logins.
- Wrap every sync + build + run sequence in **`ttp lock p100 -- ...`**. One device run at a
  time. Long runs go through `ttp detach`, with no polling. bh-33's /home is full: set
  `TT_METAL_CACHE` off /home, with a separate cache per arm (as t599 did).
- Never search / or the home folder.

Steps:
0. Build once, on the branch head. Arms: **A** = defaults (`RECFAST=0 VIS_FAST=0`, = best-iter-227);
   **B** = `GSPLAT_TT_PFWC_RECFAST=1 GSPLAT_TT_PFWC_VIS_FAST=1`; **C** (attribution only, 1 round)
   = `GSPLAT_TT_PFWC_VIS_FAST=1`. Reuse docs/pfwc-rec-loop-t599/drive.sh + remote_ab.sh and
   change only the arm env.
1. **md5 first:** `--dump-views` over the 30 views for A and B. A must give **077caaba 30/30**
   (11x10). If B gives 077caaba 30/30, go to step 2.
   If B is off golden (t488 saw views 27/29 at 1 LSB, 2 and 15 pixels), RECFAST is not the
   cause: it was bit-exact at t599. Make VIS_FAST's predicate exact before measuring. Find the
   flipped lanes with the lane model (`tests/unit/test_vis_lever2.cpp`, `sfpu_model_fast` vs
   today's predicate) on the off views' pfwc inputs. Likely spots: `tz - KN` subtract-then-compare
   under SFPU denormal flush, the merged edge-tau test, and the sum-exponent RECHECK flag. Fix,
   re-run the unit test, rebuild, and repeat step 1. If it cannot be made exact within the task,
   still measure step 2, then hand off `needs_review` with the numbers. Do not land. Include the
   per-view diff (pixels, max LSB, PSNR vs arm A) so the coordinator can decide on a golden
   refresh, as done for iter 227.
2. **b2b A/B:** one warm-up round, then 4 alternating rounds (ABBA order) of
   `render/run.py --back-to-back` (1 check + 1 warm-up + 20 measured passes, 30 bicycle views
   1024x1024), arms A and B. One extra round of arm C for attribution.
   Headline = median b2b ms/view per round. Latency without dumps is secondary. Never use
   latency with `--dump-views` for keep or shelve.
3. Optional, only if the gain is < 0.1 but > 0.05: one pfwc program window per arm with the
   t595 tooling (`docs/profile-t227/drive2.sh`, `GSPLAT_TT_B4A_FOLD=0` so it fits kcfg) to see
   whether TRISC or the movers set the window. Do this before shelving.
4. **Kill rule:** keep only if the B - A median gain is **>= 0.1 ms/view b2b** on
   render/run.py --back-to-back over the 30 bicycle views, with the same sign in all 4 rounds,
   and md5 **077caaba 30/30 on bh-33** for arm B. Otherwise shelve, with the numbers and the
   step-3 profile if it was run.
5. **Screenshot step** (arm B, before any keep): device `hero.png` rendered on bh-33 at that
   commit and config (never the CPU reference). Diff image and PSNR against
   **`benchmarks/reference_v2/hero.png`** (gate >= 42.4 dB; 227 gives 42.85), plus the md5/golden
   badge as a separate item. Look at hero and diff by eye for tile seams or block artifacts and
   say what you saw. Save under `opt/metal-screenshots/ttw-<task>/`.
6. Keep path: make both switches default 1 (`env_config.h`; `GSPLAT_TT_PFWC_RECFAST=0` /
   `GSPLAT_TT_PFWC_VIS_FAST=0` remain kill switches). Re-run md5 077caaba 30/30 with defaults.
   Add the iteration to `opt/ttw/iters.jsonl` with hero/diff/PSNR. Rebuild `REPORT.html`; the
   validator must pass. GPU numbers are published (10.75 ms/view, INRIA RTX A6000), labeled
   published. Tag `best-iter-<N>`, push the tag, fast-forward main, run
   `python3 tt-project/harness/watchers/verify_main.py <N> --ff`, and report "main
   fast-forwarded" only on its OK line. Queue the viewer deploy + page-FPS task (resource
   'viewer').

## Spec 2 (rank 2): A'. pfwc survivor compaction on top of rank 1

Queue after spec 1 lands or is shelved. Its gain is counted over whatever is the best then.
Same binding rules as spec 1. Branch from the then-best (spec 1's branch if it landed).

0. **Re-size (CPU, no device):** if spec 1 measured, re-calibrate `model603.py` with spec 1's
   measured gain as a third point. Read the incremental gain of A' at 30/40/50 cyc per survivor
   (split over both movers). Kill if it is < 0.15 at 40 cyc.
1. **Repack microbench (device, 1 core, as t599 step 1):** a `p11`-style probe in
   `render/bench/kernels/` that walks a 1024-lane survivor mask (~28% set) and copies each
   survivor's 10 input words plus its lane index from tile-layout L1 pages into dense tiles.
   Use t599's register-cursor style (loads grouped before stores, cursors in registers).
   Report cycles per survivor on BRISC and NCRISC. **Kill if > 40 cyc/survivor** (model603:
   incremental <= +0.13 over a dT 0.15 bundle at 40, <= +0.07 at 50, < 0 at 67).
2. Build per docs/structural-plan-3/PLAN.md spec A step 1. Changes from plan 3: the repack is
   split over both movers (t596: NCRISC-only loses at every measured rate), and RECFAST stays on.
   The CPU gate (t596 `gate.py`, 0 drops over 30 views) is the phase-1 predicate. Must be
   bit-exact: per-lane math is the same ops in the same order, and survivors are written in
   today's order.
3. Constraints to report: pfwc static CB total vs L1 after staging (~80 KB), pfwc kernel code
   vs kcfg 105,472 B, BRISC stack margin >= 256 B.
4. Device A/B, md5 and screenshot steps exactly as spec 1 steps 1-2 and 5 (arm A = then-best
   defaults, arm B = `GSPLAT_TT_PFWC_COMPACT=1`). Add a pfwc program window and per-RISC busy
   capture for both arms (t595 tooling).
5. **Kill rule:** b2b gain >= **0.1 ms/view** over the then-best on render/run.py
   --back-to-back over the 30 bicycle views, same sign in all rounds, md5 **077caaba 30/30 on
   bh-33** (or the then-current golden, if spec 1 refreshed it with sign-off). If the pfwc
   window drops >= 0.15 but b2b does not, profile once before shelving.

## Files

- `model603.py`: the calibrated pfwc model. Run
  `git show origin/smarton/tt-project-opt:docs/pfwc-profile-t484/out/pfwc-sr.csv.gz > /tmp/sr.csv.gz`
  then `python3 docs/structural-plan-4/model603.py /tmp/sr.csv.gz`. It imports
  `docs/pfwc-mover-t598/model598.py` and `ana.py`.
- `model603-out.txt`: its output.

Caveat: model603 fits 2 parameters to 2 measurements, so it is calibrated, not validated. Its
one independent check is the VIS_FAST-alone cap (0.058 vs measured 0.051 at iter ~222, t488, and 0.068 on
p150, t494). Spec 1 is cheap (no code), so the device run is the real test.

## Sources

docs/structural-plan-3/PLAN.md (ttp/t592 @3326d8be); docs/profile-t227/PROFILE.md and
out/ana-P.txt (ttp/t595-profile-227); docs/pfwc-compact-t596/README.md
(ttp/t596-spec-a-…); docs/pfwc-mover-t598, docs/pfwc-rec-loop-t599 (ttp/t598 @e5bdb8b0);
docs/b4-overlap/RESULT.md (#591); docs/persistent-step0/RESULT.md (ttp/t579);
docs/precision-model/RESULT.md (ttp/t580); docs/pfwc-visfast-t488/README.md and commit 62dc8abc
(t494) on smarton/tt-project-opt; docs/pfwc-profile-t484 (t484 capture);
docs/mover-sortperm/README.md (t418); t160 fast emit, commit 087942a0 (t175).
