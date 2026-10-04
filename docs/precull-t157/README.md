# Task #157: a tighter pre-cull rect (GSPLAT_TT_PRECULL=2)

Host analysis and a prototype only. No device runs. All shares below come from the
float64 host model (`analyze.py`, the t140 projection). The model is calibrated
against the device: it says lever C as built drops 132,838 records on the hero view,
and the t142 device dump dropped 132,086. **The ms figures are estimates, not
measurements.**

## Why lever C (t142) misses about 80% of the dead records

Lever C shrinks each gaussian's tile rect to the opacity-aware radius
r' = rint(sqrt(t cov) + 2). It then uses the usual tile-cell formula
floor((m ± r') / 32), which tests the rect against tile *edges*. The band cull,
however, tests against *pixel centres*, which sit 0.5 px inside each tile edge.
Two sources of slack follow:

* +1.5 to 2.5 px per side from the integer rounding (+2, then rint).
* +0.5 px per side because tile edges are used instead of pixel centres.

Most dead records that lever C keeps are just past the true extent. Hero view, of
the dead records dev leaves:

* 18.5% sit inside the keep ellipse's own bbox. These are true corners, and only an
  ellipse test would remove them.
* 42.5% are in 1-tile-wide strips.
* 8.6% belong to gaussians the shrink skips (not-ok: 64·det < a·c, or r > 4096).

## The fix: a pixel-centre rect (PRECULL_PC)

A tile's pixel centres [32k + 0.5, 32k + 31.5] meet [m − e, m + e] if and only if
the tile meets [m − (e − 0.5), m + (e − 0.5)] under the existing cell formula. So:

    r' = max(sqrt(t cov) − 0.5 + 1/8, 1/8)        (fractional, 1/8 px slack)

This keeps exactly the tiles whose pixel centres can be reached, plus 1/8 px. In the
kernel it is one SFPMAD and an SFPSWAP (`vec_min_max`), replacing the two adds of
the 2^23 rounding. It should be code-size neutral, but a device build must confirm
that the fused pfwc ELF stays within 70656 B (t142 had 16 B left).

Why it is safe (the same image as mode 1 and as base, up to lever C's ≤1 LSB from
T-checkpoint shifts):

* r' ≥ 1/8 > 0, so (r' > 0) == (r > 0).
* The max_radius cull is unchanged, because there is no shrink when r > rlim.
* The on-screen tests fail only when no pixel centre is reached.
* RECHECK lanes use the original radii, which is a superset.
* The fp32 ulp at 1024 px (~1.2e-4) is far below the 1/8 px slack.
* `tests/unit/test_precull.cpp` checks 150k random and adversarial gaussians
  against the band cull in fp32 and in double. Pixel-centre rect: 0 kept tiles
  lost, 0 bad radii. It has teeth: at slack −1/4 px it loses 647 kept tiles.

Knob: `GSPLAT_TT_PRECULL=2` (vis_mode.h) sets `PFWC_PRECULL` + `PRECULL_PC`.
Mode 1 is still the t142 integer rect, kept for A/B runs.

## Arms (host model; records removed, share of all records / of dead; live lost)

Hero view: 3,375,678 records, 19.62% dead.

| arm | what | removed | of dead | live lost |
|---|---|---|---|---|
| dev | lever C as built (t142) | 3.94% | 20.1% | 0 |
| ideal | exact log, ceil(sqrt + 1) | 4.86% | 24.8% | 0 |
| **pc** | **pixel-centre rect (this task)** | **13.96%** | **71.2%** | **0** |
| para | dev + per-row sheared band | 7.05% | 35.9% | 0 |
| row | dev + exact per-row ellipse test | 17.62% | 89.8% | 0 |
| pc+para | | 15.77% | 80.4% | 0 |
| pc+row | | 17.62% | 89.8% | 0 |

All 30 views: see `out/analyze-all.txt` (TOTAL line). The per-view shares are
within ±0.3 points of the hero view's.

30 views (92.2M records, 19.77% dead): dev 4.02% (20.4% of dead), ideal 4.89%,
**pc 13.84% (70.0% of dead)**, para 7.22%, row 17.64%, pc+para 15.76%, pc+row
17.64%. No live record lost in any arm. Max records on one tile: dev 25,019,
pc 22,976.

After pc, the dead records left (hero: 191,024) are 20% 1-wide strips and 23.8%
not-ok gaussians (needles that fail the cond-64 check). The rest are corners.

## What else is possible (not built)

* **Per-row ellipse test in TA K2** (row arm): +3.7 points beyond pc. It would run
  on 47% of gaussians (1.38M rows, 68.6% of pairs) and needs integer math on the
  data movers, which have no FPU (soft float costs ~85-90 cycles per op). The extra
  cut is small for that cost. Worth trying only if a device A/B shows the
  post-cull stages still scale with record count after pc.
* **Relaxing the cond-64 gate**: 24% of the dead records left after pc. A wider
  margin on t for ill-conditioned gaussians could cover them; it needs its own
  error analysis.

## Estimate (not measured)

On the device, lever C's 3.9% record cut saved about 0.91 ms/view gross:
mat_cull_mask −0.61, emit −0.13, count −0.05, TA scatter −0.05, blend −0.07.
pc removes 2.55× as many records (471k vs 133k on the hero view).

If those stages scale linearly with records, pc saves at most about **2.3 ms/view
more than lever C as built**, i.e. 19.33 → ~17.0 ms/view. The pfwc cost should be
the same or slightly lower. This is an upper bound. The count/emit/TA costs carry
fixed per-tile overheads, so a realistic range is 1.2 to 2.3 ms. Confirm it with a
device A/B before making any claim.
