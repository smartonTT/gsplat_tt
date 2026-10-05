# Lever C: dead-record pre-cull (task #140, code only)

Spec: lever C in `docs/reprofile-t115/README.md`. 19.7% of the per-pair records
(gaussian, tile) on bicycle have an all-zero microblock mask. They go through tile
assign, the sort and materialize for nothing.

## What it does

`GSPLAT_TT_PRECULL=1` (the default 0 is the kill switch) adds step 11.6 to the pfwc
compute kernel (`PFWC_PRECULL`, `pfwc_precull_tile` in
`render/kernels/compute/project_pfwc_compute.cpp`). It is live only with
`GSPLAT_TT_SFPU_VIS=1`. Before the visibility tests, it replaces the 3-sigma radii
with opacity-aware ones:

    t  = max(2 ln(op) + c0, 0),  c0 = 2 ln(1/floor) + 0.25   (host arg 65)
    r' = ceil(sqrt(t * a) + 1)   (x),  same with c (y)
    r' is used only if it is < r and the lane is "ok"

The lane is "ok" when a, b and c are finite, the condition test (ac - b^2)*64 >= ac
holds (no near-singular cov), and r <= rlim = min(max_radius, 4096) (host arg 66).
The tile rect, the tpg count, the aabb word and the TA K2 all come from r', so the
change needs nothing in gather, TA, sort or blend. RECHECK lanes still use the
original radii in the writer's exact_words, which is a superset and therefore safe.
The host sets c0 and rlim from `mb_contrib_floor` and `max_radius`, and turns the
cull off when the band cull is disabled.

## Why the image is the same (by construction)

1. The band cull keeps a microblock iff the ellipse m2 <= t_band, with
   t_band = 2 ln(q/65535 / floor) + 0.05, meets the microblock's pixel-centre box.
   That ellipse's x half-extent is sqrt(t_band * a). r' uses t >= t_band + 0.2, a
   margin that covers the opacity quantisation and the SFPU log/sqrt error, plus a
   1 px slack. So every tile with a kept microblock stays inside the r' rect.
   Only mask-0 records are removed.
2. A record's mask depends only on its (gaussian, tile). The surviving records keep
   their masks.
3. The per-tile sort is stable by depth, so the survivors keep their relative
   order, and the blend sees the same list. (This is an assumption about the sort.
   The device md5 check covers it.)
4. Radius invariants: r' is an integer, r' <= r, (r' > 0) == (r > 0), and there is
   no shrink when r > rlim, so (r' > max_radius) == (r > max_radius). The
   max_radius cull is unchanged.
5. The on-screen tests on r' can drop a whole gaussian. That only happens when its
   keep region misses every pixel centre on screen, so all of its records were
   dead anyway.

Host checks: `tests/unit/test_precull.cpp` is an fp32 lane model with the log and
sqrt pushed to the low side, checked against the band keep test in fp32 and in
double. It covers random and adversarial gaussians (needles, op at the floor,
op = 1, means on tile borders). Result: 0 kept tiles lost and 0 radius-invariant
violations. Without the margin and slack it loses about 7k tiles, so the test can
fail. The estimate below also loses 0 live records in all 30 views.

## Estimate (host model, `estimate.py`, all 30 bicycle views)

| | value |
|---|---|
| records (30 views) | 92,230,356 |
| dead (mask 0) | 19.77% |
| removed by the r' rect | 6.13% (about 31% of the dead) |
| live records lost | 0 |
| hero: max tile | 25,706 -> 24,595 |
| hero: tiles > 16384 | 13 -> 6 |
| rect + sheared row band (follow-up) | 9.92% removed, 0 live lost |

Expected savings: the stages that scale with records (TA, emit, radix, publish,
mat+cull, blend) total about 13.7 ms at tip a03bd1e. 6.1% of that is **about 0.84
ms/view gross, an upper bound**. The extra SFPU work in pfwc (a log, two sqrts and
some compares per gaussian) costs an estimated 0.15-0.3 ms, so the net is **about
0.5-0.7 ms/view (about 2%)**. That is below the t115 README's 1.5 ms figure, because
the rect alone reaches only a third of the dead records. The sheared row band would
reach about 10% (about 1.4 ms gross), but it changes the pfwc tpg counts and offsets.

## Risks (all need the device to check)

- Kernel size: PFWC_VIS was about 66 KB of the 70.6 KB kernel config buffer. Step
  11.6 is one non-inlined loop over the 32 vectors (no templates or unrolling) to
  keep it small. drive.sh prints the pfwc ELF sizes.
- sfpi register pressure in the log body, and whether the compiler accepts
  `#pragma GCC unroll 0`. Both are checked only by the device build (the syntax
  stub passes).
- The CB_TMP_RX/RY pop and re-push (both CBs are 2 tiles deep).

## Device A/B (ready to run)

    ttp detach t140-ab -- bash docs/precull-t140/drive.sh <SHA>

Remote tree `/localdev/smarton/gstt2-t140` on yyzo-bh-07. Each step runs under
`ttp lock p100`:
1. 3 interleaved rounds of `base` vs `precull:GSPLAT_TT_PRECULL=1`. Each round is
   30 views with md5 vs `md5-r82new.txt` (expect `ALL_VIEWS_IDENTICAL` in both
   arms) and ms/view from SUMMARY.
2. A `dc` run without and with the pre-cull: dead share 19.6% -> about 14.3% on
   the hero view, and about 6.1% fewer records.
3. An `hp` run of both arms: pfwc should grow by 0.15-0.3 ms, and TA, sort and
   blend should shrink.
4. pfwc ELF sizes.

Ship if the md5s are identical and the median ms/view gain is >= 0.3 ms (about 1%).
If a lever A or B rebase lands first, re-run on top of it. The change only touches
the radii, so it should rebase cleanly.
