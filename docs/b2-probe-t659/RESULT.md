# t666 (data from #659): B2 ellipse-cull probe vs best-iter-229, back-to-back

Written 2026-10-13 19:35 PDT. Branch ttp/t666-b2-probe-at-229 = best-iter-229 (92959f82 + 3702fbb5)
plus #633's probe wiring (GSPLAT_TT_ELLIPSE_PROBE reaches the B4a-folded K2 in matblend_brisc),
#656's opt/ab_jit_check.py and #659's drivers. Device run: #659's detached A/B on bh-33 (p100,
under `ttp lock p100`), 16:51-17:14 PDT, commit a285bb36 (same tree as this branch's probe
commits), rc=0. It completed after #659 was cancelled; #666 reused it instead of re-running.

What the arms are: B2 itself (dropping dead pairs) was never built. The probe arm runs #538's exact
per-pair ellipse-tile test in K2 with pairs unchanged, so it measures the test's cost, a lower
bound on the lever's K2 cost. off = defaults (best-iter-229 kernels); aa = defaults again with its
own JIT cache (same-day A/A).

## Back-to-back (headline), ms/view, 30 views x 20 passes per run

| round | off | probe | aa |
|---|---|---|---|
| r0 | 8.065 | 8.073 | 8.067 |
| r1 | 8.064 | 8.080 | 8.064 |
| r2 | 8.064 | 8.077 | 8.067 |
| r3 | 8.062 | 8.078 | 8.074 |
| r4 | 8.062 | 8.078 | 8.068 |
| mean r1-r4 | **8.063** | **8.078** | 8.068 |
| spread r1-r4 | 0.002 | 0.003 | 0.010 |

Order rotated per round (off/probe/aa, probe/aa/off, aa/off/probe, probe/off/aa).
probe - off = **+0.015 ms/view** (r1-r4; +0.014 over r0-r4). A/A (aa - off) = +0.005, so the cost
is just beyond the same-day noise and far under the 0.1 ms/view bar. off matches iter 229's
recorded 8.063.

Latency (secondary, no --dump-views): off 8.0, probe 8.0 ms/view (avg_frame_ms, 1 run each).

## JIT check (opt/ab_jit_check.py, out/jit_check.txt)

off vs probe: JIT-HASH OK (matblend_brisc, k2_trisc_compute, tile_assign_scatter_seg all differ;
the other 12 kernels match). aa vs probe: OK. off vs aa: A/A on all 15 kernels, as designed.
The A/B is valid.

## Correctness

raw b2b md5 1d372b94 in every run of all three arms (same as #655's iter-229 arms). Probe-arm
30-view sweep md5 077caaba, 0 of 30 views differ (= iter 229's 077caaba 30/30). Hero (device,
probe arm): opt/metal-screenshots/t659-b2probe/hero.png, diff hero_diff10.png; PSNR 42.85 dB vs
benchmarks/reference_v2/hero.png (the reference render), golden match, max 0 LSB. Visual check
(#666, looked at both images): no tile seams or blocky artifacts; the diff shows only the usual
edge detail (spokes, foliage).

## Verdict: shelve B2 (again), with a new bound

The test's cost fell from +0.116 ms/view (#538, iter 225) to +0.015 at iter 229: K2 now runs
folded into matblend_brisc (B4a) and mostly off the critical path. So the test cost no longer
decides B2. The gross gain does: dead pairs are 6.8 % of pairs (#533 T-A) and blend already skips
them, so only per-record work outside blend shrinks. At iter 229 that is sort (0.930 ms/view in
the probe-arm shot's stage split, bin_emit 0.507 of it) plus mat 0.111; even if all of it scaled
with pair count, 6.8 % is about 0.07 ms/view, minus the +0.015 test and the conic stream the
real lever needs. Net upper bound about 0.05 ms/view, under the 0.1 keep bar, for days of work.
Re-open only if non-blend pair work grows a lot, or for #538's SFPU row-span variant that also
cuts K2 slots.
