# t628/t639 L-A prefetch — result: keep (-0.22 ms/view b2b)

Lever: with `GSPLAT_TT_LA_PREFETCH=1`, the fold after-hook enqueues view N+2's early sort and
proj_M read right behind mat+blend(N+1) (CQ1, behind the K2-done event). Default off so far.

## Correction (review #635)

The first measurement (t628, commit 7cbf73fc/ed276bbe, 8.458 vs 8.456 ms/view, "rejected") was
invalid: the prefetch gate also required `env_config::host_free_mp_enabled()`, which is
`constexpr false`, so the prefetch never fired. Those runs had 0 `[LA] prefetch` log lines and no
la_wait zones; both arms measured the default path. Its numbers and conclusions are withdrawn.

t639 (commit 9c273710) drops the host-free requirement. On a pair overflow the la_wait path drains
CQ0/CQ1, sets `fold_ran` and re-enters the normal fused K2 path (pass 0 skips the folded K2, reads
the overflow, regrows, pass 1 reruns). Guards against measuring an inert lever again:
`remote.sh` fails a la/lah arm with 0 `[LA] prefetch` lines (LA_INERT), and `phases.py` fails a
lah file with no la_wait zone.

## Measurement (t639)

bh-33 (p100 lock), 2026-10-13 12:55-12:59 PDT, commit 9c273710 (main best-iter-227 + LA),
render/run.py --back-to-back, 30 bicycle views, 20 passes, interleaved rounds (r0 after a warm-up
run, then r1, r2 with HTL arms). Outputs: `out-t639/`.

| round | base ms/view | LA ms/view |
|---|---|---|
| r0 | 8.455 | 8.235 |
| r1 | 8.452 | 8.227 |
| r2 | 8.441 | 8.231 |
| mean | 8.449 | 8.231 |

Delta: **-0.218 ms/view** (118.4 -> 121.5 FPS), bar was >= 0.10. Every LA/LA+HTL arm logged
23 `[LA] prefetch` lines and 0 overflow lines. b2b raw_md5 1d372b94 in every arm (base and LA
identical). Sweep with LA on: 077caaba 30/30 (golden match, max 0 LSB), hero PSNR 42.85 dB vs
benchmarks/reference_v2/hero.png; hero.png and hero_diff10.png checked by eye: no tile seams or
artifacts (opt/metal-screenshots/ttw-t628/).

Phase split (phases.py, median over 600 views, HTL on; W = la_wait):

| run | A | A1 | B | C | D | G | W | sum |
|---|---|---|---|---|---|---|---|---|
| r1 HTL     | 0.263 | 0.138 | 1.198 | 0.146 | 6.893 | 0.055 | - | 8.554 |
| r1 LA+HTL  | 0.051 | - | 1.112 | 0.147 | 6.975 | 0.055 | 0.001 | 8.340 |
| r2 HTL     | 0.268 | 0.144 | 1.189 | 0.145 | 6.902 | 0.055 | - | 8.559 |
| r2 LA+HTL  | 0.051 | - | 1.113 | 0.145 | 6.980 | 0.053 | 0.001 | 8.344 |

Phase A (view start to gather result) falls 0.26 -> 0.05 ms: the gather for the next view is
already done when its turn comes (la_wait ~1 us), so the view-boundary dispatch gap t614 bounded
at <=0.30 ms was mostly real and the prefetch hides ~0.21 ms of it. D grows slightly (+0.08 ms)
as the device now does that work while the host waits on blend.

Decision: keep. Landing (default on, dashboard iteration, best-iter tag) follows re-review #640.
