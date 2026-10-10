# t488: pfwc vis+precull SFPU pass — GSPLAT_TT_PFWC_VIS_FAST

Lever: `GSPLAT_TT_PFWC_VIS_FAST=1` (default 0) rewrites the per-gaussian visibility and
pre-cull tests in `render/kernels/compute/project_pfwc_compute.cpp` as min/max reductions
with a single edge-tau test and a sum-exponent non-finite flag: 9 SFPU `v_if` blocks instead
of 25. Lane model and unit test: `tests/unit/test_vis_lever2.cpp`.
Target from #484 (docs/pfwc-profile-t484): the vis+precull pass is 0.621 of 1.679 ms pfwc TRISC time.

## Measurement

p100a (yyzo-bh-04, existing measurement reservation, every run under `ttp lock p100`), one build
(fc08a58), `render/run.py --back-to-back`, 30 bicycle views 1024x1024, alternating arms in one session.

| A/B | rounds | off (median b2b ms/view) | on | delta |
|---|---|---|---|---|
| VIS_FAST alone (SKIP_RGB=1 both arms) | r0-r3 | 9.348 / 9.346 / 9.352 / 9.348 | 9.297 / 9.296 / 9.300 / 9.296 | **-0.051** |
| Bundle: base vs SKIP_RGB=1 + VIS_FAST=1 | r20-r23 | 9.408 / 9.408 / 9.406 / 9.406 | 9.302 / 9.299 / 9.299 / 9.298 | -0.107 |

The bundle splits as SKIP_RGB -0.059 + VIS_FAST -0.051.

Correctness (`--dump-views`): off arms md5 906e0435 = 11x10 golden, 30/30. VIS_FAST on: 0a790b06.
Views 27 and 29 differ by 1 LSB in 2 and 15 pixels (PSNR vs off 110 / 101 dB). The other 28 views,
hero included, are bit-identical. Hero on device: md5 86524912, 42.51 dB vs
benchmarks/reference_v2/hero.png (`out/hero-bun.png`, diff `out/hero-on-diff.png`). Looked at both: no tile seams.

## Decision: not kept as an iteration

The bundle clears the 0.1 ms gate only because of SKIP_RGB, and #489 (e933dafe on
ttp/t481-shrink-pfwc-output-fuse-gather-into-pfwc) already turned SKIP_RGB on by default as part of its
ACQ_FUSE=3 bundle. Crediting it again would count the same 0.06 ms twice. VIS_FAST alone saves 0.051 ms,
under the gate. It stays as an opt-in switch, default off, to bundle with the next pfwc lever.

## What this says about pfwc

Cutting the SFPU branch count of the 0.62 ms pass by 64% saved 0.05 ms of b2b. With the #481 result
(19% fewer reads, 0.06 ms), pfwc's b2b cost (project_gather_wait ~2.3 ms) is not set by this pass's
instruction count or by read bytes. Next things to measure: per-core pfwc finish skew (survivor-heavy cores,
writer back-pressure 0.10 ms in #484) and whether the gather wait is bound by the slowest core rather than
the average TRISC time.

Files: `drive.sh`, `remote_ab.sh` (driver), `out/` (raw logs, md5 lists, hero, views 27/29 from both arms).
