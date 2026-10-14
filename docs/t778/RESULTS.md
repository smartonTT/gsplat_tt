# t778: train-flag cost split, dataset-view forward, T2 per-core planes

Measurement only, no code change. Device: bh-49 p150a (measurement box, `ttp lock p100`),
synced with the project's repo-scoped rsync at d1d50d3d, run 2026-10-14 13:36-13:50 PDT.
Bicycle 1237x822, full model. b2b = `--back-to-back`, 20 passes x 30 views, 4 rounds in
rotated arm order; numbers are the median of the 4 round medians (spread <= 0.013 ms).
Raw logs: `out.tar.gz` (untar next to `analyze.py`); summary: `analyze_out.md`.

## 1. Train-flag cost split (orbit `cameras_v2`, ms/view)

Each rung adds one env knob to the previous one; `tr` is `GSPLAT_TT_BLEND_TRAIN=1`.

| rung | knobs | ms/view | step | what the step costs |
|---|---|---:|---:|---|
| rnd | render default | 7.299 | | |
| unf | `MATBLEND_FUSE=0` | 8.777 | +1.479 | unfused program: blend +0.85, projection gather wait exposed +0.68, sort +0.10, no cross-view overlap (xview 0.15 -> 0) |
| s0 | + `BLEND_SCHED=0` | 10.062 | +1.285 | blend scheduler off (all in blend) |
| tp0 | + `BLEND_T_PERIOD=0u` | 10.861 | +0.798 | no periodic transmittance early-out (all in blend) |
| tr | `BLEND_TRAIN=1` | 14.705 | +3.845 | accurate exp + fp32 opacity/colour + fp32 RGBT output (all in blend) |
| **total** | | | **+7.406** | |

Dataset views (`cameras_train_nn`, first 30 training cameras): rnd 7.896, tr 13.857
(+5.961). The t773 figure of +5.52 ms was on a different view set; on the orbit the cost is +7.41.

JIT gate (`opt/ab_jit_check.py`): unf/s0, s0/tp0, tp0/tr each show distinct
`alpha_blend_compute_mb` hashes (OK). rnd vs unf and rnd vs tr print MISSING/FAIL because
the fused render path runs `matblend_compute` and the unfused/train paths run
`alpha_blend_compute_mb`; there is no shared kernel to compare. That confirms the arms ran
different programs; it is not an A/A run. The driver's rc=1 comes only from these two checks.

## 2. Per-view timing, train flag, 30 dataset views (latency mode)

| stage | min | median | max |
|---|---:|---:|---:|
| wall | 9.60 | 14.46 | 18.49 |
| blend | 8.58 | 13.28 | 17.14 |
| project (gather wait) | 0.44 | 0.59 | 2.02 |
| sort | 0.50 | 0.52 | 0.77 |
| host gaps (b2b `gap_ms`, `py_resid`) | | 0.000, 0.035 | |

Blend is 90 % of a forward view and varies 2x across views.

**Gap: no dump-free training step loop.** Loss, T1, T2 and optimizer cannot be timed per view
over 30 views without new integration code: the T1/T2 device inputs are built only by Python
prep from forward dumps. Only the forward was measured per view here.

## 3. T2 per-core rank planes (Mac model of the device's strided 16-group block split)

1,809,729 visible, 28,288 groups, 1,768 blocks, 972,964 planes (`t2_core_planes.json`).

| split | busiest core | mean | min | mean / busiest |
|---|---:|---:|---:|---:|
| device strided blocks | 30,746 | 8,845 | 1,497 | 0.29 |
| t773 model (contiguous) | 34,292 | 8,845 | 1,438 | 0.26 |
| strided, capped at k=16 | 2,182 | 1,741 | 1,493 | 0.80 |

The strided split is only slightly better balanced than t773 assumed; heavy groups dominate.
Capping per-group planes at 16 would cut the busiest core 14x.
